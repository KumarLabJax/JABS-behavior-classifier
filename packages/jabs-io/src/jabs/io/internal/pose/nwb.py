"""NWB adapter for PoseData using ndx-pose."""

import datetime
import json
import logging
import re
import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt
from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.core import ScratchData

from jabs.core.enums import StorageFormat
from jabs.core.types import PoseData
from jabs.io.base import Adapter
from jabs.io.registry import register_adapter

logger = logging.getLogger(__name__)

_JABS_NWB_FORMAT_VERSION = 1
_JABS_METADATA_KEY = "jabs_metadata"
_IDENTITY_MASK_KEY = "jabs_identity_mask"
_BOUNDING_BOXES_KEY = "jabs_bounding_boxes"
_PROCESSING_MODULE_NAME = "behavior"
_PROCESSING_MODULE_DESC = "JABS pose estimation data"
_SKELETON_NAME = "subject"
_REFERENCE_FRAME = "Top-left corner of video frame, x increases rightward, y increases downward"
_CONFIDENCE_DEFINITION = (
    "Validity mask from JABS: 0.0=invalid/missing keypoint, >0.0=valid keypoint"
)


@register_adapter(StorageFormat.NWB, PoseData, priority=10)
class PoseNWBAdapter(Adapter):
    """NWB adapter for PoseData."""

    @classmethod
    def can_handle(cls, data_type):  # noqa: D102
        return data_type is PoseData

    def write(self, data: PoseData, path: str | Path, **kwargs) -> None:
        """Write PoseData to NWB file(s).

        Args:
            data: The PoseData to write.
            path: Output file path (.nwb).
            **kwargs:
                per_identity_files: If True, write one NWB file per identity.
                    Default False.
                session_description: NWB session description.
                session_start_time: NWB session start time.
                identifier: NWB file identifier string.
                skeleton_name: Name for the Skeleton object.
        """
        path = Path(path)
        per_identity_files = kwargs.get("per_identity_files", False)

        if per_identity_files:
            self._write_per_identity(data, path, **kwargs)
        else:
            self._write_single_file(data, path, **kwargs)

    def read(self, path: str | Path, data_type: type | None = None) -> PoseData:
        """Read PoseData from an NWB file.

        If the file was written with ``per_identity_files=True``, sibling
        identity files are auto-detected and merged.
        """
        path = Path(path)
        pose_data, jabs_meta = self._read_single(path)

        if jabs_meta.get("per_identity_files", False):
            return self._read_merged(path, jabs_meta)

        return pose_data

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _write_single_file(self, data: PoseData, path: Path, **kwargs) -> None:
        num_identities = data.points.shape[0]
        identity_names = [self._identity_name(data, i) for i in range(num_identities)]
        if len(set(identity_names)) != len(identity_names):
            raise ValueError(f"Identity names are not unique after sanitization: {identity_names}")

        nwbfile = self._make_nwb_file(**kwargs)
        skeleton = self._make_skeleton(data.body_parts, data.edges, **kwargs)
        static_skeletons = self._build_static_skeletons(data.static_objects)
        skeletons = Skeletons(skeletons=[skeleton, *static_skeletons.values()])

        behavior = nwbfile.create_processing_module(  # type: ignore[attr-defined]
            name=_PROCESSING_MODULE_NAME,
            description=_PROCESSING_MODULE_DESC,
        )
        behavior.add(skeletons)

        for i, name in enumerate(identity_names):
            pe = self._build_pose_estimation(
                name=name,
                points=data.points[i],
                point_mask=data.point_mask[i],
                body_parts=data.body_parts,
                fps=data.fps,
                skeleton=skeleton,
            )
            behavior.add(pe)

        for obj_name, obj_skeleton in static_skeletons.items():
            behavior.add(
                self._build_static_object_pose_estimation(
                    obj_name, data.static_objects[obj_name], obj_skeleton
                )
            )

        behavior.add(
            TimeSeries(
                name=_IDENTITY_MASK_KEY,
                data=data.identity_mask.T.astype(np.uint8),  # (num_frames, num_identities)
                unit="bool",
                rate=float(data.fps),
            )
        )

        if data.bounding_boxes is not None:
            behavior.add(
                TimeSeries(
                    name=_BOUNDING_BOXES_KEY,
                    data=np.transpose(
                        data.bounding_boxes, (1, 0, 2, 3)
                    ),  # (num_frames, num_identities, 2, 2)
                    unit="pixels",
                    rate=float(data.fps),
                )
            )

        jabs_meta = self._build_jabs_metadata(data, identity_names)
        nwbfile.add_scratch(
            ScratchData(
                name=_JABS_METADATA_KEY,
                data=json.dumps(jabs_meta, default=_json_default),
                description="JABS-specific metadata for lossless PoseData roundtrip",
            )
        )

        with NWBHDF5IO(str(path), mode="w") as io:
            io.write(nwbfile)

    def _write_per_identity(self, data: PoseData, path: Path, **kwargs) -> None:
        num_identities = data.points.shape[0]
        all_names = [self._identity_name(data, i) for i in range(num_identities)]
        if len(set(all_names)) != len(all_names):
            raise ValueError(f"Identity names are not unique after sanitization: {all_names}")

        for i in range(num_identities):
            identity_name = self._identity_name(data, i)
            identity_path = self._identity_file_path(path, identity_name)

            nwbfile = self._make_nwb_file(**kwargs)
            skeleton = self._make_skeleton(data.body_parts, data.edges, **kwargs)
            # Rebuild static skeletons each iteration: HDMF objects can only
            # belong to one container, so they cannot be shared across files.
            static_skeletons = self._build_static_skeletons(data.static_objects)
            skeletons = Skeletons(skeletons=[skeleton, *static_skeletons.values()])

            behavior = nwbfile.create_processing_module(  # type: ignore[attr-defined]
                name=_PROCESSING_MODULE_NAME,
                description=_PROCESSING_MODULE_DESC,
            )
            behavior.add(skeletons)

            pe = self._build_pose_estimation(
                name=identity_name,
                points=data.points[i],
                point_mask=data.point_mask[i],
                body_parts=data.body_parts,
                fps=data.fps,
                skeleton=skeleton,
            )
            behavior.add(pe)

            for obj_name, obj_skeleton in static_skeletons.items():
                behavior.add(
                    self._build_static_object_pose_estimation(
                        obj_name, data.static_objects[obj_name], obj_skeleton
                    )
                )

            behavior.add(
                TimeSeries(
                    name=_IDENTITY_MASK_KEY,
                    data=data.identity_mask[i].astype(np.uint8),
                    unit="bool",
                    rate=float(data.fps),
                )
            )

            if data.bounding_boxes is not None:
                behavior.add(
                    TimeSeries(
                        name=_BOUNDING_BOXES_KEY,
                        data=data.bounding_boxes[i],
                        unit="pixels",
                        rate=float(data.fps),
                    )
                )

            jabs_meta = self._build_jabs_metadata(
                data,
                [identity_name],
                per_identity_files=True,
                source_identity_index=i,
                total_identities=num_identities,
            )
            nwbfile.add_scratch(
                ScratchData(
                    name=_JABS_METADATA_KEY,
                    data=json.dumps(jabs_meta, default=_json_default),
                    description="JABS-specific metadata for lossless PoseData roundtrip",
                )
            )

            with NWBHDF5IO(str(identity_path), mode="w") as io:
                io.write(nwbfile)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def _read_single(self, path: Path) -> tuple[PoseData, dict]:
        """Read a single NWB file and return (PoseData, jabs_metadata_dict)."""
        with NWBHDF5IO(str(path), mode="r") as io:
            nwbfile = io.read()
            behavior = nwbfile.processing[_PROCESSING_MODULE_NAME]

            # Parse JABS metadata
            jabs_meta_raw = str(nwbfile.scratch[_JABS_METADATA_KEY].data)
            jabs_meta = json.loads(jabs_meta_raw)

            # Discover PoseEstimation containers (skip Skeletons, TimeSeries)
            identity_names = jabs_meta.get("identity_names", [])
            identity_names_set = set(identity_names)
            pe_containers = {
                name: obj
                for name, obj in behavior.data_interfaces.items()
                if isinstance(obj, PoseEstimation)
            }

            if not pe_containers:
                raise ValueError(
                    f"No PoseEstimation containers found in behavior module of {path}"
                )

            # Separate animal identity containers from static object containers
            identity_pe_containers = {
                name: obj for name, obj in pe_containers.items() if name in identity_names_set
            }

            # Order containers by identity_names if available.
            # Fall back to sorted keys if identity_names is absent or none of
            # its entries match the containers found in the file.
            ordered_names = [n for n in identity_names if n in identity_pe_containers]
            if not ordered_names:
                ordered_names = sorted(identity_pe_containers.keys())

            # Extract body_parts in the original written order from jabs_metadata.
            # pynwb returns PoseEstimationSeries alphabetically from HDF5, so we
            # cannot rely on pose_estimation_series.values() for ordering.
            first_pe = identity_pe_containers[ordered_names[0]]
            stored_body_parts = jabs_meta.get("body_parts")
            if stored_body_parts:
                body_parts = stored_body_parts
            else:
                body_parts = [series.name for series in first_pe.pose_estimation_series.values()]
            num_keypoints = len(body_parts)
            fps_value = first_pe.pose_estimation_series[body_parts[0]].rate

            # Recover skeleton edges
            skeleton_obj = first_pe.skeleton
            if skeleton_obj is not None and skeleton_obj.edges is not None:
                raw_edges = np.array(skeleton_obj.edges[:])
                edges = [tuple(row) for row in raw_edges.tolist()]
            else:
                edges = []

            all_points = []
            all_point_mask = []

            for pe_name in ordered_names:
                pe = identity_pe_containers[pe_name]
                identity_points = np.empty(
                    (len(pe.pose_estimation_series[body_parts[0]].data), num_keypoints, 2)
                )
                identity_mask = np.empty(
                    (len(pe.pose_estimation_series[body_parts[0]].data), num_keypoints),
                    dtype=bool,
                )
                for j, bp in enumerate(body_parts):
                    series = pe.pose_estimation_series[bp]
                    identity_points[:, j, :] = np.array(series.data[:])
                    identity_mask[:, j] = np.array(series.confidence[:]) > 0.0

                all_points.append(identity_points)
                all_point_mask.append(identity_mask)

            points = np.stack(all_points, axis=0)
            point_mask = np.stack(all_point_mask, axis=0)

            # Identity mask
            # Per-identity files store (num_frames,); single files store (num_frames, num_identities).
            # Both need to be returned as (num_identities, num_frames).
            identity_mask_ts = behavior[_IDENTITY_MASK_KEY]
            identity_mask = np.array(identity_mask_ts.data[:]).astype(bool)
            if identity_mask.ndim == 1:
                identity_mask = identity_mask[np.newaxis, :]  # (1, num_frames)
            else:
                identity_mask = identity_mask.T  # (num_identities, num_frames)

            # Bounding boxes
            # Per-identity files store (num_frames, 2, 2); single files store (num_frames, num_identities, 2, 2).
            # Both need to be returned as (num_identities, num_frames, 2, 2).
            bounding_boxes = None
            if _BOUNDING_BOXES_KEY in behavior.data_interfaces:
                bb_ts = behavior[_BOUNDING_BOXES_KEY]
                bounding_boxes = np.array(bb_ts.data[:])
                if bounding_boxes.ndim == 3:
                    bounding_boxes = bounding_boxes[np.newaxis, :]  # (1, num_frames, 2, 2)
                else:
                    bounding_boxes = np.transpose(
                        bounding_boxes, (1, 0, 2, 3)
                    )  # (num_identities, num_frames, 2, 2)

            # Recover JABS-specific fields
            cm_per_pixel = jabs_meta.get("cm_per_pixel")
            # Per-identity files store the full external_ids list, but this
            # intermediate PoseData holds only one identity. Pass None here and
            # recover the full list from jabs_meta in _read_merged.
            external_ids = (
                None if jabs_meta.get("per_identity_files") else jabs_meta.get("external_ids")
            )
            metadata = jabs_meta.get("metadata", {})

            # Read static objects from NWB-native PoseEstimation containers.
            # Any PoseEstimation not in identity_names_set is a static object.
            static_objects = self._read_static_objects(pe_containers, identity_names_set)

        pose_data = PoseData(
            points=points,
            point_mask=point_mask,
            identity_mask=identity_mask,
            body_parts=body_parts,
            edges=edges,
            fps=int(fps_value),
            cm_per_pixel=cm_per_pixel,
            bounding_boxes=bounding_boxes,
            static_objects=static_objects,
            external_ids=external_ids,
            metadata=metadata,
        )
        return pose_data, jabs_meta

    def _read_merged(self, path: Path, jabs_meta: dict) -> PoseData:
        """Auto-detect and merge sibling per-identity NWB files."""
        total = jabs_meta["total_identities"]
        stem = path.stem

        # Find the base stem by removing the identity suffix
        identity_name = jabs_meta["identity_names"][0]
        expected_suffix = f"_{identity_name}"
        base_stem = stem[: -len(expected_suffix)] if stem.endswith(expected_suffix) else stem

        # Glob for candidate siblings, then filter to only those whose
        # jabs_metadata identifies them as belonging to this specific file set
        # (same base_stem and total_identities). This prevents stale files from
        # prior runs matching the glob pattern and producing extra identities.
        candidates = sorted(path.parent.glob(f"{base_stem}_*.nwb"))
        parts: list[tuple[int, PoseData, dict]] = []
        for sibling_path in candidates:
            pd, meta = self._read_single(sibling_path)
            if meta.get("per_identity_files") and meta.get("total_identities") == total:
                idx = meta.get("source_identity_index", 0)
                parts.append((idx, pd, meta))

        if len(parts) != total:
            raise ValueError(
                f"Expected {total} identity files for '{base_stem}', "
                f"found {len(parts)} matching candidates out of {len(candidates)} glob results"
            )

        parts.sort(key=lambda x: x[0])
        pose_datas = [pd for _, pd, _ in parts]

        # Validate consistency
        ref = pose_datas[0]
        for i, pd in enumerate(pose_datas[1:], 1):
            if pd.body_parts != ref.body_parts:
                raise ValueError(f"body_parts mismatch between file 0 and file {i}")
            if pd.fps != ref.fps:
                raise ValueError(f"fps mismatch between file 0 and file {i}")
            if pd.points.shape[1] != ref.points.shape[1]:
                raise ValueError(f"num_frames mismatch between file 0 and file {i}")

        # Merge arrays
        points = np.concatenate([pd.points for pd in pose_datas], axis=0)
        point_mask = np.concatenate([pd.point_mask for pd in pose_datas], axis=0)
        identity_mask = np.concatenate([pd.identity_mask for pd in pose_datas], axis=0)

        bounding_boxes = None
        if all(pd.bounding_boxes is not None for pd in pose_datas):
            bounding_boxes = np.concatenate([pd.bounding_boxes for pd in pose_datas], axis=0)

        # Recover external_ids from jabs_meta of the first file; each per-identity
        # file stores the full original list, so any file's meta will do.
        external_ids = parts[0][2].get("external_ids")

        return PoseData(
            points=points,
            point_mask=point_mask,
            identity_mask=identity_mask,
            body_parts=ref.body_parts,
            edges=ref.edges,
            fps=ref.fps,
            cm_per_pixel=ref.cm_per_pixel,
            bounding_boxes=bounding_boxes,
            static_objects=ref.static_objects,
            external_ids=external_ids,
            metadata=ref.metadata,
        )

    @staticmethod
    def _make_nwb_file(**kwargs) -> NWBFile:
        return NWBFile(
            session_description=kwargs.get("session_description", "JABS PoseEstimation Data"),
            session_start_time=kwargs.get(
                "session_start_time",
                datetime.datetime.now(datetime.timezone.utc),
            ),
            identifier=str(kwargs.get("identifier", uuid.uuid4())),
        )

    @staticmethod
    def _make_skeleton(
        body_parts: list[str],
        edges: list[tuple[int, int]],
        **kwargs,
    ) -> Skeleton:  # type: ignore[valid-type]
        return Skeleton(
            name=kwargs.get("skeleton_name", _SKELETON_NAME),
            nodes=body_parts,
            edges=np.array(edges, dtype=np.uint8) if edges else None,
        )

    @staticmethod
    def _build_pose_estimation(
        name: str,
        points: np.ndarray,
        point_mask: np.ndarray,
        body_parts: list[str],
        fps: int,
        skeleton: Skeleton,  # type: ignore[valid-type]
    ) -> PoseEstimation:
        """Build a PoseEstimation container for one identity.

        Args:
            name: Name for this PoseEstimation (e.g., identity name).
            points: Shape (num_frames, num_keypoints, 2).
            point_mask: Shape (num_frames, num_keypoints).
            body_parts: List of body part names corresponding to keypoints.
            fps: Frames per second of the source video.
            skeleton: Skeleton object defining the keypoint connections.
        """
        series_list = []
        for j, bp_name in enumerate(body_parts):
            series = PoseEstimationSeries(
                name=bp_name,
                data=points[:, j, :],
                confidence=point_mask[:, j].astype(np.float64),
                confidence_definition=_CONFIDENCE_DEFINITION,
                reference_frame=_REFERENCE_FRAME,
                rate=float(fps),
                unit="pixels",
            )
            series_list.append(series)

        return PoseEstimation(
            name=name,
            pose_estimation_series=series_list,
            description=f"Pose estimation for {name}",
            skeleton=skeleton,
            source_software="JABS",
        )

    @staticmethod
    def _build_static_skeletons(
        static_objects: dict[str, npt.NDArray],
    ) -> dict[str, Skeleton]:  # type: ignore[valid-type]
        """Build a Skeleton for each 2-D static object array.

        Only arrays with ``ndim == 2`` (shape ``(N, 2)``) are supported.
        Objects with other shapes are skipped with a warning.

        Args:
            static_objects: Mapping of object name to coordinate array.

        Returns:
            Ordered dict mapping object name to its Skeleton.
        """
        skeletons: dict[str, Skeleton] = {}  # type: ignore[valid-type]
        for name, pts in static_objects.items():
            if pts.ndim != 2:
                logger.warning(
                    "Skipping static object %r: expected 2-D array (N, 2), got shape %s",
                    name,
                    pts.shape,
                )
                continue
            nodes = [f"{name}_{i}" for i in range(pts.shape[0])]
            skeletons[name] = Skeleton(name=name, nodes=nodes)
        return skeletons

    @staticmethod
    def _build_static_object_pose_estimation(
        name: str,
        points: npt.NDArray,
        skeleton: Skeleton,  # type: ignore[valid-type]
    ) -> PoseEstimation:
        """Build a single-timestamp PoseEstimation for a static spatial object.

        Each node in the skeleton corresponds to one row of ``points`` and is
        stored as a ``PoseEstimationSeries`` with a single timestamp at t=0.

        Args:
            name: Name for this PoseEstimation (matches the static object key).
            points: Shape (N, 2) array of x, y coordinates.
            skeleton: Skeleton with N nodes, one per point.
        """
        series_list = [
            PoseEstimationSeries(
                name=f"{name}_{i}",
                data=points[i : i + 1, :].astype(np.float64),  # shape (1, 2)
                confidence=np.ones(1, dtype=np.float64),
                confidence_definition="Static landmark; confidence is always 1.0",
                timestamps=[0.0],
                unit="pixels",
                reference_frame=_REFERENCE_FRAME,
            )
            for i in range(points.shape[0])
        ]
        return PoseEstimation(
            name=name,
            pose_estimation_series=series_list,
            description=f"Static object: {name}",
            skeleton=skeleton,
            source_software="JABS",
        )

    @staticmethod
    def _read_static_objects(
        pe_containers: dict[str, PoseEstimation],
        identity_names_set: set[str],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Reconstruct static_objects from non-identity PoseEstimation containers.

        Args:
            pe_containers: All PoseEstimation containers from the behavior module.
            identity_names_set: Set of container names that are animal identities.

        Returns:
            Mapping of static object name to (N, 2) float64 array.
        """
        static_objects: dict[str, npt.NDArray[np.float64]] = {}
        for name, pe in pe_containers.items():
            if name in identity_names_set:
                continue
            # Sort series by numeric suffix to restore original point order.
            series_items = sorted(
                pe.pose_estimation_series.items(),
                key=lambda kv: int(kv[0].rsplit("_", 1)[-1]),
            )
            pts = np.array(
                [np.array(series.data[0]) for _, series in series_items],
                dtype=np.float64,
            )  # shape (N, 2)
            static_objects[name] = pts
        return static_objects

    @staticmethod
    def _build_jabs_metadata(
        data: PoseData,
        identity_names: list[str],
        **extra,
    ) -> dict:
        meta = {
            "format_version": _JABS_NWB_FORMAT_VERSION,
            "cm_per_pixel": data.cm_per_pixel,
            "external_ids": data.external_ids,
            "identity_names": identity_names,
            "num_identities": data.points.shape[0],
            "body_parts": data.body_parts,
            "metadata": data.metadata,
        }
        meta.update(extra)
        return meta

    @staticmethod
    def _sanitize_identity_name(name: str) -> str:
        """Sanitize an external ID for use as an NWB/HDF5 container name.

        Strips leading/trailing whitespace and replaces any character that is
        not alphanumeric, underscore, or hyphen with an underscore.
        """
        name = name.strip()
        if not name:
            raise ValueError("Identity name cannot be empty or whitespace-only")
        return re.sub(r"[^A-Za-z0-9_\-]", "_", name)

    @staticmethod
    def _identity_name(data: PoseData, index: int) -> str:
        if data.external_ids is not None:
            return PoseNWBAdapter._sanitize_identity_name(data.external_ids[index])
        return f"identity_{index}"

    @staticmethod
    def _identity_file_path(base_path: Path, identity_name: str) -> Path:
        return base_path.with_stem(f"{base_path.stem}_{identity_name}")


def _json_default(obj):
    """Handle non-JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
