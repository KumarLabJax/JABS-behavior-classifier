"""NWB adapter for PoseData using ndx-pose."""

from __future__ import annotations

import datetime
import json
import logging
import re
import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt

try:
    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.core import ScratchData
    from pynwb.file import Subject

    _NWB_AVAILABLE = True
except ImportError:
    _NWB_AVAILABLE = False

from jabs.core.abstract.pose_est import PoseEstimation as _JABSPoseEstimation
from jabs.core.enums import StorageFormat
from jabs.core.types import DynamicObjectData, PoseData
from jabs.io.base import Adapter
from jabs.io.registry import register_adapter

logger = logging.getLogger(__name__)

_JABS_NWB_FORMAT_VERSION = 1

# Maps canonical keypoint name → its integer index in KeypointIndex.
# Used by _sort_body_parts to recover the correct keypoint order from
# PoseEstimationSeries names, since pynwb returns series alphabetically from HDF5.
_KEYPOINT_ORDER: dict[str, int] = {
    kpt.name: kpt.value for kpt in _JABSPoseEstimation.KeypointIndex
}


def _sort_body_parts(names: list[str]) -> list[str]:
    """Return keypoint names sorted by canonical KeypointIndex order.

    Names present in KeypointIndex are sorted by their enum value.  Names not
    found in KeypointIndex are appended at the end in the order received (i.e.
    the order pynwb returns them from HDF5, which is alphabetical).

    Args:
        names: Keypoint names as returned by pynwb (alphabetical HDF5 order).

    Returns:
        Names reordered so that known keypoints follow KeypointIndex order,
        with any unrecognised names appended afterward.
    """
    known = [n for n in names if n in _KEYPOINT_ORDER]
    unknown = [n for n in names if n not in _KEYPOINT_ORDER]
    return sorted(known, key=lambda n: _KEYPOINT_ORDER[n]) + sorted(unknown)


_JABS_METADATA_KEY = "jabs_metadata"
_IDENTITY_MASK_KEY = "jabs_identity_mask"
_BOUNDING_BOXES_PREFIX = "jabs_bounding_boxes"
_PROCESSING_MODULE_NAME = "behavior"
_PROCESSING_MODULE_DESC = "JABS pose estimation data"
_SKELETON_NAME = "subject"
_REFERENCE_FRAME = "Top-left corner of video frame, x increases rightward, y increases downward"
_CONFIDENCE_DEFINITION = "0.0=invalid/missing keypoint, >0.0=valid keypoint"
_DYNAMIC_CONFIDENCE_DEFINITION = (
    "1.0=valid object instance in this slot, 0.0=slot unoccupied at this prediction"
)


def _bounding_box_key(identity_name: str) -> str:
    """Return the TimeSeries name for bounding boxes of a given identity."""
    return f"{_BOUNDING_BOXES_PREFIX}_{identity_name}"


@register_adapter(StorageFormat.NWB, PoseData, priority=10)
class PoseNWBAdapter(Adapter):
    """NWB adapter for PoseData."""

    def __init__(self) -> None:
        """Initialize the adapter, raising ImportError if NWB deps are not installed."""
        self._require_nwb()

    @staticmethod
    def _require_nwb() -> None:
        """Raise a clear ImportError if pynwb / ndx-pose are not installed."""
        if not _NWB_AVAILABLE:
            raise ImportError(
                "pynwb and ndx-pose are required for NWB format support. "
                "Install with: pip install 'jabs-io[nwb]'"
            )

    @classmethod
    def can_handle(cls, data_type):  # noqa: D102
        return data_type is PoseData

    def write(self, data: PoseData, path: str | Path, **kwargs) -> None:
        """Write PoseData to NWB file(s).

        By default, all identities are written to a single NWB file at ``path``.
        Pass ``per_identity_files=True`` to write one file per identity instead.
        In that mode, ``path`` is used as a naming template — the base file is
        *not* created; instead, each identity is written to a sibling file whose
        stem is ``{path.stem}_{identity_name}``.  Identity names come from
        ``data.external_ids`` (sanitized for HDF5 compatibility) or fall back to
        ``identity_0``, ``identity_1``, … when ``external_ids`` is ``None``.

        Example — single file::

            save(pose_data, "session.nwb")
            # → session.nwb  (all identities)

        Example — per-identity files with external IDs::

            save(pose_data, "session.nwb", per_identity_files=True)
            # pose_data.external_ids = ["mouse_a", "mouse_b"]
            # → session_mouse_a.nwb
            # → session_mouse_b.nwb

        Example — per-identity files without external IDs::

            save(pose_data, "session.nwb", per_identity_files=True)
            # pose_data.external_ids = None
            # → session_subject_0.nwb
            # → session_subject_1.nwb

        The NWB layout written by this adapter (ndx-pose 0.2)::

            processing/behavior/
              Skeletons/
                subject/              ← animal skeleton (keypoints + edges)
                <obj_name>/           ← one Skeleton per static object
              <identity_name>/        ← one PoseEstimation per identity
                <keypoint>/           ← one PoseEstimationSeries per keypoint
              <obj_name>/             ← one PoseEstimation per static object
                <obj_name>_0/         ← one PoseEstimationSeries per point
              jabs_identity_mask              ← TimeSeries, uint8 presence mask
              jabs_bounding_boxes_<identity>  ← TimeSeries per identity, optional (num_frames, 2, 2)
            scratch/
              jabs_metadata           ← JSON: format_version, cm_per_pixel,
                                        identity_names, body_parts, metadata, …

        Args:
            data: The PoseData to write.
            path: Output file path (.nwb).  In per-identity mode this is a
                naming template; the actual files are written alongside it.
            **kwargs:
                per_identity_files (bool): Write one NWB file per identity.
                    Default ``False``.
                session_description (str): NWB session description string.
                    Default ``"JABS PoseEstimation Data"``.
                session_start_time (datetime.datetime): NWB session start time.
                    Default: current UTC time at write time.
                identifier (str): NWB file identifier.  Default: random UUID.
                skeleton_name (str): Name for the animal Skeleton object.
                    Default ``"subject"``.

        Raises:
            ValueError: If sanitized identity names are not unique (collision
                after special-character replacement).
        """
        path = Path(path)
        per_identity_files = kwargs.get("per_identity_files", False)

        if per_identity_files:
            self._write_per_identity(data, path, **kwargs)
        else:
            self._write_single_file(data, path, **kwargs)

    def read(self, path: str | Path, data_type: type | None = None) -> PoseData:
        """Read PoseData from an NWB file.

        Handles both single-file and per-identity file layouts transparently —
        no kwargs are needed; the file records which layout was used.

        **Single-file layout:** point ``path`` at the file written by
        ``write()``.  All identities are returned in one ``PoseData``.

        **Per-identity layout:** point ``path`` at *any one* of the sibling
        files.  The reader detects the per-identity flag in the embedded
        ``jabs_metadata``, then globs for siblings matching
        ``{base_stem}_*.nwb`` in the same directory, filters to those that
        belong to the same write session (matching ``total_identities``), sorts
        by ``source_identity_index``, and concatenates them into a single
        ``PoseData`` with all identities restored in their original order.

        Example::

            # Single file
            pose_data = load("session.nwb", PoseData)

            # Per-identity — point at any sibling; result is the same
            pose_data = load("session_mouse_a.nwb", PoseData)
            pose_data = load("session_mouse_b.nwb", PoseData)

        Args:
            path: Path to an NWB file written by this adapter.
            data_type: Ignored; present for interface compatibility.

        Returns:
            ``PoseData`` with all identities merged in their original order.

        Raises:
            ValueError: If no ``PoseEstimation`` containers are found, or if
                the expected number of sibling files cannot be located when
                reading a per-identity layout.
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
        dynamic_skeletons = self._build_dynamic_skeletons(data.dynamic_objects)
        skeletons = Skeletons(
            skeletons=[skeleton, *static_skeletons.values(), *dynamic_skeletons.values()]
        )

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

        for obj_name, obj_skeleton in dynamic_skeletons.items():
            behavior.add(
                self._build_dynamic_object_pose_estimation(
                    obj_name, data.dynamic_objects[obj_name], data.fps, obj_skeleton
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
            for i, name in enumerate(identity_names):
                behavior.add(
                    TimeSeries(
                        name=_bounding_box_key(name),
                        data=data.bounding_boxes[i],  # (num_frames, 2, 2)
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

            subject_meta = (data.subjects or {}).get(identity_name)
            subject = self._make_subject(subject_meta) if subject_meta else None
            nwbfile = self._make_nwb_file(subject=subject, **kwargs)
            skeleton = self._make_skeleton(data.body_parts, data.edges, **kwargs)
            # Rebuild static/dynamic skeletons each iteration: HDMF objects can
            # only belong to one container, so they cannot be shared across files.
            static_skeletons = self._build_static_skeletons(data.static_objects)
            dynamic_skeletons = self._build_dynamic_skeletons(data.dynamic_objects)
            skeletons = Skeletons(
                skeletons=[skeleton, *static_skeletons.values(), *dynamic_skeletons.values()]
            )

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

            for obj_name, obj_skeleton in dynamic_skeletons.items():
                behavior.add(
                    self._build_dynamic_object_pose_estimation(
                        obj_name, data.dynamic_objects[obj_name], data.fps, obj_skeleton
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
                        name=_bounding_box_key(identity_name),
                        data=data.bounding_boxes[i],  # (num_frames, 2, 2)
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
            dynamic_object_names = jabs_meta.get("dynamic_object_names", [])
            dynamic_object_shapes = jabs_meta.get("dynamic_object_shapes", {})
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

            # Derive body_parts order from series names using KeypointIndex.
            # pynwb returns PoseEstimationSeries alphabetically from HDF5; _sort_body_parts
            # restores the canonical order.  Missing canonical keypoints are included and
            # padded with NaN so the returned array always has a consistent shape.
            first_pe = identity_pe_containers[ordered_names[0]]
            series_in_file = set(first_pe.pose_estimation_series.keys())
            missing_keypoints = [n for n in _KEYPOINT_ORDER if n not in series_in_file]
            canonical_present = [n for n in _KEYPOINT_ORDER if n in series_in_file]
            if canonical_present and missing_keypoints:
                logger.warning(
                    "NWB file %s is missing %d canonical keypoint(s): %s. "
                    "Missing keypoints will be padded with NaN.",
                    path,
                    len(missing_keypoints),
                    missing_keypoints,
                )
                body_parts = _sort_body_parts(list(series_in_file) + missing_keypoints)
            else:
                body_parts = _sort_body_parts(list(series_in_file))

            unknown_keypoints = [bp for bp in body_parts if bp not in _KEYPOINT_ORDER]
            if unknown_keypoints:
                logger.warning(
                    "NWB file %s contains %d unrecognised keypoint(s): %s.",
                    path,
                    len(unknown_keypoints),
                    unknown_keypoints,
                )

            num_keypoints = len(body_parts)

            # Use a keypoint that is present in the file to determine num_frames and fps.
            present_body_parts = [bp for bp in body_parts if bp in series_in_file]
            fps_value = first_pe.pose_estimation_series[present_body_parts[0]].rate

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
                num_frames = len(pe.pose_estimation_series[present_body_parts[0]].data)
                identity_points = np.full((num_frames, num_keypoints, 2), np.nan)
                identity_mask = np.zeros((num_frames, num_keypoints), dtype=bool)
                for j, bp in enumerate(body_parts):
                    if bp not in pe.pose_estimation_series:
                        continue  # already NaN / False
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

            # Bounding boxes — one TimeSeries per identity, each (num_frames, 2, 2).
            # Stack to (num_identities, num_frames, 2, 2). Present only if all identity
            # containers exist (bounding boxes are all-or-nothing in PoseData).
            bounding_boxes = None
            bb_keys = [_bounding_box_key(name) for name in ordered_names]
            if all(k in behavior.data_interfaces for k in bb_keys):
                bounding_boxes = np.stack(
                    [np.array(behavior[k].data[:]) for k in bb_keys], axis=0
                )  # (num_identities, num_frames, 2, 2)

            # Recover JABS-specific fields
            cm_per_pixel = jabs_meta.get("cm_per_pixel")
            # Per-identity files store the full external_ids and subjects dicts, but
            # this intermediate PoseData holds only one identity. Pass None here and
            # recover the full values from jabs_meta in _read_merged.
            per_id = jabs_meta.get("per_identity_files", False)
            external_ids = None if per_id else jabs_meta.get("external_ids")
            subjects = None if per_id else jabs_meta.get("subjects")
            metadata = jabs_meta.get("metadata", {})

            # Read dynamic objects from NWB-native PoseEstimation containers.
            dynamic_objects = self._read_dynamic_objects(
                pe_containers, dynamic_object_names, dynamic_object_shapes, int(fps_value)
            )

            # Read static objects from NWB-native PoseEstimation containers.
            static_object_names = jabs_meta.get("static_object_names", [])
            static_objects = self._read_static_objects(pe_containers, static_object_names)

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
            dynamic_objects=dynamic_objects,
            external_ids=external_ids,
            subjects=subjects,
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

        # Recover external_ids and subjects from jabs_meta of the first file;
        # each per-identity file stores the full original values, so any file's meta will do.
        first_meta = parts[0][2]
        external_ids = first_meta.get("external_ids")
        subjects = first_meta.get("subjects")

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
            dynamic_objects=ref.dynamic_objects,
            external_ids=external_ids,
            subjects=subjects,
            metadata=ref.metadata,
        )

    @staticmethod
    def _make_nwb_file(**kwargs) -> NWBFile:
        nwb_kwargs: dict = {
            "session_description": kwargs.get("session_description", "JABS PoseEstimation Data"),
            "session_start_time": kwargs.get(
                "session_start_time",
                datetime.datetime.now(datetime.timezone.utc),
            ),
            "identifier": str(kwargs.get("identifier", uuid.uuid4())),
        }
        for _field in (
            "experimenter",
            "lab",
            "institution",
            "experiment_description",
            "session_id",
        ):
            if kwargs.get(_field) is not None:
                nwb_kwargs[_field] = kwargs[_field]
        if kwargs.get("subject") is not None:
            nwb_kwargs["subject"] = kwargs["subject"]
        return NWBFile(**nwb_kwargs)

    @staticmethod
    def _make_subject(subject_meta: dict) -> Subject:
        """Build a pynwb Subject from a subject metadata dict.

        Args:
            subject_meta: Dict with optional keys subject_id, sex, genotype,
                strain, age, weight, species, description.  None values are
                omitted so pynwb uses its own defaults.

        Returns:
            A pynwb Subject object populated from the non-None fields.
        """
        _SUBJECT_FIELDS = (
            "subject_id",
            "sex",
            "genotype",
            "strain",
            "age",
            "weight",
            "species",
            "description",
        )
        kwargs = {k: v for k, v in subject_meta.items() if k in _SUBJECT_FIELDS and v is not None}
        return Subject(**kwargs)

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
    def _build_dynamic_skeletons(
        dynamic_objects: dict[str, DynamicObjectData],
    ) -> dict[str, Skeleton]:  # type: ignore[valid-type]
        """Build a Skeleton for each dynamic object.

        For single-keypoint objects, nodes are named ``{name}_{slot}``.
        For multi-keypoint objects, nodes are named ``{name}_{slot}_{kp}``.

        Args:
            dynamic_objects: Mapping of object name to DynamicObjectData.

        Returns:
            Ordered dict mapping object name to its Skeleton.
        """
        skeletons: dict[str, Skeleton] = {}  # type: ignore[valid-type]
        for name, dyn_obj in dynamic_objects.items():
            max_count = dyn_obj.points.shape[1]
            n_keypoints = dyn_obj.points.shape[2]
            if n_keypoints == 1:
                nodes = [f"{name}_{slot}" for slot in range(max_count)]
            else:
                nodes = [
                    f"{name}_{slot}_{kp}" for slot in range(max_count) for kp in range(n_keypoints)
                ]
            skeletons[name] = Skeleton(name=name, nodes=nodes)
        return skeletons

    @staticmethod
    def _build_dynamic_object_pose_estimation(
        name: str,
        dyn_obj: DynamicObjectData,
        fps: int,
        skeleton: Skeleton,  # type: ignore[valid-type]
    ) -> PoseEstimation:
        """Build a PoseEstimation container for one dynamic object.

        Each slot/keypoint combination is stored as a PoseEstimationSeries with
        irregular timestamps corresponding to the sample indices.  Confidence of
        1.0 indicates the slot is occupied at that prediction; 0.0 means empty.

        Args:
            name: Name of the dynamic object (e.g. "fecal_boli").
            dyn_obj: DynamicObjectData with shape (n_predictions, max_count, n_keypoints, 2).
            fps: Frames per second used to convert sample indices to timestamps.
            skeleton: Skeleton built by :meth:`_build_dynamic_skeletons`.

        Returns:
            PoseEstimation container ready to add to the behavior module.
        """
        n_predictions, max_count, n_keypoints, _ = dyn_obj.points.shape
        timestamps = (dyn_obj.sample_indices / fps).tolist()
        series_list = []
        for slot in range(max_count):
            slot_confidence = (dyn_obj.counts > slot).astype(np.float64)
            for kp in range(n_keypoints):
                series_name = f"{name}_{slot}" if n_keypoints == 1 else f"{name}_{slot}_{kp}"
                series_list.append(
                    PoseEstimationSeries(
                        name=series_name,
                        data=dyn_obj.points[:, slot, kp, :].astype(np.float64),
                        confidence=slot_confidence,
                        confidence_definition=_DYNAMIC_CONFIDENCE_DEFINITION,
                        timestamps=timestamps,
                        unit="pixels",
                        reference_frame=_REFERENCE_FRAME,
                    )
                )
        return PoseEstimation(
            name=name,
            pose_estimation_series=series_list,
            description=f"Dynamic object: {name}",
            skeleton=skeleton,
            source_software="JABS",
        )

    @staticmethod
    def _read_dynamic_objects(
        pe_containers: dict[str, PoseEstimation],
        dynamic_object_names: list[str],
        dynamic_object_shapes: dict[str, list[int]],
        fps: int,
    ) -> dict[str, DynamicObjectData]:
        """Reconstruct dynamic_objects from PoseEstimation containers.

        Args:
            pe_containers: All PoseEstimation containers from the behavior module.
            dynamic_object_names: Names of dynamic objects recorded in jabs_metadata.
            dynamic_object_shapes: Mapping of name to [max_count, n_keypoints].
            fps: Frames per second used to convert timestamps back to sample indices.

        Returns:
            Mapping of object name to DynamicObjectData.
        """
        dynamic_objects: dict[str, DynamicObjectData] = {}
        for name in dynamic_object_names:
            if name not in pe_containers:
                logger.warning("Dynamic object %r not found in behavior module", name)
                continue
            pe = pe_containers[name]
            max_count, n_keypoints = dynamic_object_shapes[name]
            first_series = next(iter(pe.pose_estimation_series.values()))
            timestamps = np.array(first_series.timestamps[:])
            sample_indices = np.round(timestamps * fps).astype(np.int64)
            n_predictions = len(sample_indices)
            points = np.empty((n_predictions, max_count, n_keypoints, 2), dtype=np.float64)
            counts = np.zeros(n_predictions, dtype=np.int64)
            for slot in range(max_count):
                slot_confidence = None
                for kp in range(n_keypoints):
                    series_name = f"{name}_{slot}" if n_keypoints == 1 else f"{name}_{slot}_{kp}"
                    series = pe.pose_estimation_series[series_name]
                    points[:, slot, kp, :] = np.array(series.data[:])
                    if slot_confidence is None:
                        slot_confidence = np.array(series.confidence[:])
                if slot_confidence is not None:
                    counts += (slot_confidence > 0).astype(np.int64)
            dynamic_objects[name] = DynamicObjectData(
                points=points, counts=counts, sample_indices=sample_indices
            )
        return dynamic_objects

    @staticmethod
    def _read_static_objects(
        pe_containers: dict[str, PoseEstimation],
        static_names: list[str],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Reconstruct static_objects from PoseEstimation containers.

        Args:
            pe_containers: All PoseEstimation containers from the behavior module.
            static_names: Ordered list of static object container names from
                ``jabs_metadata``.

        Returns:
            Mapping of static object name to (N, 2) float64 array.
        """
        static_objects: dict[str, npt.NDArray[np.float64]] = {}
        for name in static_names:
            if name not in pe_containers:
                logger.warning("Static object %r listed in metadata but not found in file", name)
                continue
            series_items = sorted(
                pe_containers[name].pose_estimation_series.items(),
                key=lambda kv: int(kv[0].rsplit("_", 1)[-1]),
            )
            static_objects[name] = np.array(
                [np.array(series.data[0]) for _, series in series_items],
                dtype=np.float64,
            )  # shape (N, 2)
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
            "subjects": data.subjects,
            "metadata": data.metadata,
        }
        if data.static_objects:
            meta["static_object_names"] = list(data.static_objects.keys())
        if data.dynamic_objects:
            meta["dynamic_object_names"] = list(data.dynamic_objects.keys())
            meta["dynamic_object_shapes"] = {
                name: [dyn.points.shape[1], dyn.points.shape[2]]
                for name, dyn in data.dynamic_objects.items()
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
        return f"subject_{index}"

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
