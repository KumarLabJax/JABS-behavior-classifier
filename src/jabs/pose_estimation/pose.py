"""The ``Pose`` class: JABS's pose API over one unified pose file (ADR 0002).

``Pose`` implements :class:`jabs.core.pose_protocol.PoseLike`. It holds one
:class:`~jabs.io.internal.pose_file.ComponentReader` open for its lifetime,
loads the keypoint-scale components eagerly on open, and reads segmentation
lazily in frame windows. It never loads the whole file.
"""

import logging
import re
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry

from jabs.core.enums import PoseCapability
from jabs.core.pose_protocol import (
    NATIVE_SOURCE_FORMAT,
    parse_model_metadata,
    source_format_from_path,
)
from jabs.core.pose_skeleton import (
    FULL_CONNECTED_SEGMENTS,
    NVSN_CONNECTED_SEGMENTS,
    TAIL_KEYPOINTS,
    KeypointIndex,
)
from jabs.core.types import DynamicObjectData, SegmentationWindow
from jabs.core.utils import hash_file
from jabs.io.internal.pose_file import ComponentReader, HistoryEntry, PoseFileError

logger = logging.getLogger(__name__)

__all__ = ["Pose"]

_POINTS = "jabs.pose.points"
_POINT_VALID = "jabs.pose.point_valid"
_SLOT_OCCUPIED = "jabs.pose.slot_occupied"
_SLOT_USABLE = "jabs.pose.slot_usable"
_TRACKLET_ID = "jabs.pose.tracklet_id"
BBOX_COMPONENT = "jabs.pose.bbox"
_EXTERNAL_IDS = "jabs.identity.external_ids"
STATIC_OBJECT_PREFIX = "jabs.static_objects."
_DYNAMIC_PREFIX = "jabs.dynamic_objects."
_DYNAMIC_PARTS = ("frame_index", "points", "counts")

_CONTOURS = "jabs.segmentation.contours"
_CONTOUR_COUNT = "jabs.segmentation.contour_count"
_CONTOUR_LENGTH = "jabs.segmentation.contour_length"
_EXTERNAL_FLAG = "jabs.segmentation.external_flag"
_SEGMENTATION_REQUIRED = (_CONTOURS, _CONTOUR_COUNT, _CONTOUR_LENGTH)

# Segmentation is read in windows of this many frames, aligned to multiples of
# it: the writer chunks compressed components at 300 frames, so a window is
# whole chunks. The last few windows are kept, across every slot, so walking
# the frames identity by identity does not re-read a window per identity.
_SEGMENTATION_BLOCK = 300
_SEGMENTATION_CACHE_SIZE = 8

_DEFAULT_FPS = 30

# The minimum number of valid non-tail keypoints for an identity to be present,
# and for a convex hull.
_MIN_BODY_KEYPOINTS = 3

# Keypoints the Envision Hydranet (NVSN) pose lacks; when none of them is ever
# valid, the NVSN skeleton is drawn (pose_est_v5.py).
_MISSING_FROM_NVSN = (
    KeypointIndex.BASE_NECK.value,
    KeypointIndex.LEFT_FRONT_PAW.value,
    KeypointIndex.RIGHT_FRONT_PAW.value,
    KeypointIndex.CENTER_SPINE.value,
    KeypointIndex.LEFT_REAR_PAW.value,
    KeypointIndex.RIGHT_REAR_PAW.value,
    KeypointIndex.MID_TAIL.value,
)

# The convert history's source format of a legacy v3 conversion, whose tracks
# are its identities.
_V3_FORMAT = "pose_est_v3"

_POSE_NAME_SUFFIX_RE = re.compile(r"(_pose_est_v\d+|_pose)$")

# One segmentation window across every slot: contours (n, S, C, P, 2),
# contour_count (n, S), contour_length (n, S, C), external_flag (n, S, C) or None.
_SegmentationBlock = tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.bool_] | None,
]


def _video_base(path: Path) -> str:
    """Return the video stem a pose file name was derived from.

    ``vid_pose_est_v6.h5`` and ``vid_pose.h5`` both give ``vid``; any other
    name gives its stem.
    """
    return _POSE_NAME_SUFFIX_RE.sub("", path.with_suffix("").name)


def _last_convert_entry(history: tuple[HistoryEntry, ...]) -> HistoryEntry | None:
    """Return the most recent ``convert`` history entry, or None."""
    for entry in reversed(history):
        if entry.operation == "convert":
            return entry
    return None


def converted_from_format(convert: HistoryEntry | None) -> str | None:
    """Return the source format a ``convert`` history entry names, or None.

    Args:
        convert: The file's last ``convert`` history entry, or None.

    Returns:
        The entry's ``source.format``, such as ``"pose_est_v3"``; None without one.
    """
    return (convert.source or {}).get("format") if convert is not None else None


def tracks_are_identities(manifest: dict, converted_from: str | None) -> bool:
    """Return True when every slot of a unified file is an identity.

    That is a legacy v3 conversion: ``dimensions.identity == 0``, a
    ``jabs.pose.tracklet_id`` component, and a last ``convert`` entry from
    ``pose_est_v3``. Any other ``identity == 0`` file (such as a v4+ file
    whose instances are all masked, which keeps its tracklets) has no
    identities.

    Args:
        manifest: The file's manifest.
        converted_from: :func:`converted_from_format` of its last convert entry.

    Returns:
        Whether the file's tracks are its identities.
    """
    component_ids = {spec["id"] for spec in manifest["components"]}
    return (
        manifest["dimensions"]["identity"] == 0
        and _TRACKLET_ID in component_ids
        and converted_from == _V3_FORMAT
    )


def identity_count(manifest: dict, converted_from: str | None) -> int:
    """Return a unified file's number of identities.

    ``dimensions.slot`` when its tracks are its identities (see
    :func:`tracks_are_identities`), otherwise ``dimensions.identity``.

    Args:
        manifest: The file's manifest.
        converted_from: :func:`converted_from_format` of its last convert entry.

    Returns:
        The number of identities.
    """
    dimensions = manifest["dimensions"]
    if tracks_are_identities(manifest, converted_from):
        return int(dimensions["slot"])
    return int(dimensions["identity"])


def has_segmentation_components(component_ids: set[str]) -> bool:
    """Return True when the components needed to read segmentation are all present.

    Args:
        component_ids: The ids of a file's components.
    """
    return all(c in component_ids for c in _SEGMENTATION_REQUIRED)


def resolve_source_format(path: Path, converted_from: str | None) -> str:
    """Return the ``source_format`` of the pose file the user has.

    Taken from ``path``'s name; when that follows neither naming convention,
    from the convert history entry, else native.

    Args:
        path: The pose file the user has.
        converted_from: :func:`converted_from_format` of its last convert
            entry, or None (always None for a legacy file).

    Returns:
        ``"pose_est_v<N>"`` or ``"jabs.pose-file"``.
    """
    return source_format_from_path(path) or converted_from or NATIVE_SOURCE_FORMAT


class Pose:
    """Read access to one video's pose, from a unified pose file.

    Implements :class:`jabs.core.pose_protocol.PoseLike`. Identities are slots
    ``0..dimensions.identity - 1``; a legacy v3 conversion (``dimensions.identity
    == 0``, ``jabs.pose.tracklet_id``, and a ``convert`` history entry from
    ``pose_est_v3``) has every slot as an identity. Any other
    ``dimensions.identity == 0`` file has no identities. Points are float32
    with invalid points NaN; masks are bool.

    Args:
        pose_path: The unified pose file to read.
        source_path: The file the user has, when ``pose_path`` was converted
            from it; defaults to ``pose_path``. It names ``pose_file`` and
            ``source_format``.
        source_hash: ``jabs.core.utils.hash_file`` of ``source_path``, when the
            caller already has it; computed otherwise.
        cache_dir: Optional directory to cache convex hulls in.
        converter_revision: The converter revision that produced ``pose_path``
            from ``source_path``, or None when the file was not converted.

    Raises:
        NotAPoseFileError: If ``pose_path`` is not a unified pose file.
        PoseFileError: If the file is unreadable, disagrees with its manifest,
            lacks a required component, or does not use the JABS 12-keypoint
            mouse skeleton.
        NotImplementedError: If a component's encoding cannot be decoded.
    """

    def __init__(
        self,
        pose_path: Path,
        *,
        source_path: Path | None = None,
        source_hash: str | None = None,
        cache_dir: Path | None = None,
        converter_revision: int | None = None,
    ) -> None:
        self._pose_path = pose_path
        self._converter_revision = converter_revision
        self._path = source_path if source_path is not None else pose_path
        self._cache_dir = cache_dir
        # hashed before the reader opens, so a failure here leaks no open file
        self._hash = source_hash if source_hash is not None else hash_file(self._path)
        self._reader = ComponentReader(pose_path)
        # closes the file when the Pose is collected; never refers to self
        self._finalizer = weakref.finalize(self, self._reader.close)
        self._convex_hull_cache: dict[int, list[BaseGeometry | None]] = {}
        self._connected_segments: tuple[tuple[KeypointIndex, ...], ...] | None = None
        self._segmentation_cache: OrderedDict[int, _SegmentationBlock] = OrderedDict()
        try:
            self._load()
        except BaseException:
            self.close()
            raise

    def _load(self) -> None:
        """Read the manifest, provenance and keypoint-scale components."""
        manifest = self._reader.manifest
        self._component_ids = {spec["id"] for spec in manifest["components"]}
        missing = [
            c for c in (_POINTS, _POINT_VALID, _SLOT_OCCUPIED) if c not in self._component_ids
        ]
        if missing:
            raise PoseFileError(f"{self._pose_path}: missing required components {missing}")
        body_parts = self._check_skeleton(manifest)

        convert = _last_convert_entry(self._reader.read_provenance().history)
        extra = (convert.extra or {}) if convert is not None else {}
        converted_from = converted_from_format(convert)

        self._num_frames = int(manifest["dimensions"]["frame"])
        self._tracks_are_identities = tracks_are_identities(manifest, converted_from)
        num_identities = identity_count(manifest, converted_from)
        self._identities = list(range(num_identities))

        video = manifest["video"]
        self._fps: int | float = _DEFAULT_FPS
        if video.get("fps") is None:
            logger.info("%s records no fps; using %d", self._path.name, _DEFAULT_FPS)
        else:
            fps = float(video["fps"])
            self._fps = int(fps) if fps.is_integer() else fps
        scale = video.get("cm_per_pixel")
        self._cm_per_pixel = None if scale is None else float(scale)

        self._load_pose(num_identities, body_parts)
        self._load_objects(manifest)
        self._external_identities: list[str] | None = None
        if _EXTERNAL_IDS in self._component_ids:
            names = self._reader.read(_EXTERNAL_IDS)
            self._external_identities = [str(name) for name in names.tolist()]

        self._has_segmentation = has_segmentation_components(self._component_ids)
        self._segmentation_capacity: tuple[int, int] = (0, 0)
        if self._has_segmentation:
            spec = next(s for s in manifest["components"] if s["id"] == _CONTOURS)
            self._segmentation_capacity = (int(spec["shape"][2]), int(spec["shape"][3]))

        self._model_metadata = parse_model_metadata(extra.get("model_metadata_json"))
        self._source_format = resolve_source_format(self._path, converted_from)

    def _check_skeleton(self, manifest: dict) -> tuple[str, ...]:
        """Return the points' body parts, refusing anything but the JABS mouse skeleton."""
        spec = next(s for s in manifest["components"] if s["id"] == _POINTS)
        skeleton = manifest.get("skeletons", {}).get(spec.get("skeleton"), {})
        body_parts = tuple(skeleton.get("body_parts", ()))
        expected = tuple(k.name for k in KeypointIndex)
        if body_parts != expected:
            raise PoseFileError(
                f"{self._pose_path}: {_POINTS} uses body parts {list(body_parts)}; JABS "
                f"reads only the 12-keypoint mouse skeleton {list(expected)}"
            )
        return body_parts

    def _load_pose(self, num_identities: int, body_parts: tuple[str, ...]) -> None:
        """Read points, validity, presence, tracks and boxes for the identity slots."""
        identity_slots = slice(0, num_identities)
        # identity-major copies, so one identity's frames are contiguous
        valid = np.ascontiguousarray(
            self._reader.read(_POINT_VALID)[:, identity_slots].astype(bool).transpose(1, 0, 2)
        )
        points = self._reader.read(_POINTS)[:, identity_slots].astype(np.float32)
        points = np.ascontiguousarray(points.transpose(1, 0, 2, 3))
        points[~valid] = np.nan
        self._points = points
        self._point_mask = valid

        if _SLOT_USABLE in self._component_ids:
            usable = self._reader.read(_SLOT_USABLE)[:, identity_slots].astype(bool).T
        else:
            body = [i for i, name in enumerate(body_parts) if name not in TAIL_KEYPOINTS]
            usable = valid[..., body].sum(axis=-1) >= _MIN_BODY_KEYPOINTS
        self._identity_mask = np.ascontiguousarray(usable)

        self._identity_to_track: npt.NDArray[np.int32] | None = None
        if self._tracks_are_identities:
            occupied = self._reader.read(_SLOT_OCCUPIED)[:, identity_slots].astype(bool)
            tracklets = self._reader.read(_TRACKLET_ID)[:, identity_slots].astype(np.int64)
            # tracklet 0 means none; v3 track ids are stored plus one
            self._identity_to_track = np.ascontiguousarray(
                np.where(occupied, tracklets - 1, -1).T.astype(np.int32)
            )

        self._bboxes: npt.NDArray[np.float32] | None = None
        if BBOX_COMPONENT in self._component_ids:
            boxes = self._reader.read(BBOX_COMPONENT)[:, identity_slots].astype(np.float32)
            self._bboxes = np.ascontiguousarray(boxes.transpose(1, 0, 2, 3))

    def _load_objects(self, manifest: dict) -> None:
        """Read static and dynamic objects."""
        self._static_objects: dict[str, npt.NDArray[np.float32]] = {}
        self._lixit_keypoints = 0
        dynamic_names: list[str] = []
        for spec in manifest["components"]:
            component_id = spec["id"]
            if component_id.startswith(STATIC_OBJECT_PREFIX):
                name = component_id[len(STATIC_OBJECT_PREFIX) :]
                data = self._reader.read(component_id).astype(np.float32)
                if name == "lixit":
                    self._lixit_keypoints = int(data.shape[1])
                self._static_objects[name] = data[:, 0] if data.shape[1] == 1 else data
            elif component_id.startswith(_DYNAMIC_PREFIX):
                name = component_id[len(_DYNAMIC_PREFIX) :].rsplit(".", 1)[0]
                if name not in dynamic_names:
                    dynamic_names.append(name)

        self._dynamic_objects: dict[str, DynamicObjectData] = {}
        for name in dynamic_names:
            ids = {part: f"{_DYNAMIC_PREFIX}{name}.{part}" for part in _DYNAMIC_PARTS}
            absent = [i for i in ids.values() if i not in self._component_ids]
            if absent:
                logger.warning("Dynamic object %r is missing %s; skipping it", name, absent)
                continue
            self._dynamic_objects[name] = DynamicObjectData(
                points=self._reader.read(ids["points"]).astype(np.float64),
                counts=self._reader.read(ids["counts"]).astype(np.int64),
                sample_indices=self._reader.read(ids["frame_index"]).astype(np.int64),
            )

    def close(self) -> None:
        """Close the pose file; a no-op when already closed.

        Eagerly loaded data stays readable; segmentation does not.
        """
        self._finalizer()
        self._segmentation_cache.clear()

    def __enter__(self) -> "Pose":
        """Return self, for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Call :meth:`close`."""
        self.close()

    # -- metadata ---------------------------------------------------------

    @property
    def num_frames(self) -> int:
        """Number of frames."""
        return self._num_frames

    @property
    def identities(self) -> list[int]:
        """Identity indexes, ``0..num_identities-1``."""
        return self._identities

    @property
    def num_identities(self) -> int:
        """Number of identities."""
        return len(self._identities)

    @property
    def cm_per_pixel(self) -> float | None:
        """Centimeters per pixel from the manifest, or None when unknown."""
        return self._cm_per_pixel

    @property
    def fps(self) -> int | float:
        """The manifest's ``video.fps``, 30 when absent; an int when whole."""
        return self._fps

    @property
    def pose_file(self) -> Path:
        """Path of the pose file the user has (the legacy file for a conversion)."""
        return self._path

    @property
    def hash(self) -> str:
        """Hash of :attr:`pose_file`."""
        return self._hash

    @property
    def converter_revision(self) -> int | None:
        """Converter revision of a converted legacy file; None for a unified file."""
        return self._converter_revision

    @property
    def source_format(self) -> str:
        """``"pose_est_v<N>"`` for a legacy source, ``"jabs.pose-file"`` for native.

        Taken from :attr:`pose_file`'s name; when that follows neither naming
        convention, from the convert history entry, else native.
        """
        return self._source_format

    @property
    def model_metadata(self) -> dict[str, Any] | None:
        """The convert entry's preserved ``model_metadata_json``, parsed; None when absent."""
        return self._model_metadata

    @property
    def identity_to_track(self) -> npt.NDArray[np.int32] | None:
        """``(I, F)`` int32 track ids, -1 where a slot is unoccupied.

        Only a file whose identities are tracks (``dimensions.identity == 0``
        with ``jabs.pose.tracklet_id``, a v3 conversion) has one; None otherwise.
        """
        return self._identity_to_track

    @property
    def external_identities(self) -> list[str] | None:
        """External identity names indexed by identity, or None."""
        return self._external_identities

    @property
    def static_objects(self) -> dict[str, npt.NDArray[np.float32]]:
        """Static object name to float32 xy points: ``(O, P, 2)``, ``(O, 2)`` when P == 1."""
        return self._static_objects

    @property
    def num_lixit_keypoints(self) -> int:
        """Keypoints per lixit, 0 without a lixit."""
        return self._lixit_keypoints

    @property
    def dynamic_objects(self) -> dict[str, DynamicObjectData]:
        """Dynamic object name to its data; empty when the file has none."""
        return self._dynamic_objects

    def get_dynamic_object(self, name: str) -> DynamicObjectData | None:
        """Return the named dynamic object, or None when absent.

        Args:
            name: Dynamic object name, e.g. ``"fecal_boli"``.
        """
        return self._dynamic_objects.get(name)

    def has_static_object(self, name: str) -> bool:
        """Return True when the file has the named static object.

        Args:
            name: Static object name, e.g. ``"lixit"``.
        """
        return name in self._static_objects

    @property
    def capabilities(self) -> frozenset[PoseCapability]:
        """Optional data the file provides.

        ``SOCIAL`` iff more than one identity, ``SEGMENTATION`` iff it has
        segmentation contours, ``BOUNDING_BOXES`` iff it has bounding boxes.
        """
        capabilities = set()
        if self.has_social:
            capabilities.add(PoseCapability.SOCIAL)
        if self.has_segmentation:
            capabilities.add(PoseCapability.SEGMENTATION)
        if self.has_bounding_boxes:
            capabilities.add(PoseCapability.BOUNDING_BOXES)
        return frozenset(capabilities)

    @property
    def has_social(self) -> bool:
        """True when the file has more than one identity; tail slots do not count."""
        return self.num_identities > 1

    @property
    def has_segmentation(self) -> bool:
        """True when the file has segmentation contours."""
        return self._has_segmentation

    @property
    def has_bounding_boxes(self) -> bool:
        """True when the file has bounding boxes."""
        return self._bboxes is not None

    # -- points -----------------------------------------------------------

    def _check_identity(self, identity: int) -> None:
        """Raise ValueError when ``identity`` is not one of :attr:`identities`."""
        if not 0 <= identity < self.num_identities:
            raise ValueError(f"identity {identity} out of range (0 to {self.num_identities - 1})")

    def get_points(
        self, frame_index: int, identity: int, scale: float | None = None
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]] | tuple[None, None]:
        """Return one frame's points and point mask for an identity.

        Args:
            frame_index: Frame index.
            identity: Identity index.
            scale: Optional factor applied to the points (e.g. ``cm_per_pixel``).

        Returns:
            ``(K, 2)`` points, NaN where invalid, and the ``(K,)`` point mask,
            or ``(None, None)`` when the identity is absent from the frame.
        """
        if not self._identity_mask[identity, frame_index]:
            return None, None
        points = self._points[identity, frame_index]
        if scale is not None:
            points = points * scale
        return points, self._point_mask[identity, frame_index]

    def get_identity_poses(
        self, identity: int, scale: float | None = None
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Return all points and point masks for an identity.

        Args:
            identity: Identity index.
            scale: Optional factor applied to the points (e.g. ``cm_per_pixel``).

        Returns:
            ``(F, K, 2)`` float32 points, NaN where invalid, and the ``(F, K)``
            point mask.
        """
        points = self._points[identity]
        if scale is not None:
            points = points * scale
        return points, self._point_mask[identity]

    def get_identity_point_mask(self, identity: int) -> npt.NDArray[np.bool_]:
        """Return the ``(F, K)`` point mask for an identity.

        Args:
            identity: Identity index.
        """
        return self._point_mask[identity]

    def identity_mask(self, identity: int) -> npt.NDArray[np.bool_]:
        """Return the ``(F,)`` bool mask of frames where the identity is present.

        Presence is ``jabs.pose.slot_usable`` when the file has it, else at
        least three valid keypoints other than the tail ones.

        Args:
            identity: Identity index.
        """
        return self._identity_mask[identity]

    def get_reduced_point_mask(self) -> npt.NDArray[np.bool_]:
        """Return a ``(K,)`` bool mask of keypoints valid for any identity in any frame."""
        return np.any(self._point_mask, axis=(0, 1))

    def get_connected_segments(self) -> tuple[tuple[KeypointIndex, ...], ...]:
        """Return the keypoint segments to draw the skeleton with.

        The NVSN segments when none of the keypoints NVSN pose lacks is ever
        valid, the full segments otherwise.
        """
        if self._connected_segments is None:
            reduced = self.get_reduced_point_mask()
            self._connected_segments = (
                FULL_CONNECTED_SEGMENTS
                if np.any(reduced[list(_MISSING_FROM_NVSN)])
                else NVSN_CONNECTED_SEGMENTS
            )
        return self._connected_segments

    def get_bounding_boxes(self, identity: int) -> npt.NDArray[np.float32] | None:
        """Return the ``(F, 2, 2)`` bounding boxes of an identity, or None.

        Args:
            identity: Identity index.

        Returns:
            ``[[upper_left_x, upper_left_y], [lower_right_x, lower_right_y]]``
            per frame, or None when the file has no bounding boxes.

        Raises:
            ValueError: If the identity is out of range.
        """
        if self._bboxes is None:
            return None
        self._check_identity(identity)
        return self._bboxes[identity]

    def identity_index_to_display(self, identity_index: int) -> str:
        """Return the display name of an identity.

        Args:
            identity_index: Identity index.

        Returns:
            The external identity name when there is one, else the index.
        """
        if self.external_identities and 0 <= identity_index < len(self.external_identities):
            return self.external_identities[identity_index]
        return str(identity_index)

    # -- geometry ---------------------------------------------------------

    def compute_bearing(self, points: npt.NDArray[np.floating], use_nose: bool = False) -> float:
        """Return the bearing in degrees of one frame's ``(K, 2)`` points.

        From the base of the tail to the base of the neck.

        Args:
            points: One frame's points.
            use_nose: Use the nose instead of the base of the neck, for pose
                that lacks the base of the neck.
        """
        front = KeypointIndex.NOSE if use_nose else KeypointIndex.BASE_NECK
        offset_xy = points[front.value].astype(np.float32) - points[
            KeypointIndex.BASE_TAIL.value
        ].astype(np.float32)
        return np.degrees(np.arctan2(offset_xy[1], offset_xy[0]))

    def compute_all_bearings(self, identity: int) -> npt.NDArray[np.float32]:
        """Return the ``(F,)`` float32 bearing of an identity, NaN where absent.

        Equal to :meth:`compute_bearing` of each frame's points, falling back
        to the nose when the base of the neck is never valid.

        Args:
            identity: Identity index.
        """
        use_nose = not self.get_reduced_point_mask()[KeypointIndex.BASE_NECK.value]
        if use_nose:
            logger.warning("Falling back to using nose keypoint for bearing computation")
        points, _ = self.get_identity_poses(identity)
        present = self.identity_mask(identity)
        front = KeypointIndex.NOSE if use_nose else KeypointIndex.BASE_NECK
        offset_xy = points[present, front.value].astype(np.float32) - points[
            present, KeypointIndex.BASE_TAIL.value
        ].astype(np.float32)
        bearings = np.full(self.num_frames, np.nan, dtype=np.float32)
        bearings[present] = np.degrees(np.arctan2(offset_xy[:, 1], offset_xy[:, 0]))
        return bearings

    def _hull_cache_path(self, identity: int) -> Path | None:
        """Return the convex hull cache file of an identity, or None without a cache."""
        if self._cache_dir is None:
            return None
        # a conversion's hulls also depend on the converter that produced it
        key = self._hash
        if self._converter_revision is not None:
            key = f"{key}-r{self._converter_revision}"
        return (
            self._cache_dir
            / "convex_hulls"
            / _video_base(self._path)
            / key
            / f"convex_hulls_{identity}.pickle"
        )

    def get_identity_convex_hulls(self, identity: int) -> list[BaseGeometry | None]:
        """Return the per-frame convex hull of an identity's body keypoints.

        Hulls use every valid keypoint except the tail ones. They are cached
        in memory and, with a cache directory, on disk under
        ``convex_hulls/<video base>/<hash>/`` (``<hash>-r<converter revision>``
        for a converted legacy file), so a changed pose file, or a conversion
        by another converter revision, never reuses another file's hulls.

        Args:
            identity: Identity index.

        Returns:
            List of length F, None where there are fewer than three valid
            non-tail keypoints.
        """
        if identity in self._convex_hull_cache:
            return self._convex_hull_cache[identity]

        convex_hulls = None
        path = self._hull_cache_path(identity)
        if path is not None:
            # the project's own cache, written below, as the legacy readers do
            try:
                with path.open("rb") as f:
                    convex_hulls = joblib.load(f)
            except Exception:
                # absent or unreadable: recompute it
                logger.debug("No usable convex hull cache at %s", path)

        if convex_hulls is None:
            points, point_masks = self.get_identity_poses(identity)
            body = [k.value for k in KeypointIndex if k.name not in TAIL_KEYPOINTS]
            body_points = points[:, body, :]
            body_point_masks = point_masks[:, body]
            convex_hulls = []
            for frame_index in range(self.num_frames):
                frame_mask = body_point_masks[frame_index]
                if frame_mask.sum() >= _MIN_BODY_KEYPOINTS:
                    convex_hulls.append(
                        MultiPoint(body_points[frame_index, frame_mask]).convex_hull
                    )
                else:
                    convex_hulls.append(None)
            if path is not None:
                try:
                    path.parent.mkdir(mode=0o775, parents=True, exist_ok=True)
                    with path.open("wb") as f:
                        joblib.dump(convex_hulls, f)
                except OSError:
                    logger.warning("Unable to cache convex hulls at %s", path, exc_info=True)

        self._convex_hull_cache[identity] = convex_hulls
        return convex_hulls

    # -- segmentation -----------------------------------------------------

    def _segmentation_block(self, block: int) -> _SegmentationBlock:
        """Return one cached 300-frame segmentation window across every slot."""
        cached = self._segmentation_cache.get(block)
        if cached is not None:
            self._segmentation_cache.move_to_end(block)
            return cached
        start = block * _SEGMENTATION_BLOCK
        frames = slice(start, min(start + _SEGMENTATION_BLOCK, self.num_frames))
        flags = None
        if _EXTERNAL_FLAG in self._component_ids:
            flags = self._reader.read(_EXTERNAL_FLAG, frames).astype(bool, copy=False)
        loaded: _SegmentationBlock = (
            self._reader.read(_CONTOURS, frames).astype(np.int32, copy=False),
            self._reader.read(_CONTOUR_COUNT, frames).astype(np.uint32, copy=False),
            self._reader.read(_CONTOUR_LENGTH, frames).astype(np.uint32, copy=False),
            flags,
        )
        self._segmentation_cache[block] = loaded
        if len(self._segmentation_cache) > _SEGMENTATION_CACHE_SIZE:
            self._segmentation_cache.popitem(last=False)
        return loaded

    def _require_segmentation(self) -> None:
        """Raise NotImplementedError when the file has no segmentation."""
        if not self._has_segmentation:
            raise NotImplementedError(f"{self._path.name} has no segmentation data")

    def segmentation_window(self, identity: int, start: int, stop: int) -> SegmentationWindow:
        """Return an identity's segmentation contours for frames ``[start, stop)``.

        Valid contours are packed first on each frame, as the file stores
        them. Windows are read from the file 300 frames at a time, across
        every slot, and the last few are kept.

        Args:
            identity: Identity index.
            start: First frame, inclusive.
            stop: Last frame, exclusive.

        Returns:
            The window; ``contours`` is ``(stop - start, C, P, 2)`` int32.

        Raises:
            NotImplementedError: If the file has no segmentation.
            ValueError: If the identity or frame range is out of bounds.
        """
        self._require_segmentation()
        self._check_identity(identity)
        if not 0 <= start <= stop <= self.num_frames:
            raise ValueError(f"frame range [{start}, {stop}) is not within [0, {self.num_frames})")
        if start == stop:
            contours, points = self._segmentation_capacity
            has_flags = _EXTERNAL_FLAG in self._component_ids
            return SegmentationWindow(
                contours=np.zeros((0, contours, points, 2), dtype=np.int32),
                contour_count=np.zeros(0, dtype=np.uint32),
                contour_length=np.zeros((0, contours), dtype=np.uint32),
                external_flag=np.zeros((0, contours), dtype=bool) if has_flags else None,
                start=start,
            )

        parts: list[_SegmentationBlock] = []
        for block in range(start // _SEGMENTATION_BLOCK, (stop - 1) // _SEGMENTATION_BLOCK + 1):
            offset = block * _SEGMENTATION_BLOCK
            rows = slice(
                max(start, offset) - offset, min(stop, offset + _SEGMENTATION_BLOCK) - offset
            )
            contours, count, length, flags = self._segmentation_block(block)
            parts.append(
                (
                    contours[rows, identity],
                    count[rows, identity],
                    length[rows, identity],
                    None if flags is None else flags[rows, identity],
                )
            )
        flag_parts = [p[3] for p in parts]
        return SegmentationWindow(
            contours=np.concatenate([p[0] for p in parts]),
            contour_count=np.concatenate([p[1] for p in parts]),
            contour_length=np.concatenate([p[2] for p in parts]),
            external_flag=None if flag_parts[0] is None else np.concatenate(flag_parts),
            start=start,
        )

    def iter_segmentation(self, identity: int, block: int = 300) -> Iterator[SegmentationWindow]:
        """Yield consecutive segmentation windows covering every frame.

        Arguments are validated on the call, not on the first ``next()``.

        Args:
            identity: Identity index.
            block: Frames per window; the last window may be shorter.

        Raises:
            NotImplementedError: If the file has no segmentation.
            ValueError: If ``block`` is not positive or the identity is out of range.
        """
        self._require_segmentation()
        self._check_identity(identity)
        if block <= 0:
            raise ValueError(f"block must be positive, got {block}")
        return self._segmentation_windows(identity, block)

    def _segmentation_windows(self, identity: int, block: int) -> Iterator[SegmentationWindow]:
        """Yield the windows of :meth:`iter_segmentation`."""
        for start in range(0, self.num_frames, block):
            yield self.segmentation_window(identity, start, min(start + block, self.num_frames))
