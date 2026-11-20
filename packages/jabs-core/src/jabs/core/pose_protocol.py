"""The ``PoseLike`` protocol: the pose API JABS callers program against.

Both the legacy per-version readers (:class:`jabs.core.abstract.PoseEstimation`
and its subclasses) and the unified ``Pose`` class implement this protocol, so
callers typed against ``PoseLike`` work with either.
"""

import re
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from shapely.geometry.base import BaseGeometry

from jabs.core.enums import PoseCapability
from jabs.core.pose_skeleton import KeypointIndex
from jabs.core.types import DynamicObjectData, SegmentationWindow

__all__ = [
    "NATIVE_SOURCE_FORMAT",
    "PoseLike",
    "source_format_from_path",
]

# source_format of a native unified pose file (``<video stem>_pose.h5``)
NATIVE_SOURCE_FORMAT = "jabs.pose-file"

_LEGACY_NAME_RE = re.compile(r"_pose_est_v(\d+)\.h5$")
_NATIVE_NAME_RE = re.compile(r"_pose\.h5$")


def source_format_from_path(path: Path) -> str | None:
    """Derive a pose source's ``source_format`` from its file name.

    A legacy file named ``*_pose_est_v<N>.h5`` gives ``"pose_est_v<N>"``; a
    native unified file named ``*_pose.h5`` gives ``"jabs.pose-file"``. The
    file's contents are not inspected: legacy ``version`` attributes are not
    reliable.

    Args:
        path: Path of the pose file the user has.

    Returns:
        The source format, or None when the name follows neither convention.
    """
    match = _LEGACY_NAME_RE.search(path.name)
    if match is not None:
        return f"pose_est_v{int(match.group(1))}"
    if _NATIVE_NAME_RE.search(path.name):
        return NATIVE_SOURCE_FORMAT
    return None


@runtime_checkable
class PoseLike(Protocol):
    """Read access to one video's pose data.

    Shapes use ``F`` for frames, ``K`` for keypoints (12) and ``I`` for identities.
    Point coordinates are ``(x, y)`` pixels. Dtypes are given for both
    implementations where they differ: legacy readers return float64 points and
    integer (0/1) masks, ``Pose`` returns float32 points and bool masks. Callers
    must treat masks as truthy values, not compare them to a dtype.

    Attributes:
        num_frames: Number of frames.
        identities: Identity indexes, ``0..num_identities-1``.
        num_identities: Number of identities.
        cm_per_pixel: Centimeters per pixel, or None when unknown.
        fps: Frames per second.
        pose_file: Path of the pose file the user has.
        hash: Hash of that file (``jabs.core.utils.hash_file``).
        format_major_version: Major version of the legacy format the data came from.
        identity_to_track: ``(I, F)`` int32 track id of each identity in each frame,
            -1 where it has no track; None when the file has no tracks (only legacy
            v3 files have them).
        static_objects: Static object name to its points, float xy.
        dynamic_objects: Dynamic object name to its data.
        num_lixit_keypoints: Keypoints per lixit, 0 without a lixit.
        external_identities: External identity names indexed by identity, or None.
        capabilities: Optional data the source provides: ``SOCIAL`` iff more than
            one identity, ``SEGMENTATION`` iff it has segmentation contours,
            ``BOUNDING_BOXES`` iff it has bounding boxes.
        has_social: ``PoseCapability.SOCIAL in capabilities``.
        has_segmentation: ``PoseCapability.SEGMENTATION in capabilities``.
        has_bounding_boxes: ``PoseCapability.BOUNDING_BOXES in capabilities``.
        source_format: ``"pose_est_v<N>"`` for a legacy source file, from its name,
            or ``"jabs.pose-file"`` for a native unified file.
        model_metadata: Parsed ``poseest`` ``model_metadata_json`` attribute, or None
            when absent, empty or not valid JSON.
    """

    @property
    def num_frames(self) -> int:
        """Number of frames."""
        ...

    @property
    def identities(self) -> list[int]:
        """Identity indexes, ``0..num_identities-1``."""
        ...

    @property
    def num_identities(self) -> int:
        """Number of identities."""
        ...

    @property
    def cm_per_pixel(self) -> float | None:
        """Centimeters per pixel, or None when unknown."""
        ...

    @property
    def fps(self) -> float:
        """Frames per second."""
        ...

    @property
    def pose_file(self) -> Path:
        """Path of the pose file the user has."""
        ...

    @property
    def hash(self) -> str:
        """Hash of the pose file."""
        ...

    @property
    def format_major_version(self) -> int:
        """Major version of the legacy pose format the data came from."""
        ...

    @property
    def identity_to_track(self) -> npt.NDArray[np.int32] | None:
        """``(I, F)`` int32 track ids, -1 where no track; None without tracks."""
        ...

    @property
    def static_objects(self) -> dict[str, npt.NDArray[np.floating]]:
        """Static object name to points: ``(O, P, 2)``, or ``(O, 2)`` when P == 1."""
        ...

    @property
    def dynamic_objects(self) -> dict[str, DynamicObjectData]:
        """Dynamic object name to its data; empty when the source has none."""
        ...

    @property
    def num_lixit_keypoints(self) -> int:
        """Keypoints per lixit (1 or 3), 0 without a lixit."""
        ...

    @property
    def external_identities(self) -> list[str] | None:
        """External identity names indexed by identity, or None."""
        ...

    @property
    def capabilities(self) -> frozenset[PoseCapability]:
        """Optional data the source provides."""
        ...

    @property
    def has_social(self) -> bool:
        """True when the source has more than one identity."""
        ...

    @property
    def has_segmentation(self) -> bool:
        """True when the source has segmentation contours."""
        ...

    @property
    def has_bounding_boxes(self) -> bool:
        """True when the source has bounding boxes."""
        ...

    @property
    def source_format(self) -> str:
        """``"pose_est_v<N>"`` for a legacy source, ``"jabs.pose-file"`` for native."""
        ...

    @property
    def model_metadata(self) -> dict[str, Any] | None:
        """Parsed ``model_metadata_json``, or None."""
        ...

    def close(self) -> None:
        """Release any open file handles; a no-op when none are held."""
        ...

    def __enter__(self) -> "PoseLike":
        """Return self, for use as a context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Call :meth:`close`."""
        ...

    def has_static_object(self, name: str) -> bool:
        """Return True when the source has the named static object.

        Args:
            name: Static object name, e.g. ``"lixit"``.
        """
        ...

    def get_points(
        self, frame_index: int, identity: int, scale: float | None = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[Any]] | tuple[None, None]:
        """Return one frame's points and point mask for an identity.

        Args:
            frame_index: Frame index.
            identity: Identity index.
            scale: Optional factor applied to the points (e.g. ``cm_per_pixel``).

        Returns:
            ``(K, 2)`` points and ``(K,)`` point mask, or ``(None, None)`` when the
            identity is absent from the frame.
        """
        ...

    def get_identity_poses(
        self, identity: int, scale: float | None = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[Any]]:
        """Return all points and point masks for an identity.

        Args:
            identity: Identity index.
            scale: Optional factor applied to the points (e.g. ``cm_per_pixel``).

        Returns:
            ``(F, K, 2)`` points and ``(F, K)`` point mask. Points are only
            meaningful where the mask is true.
        """
        ...

    def get_identity_point_mask(self, identity: int) -> npt.NDArray[Any]:
        """Return the ``(F, K)`` point mask for an identity.

        Args:
            identity: Identity index.
        """
        ...

    def get_reduced_point_mask(self) -> npt.NDArray[np.bool_]:
        """Return a ``(K,)`` bool mask of keypoints valid for any identity in any frame."""
        ...

    def get_connected_segments(self) -> tuple[tuple[KeypointIndex, ...], ...]:
        """Return the keypoint segments to draw the skeleton with."""
        ...

    def identity_mask(self, identity: int) -> npt.NDArray[Any]:
        """Return the ``(F,)`` mask of frames where the identity is present.

        Args:
            identity: Identity index.
        """
        ...

    def get_dynamic_object(self, name: str) -> DynamicObjectData | None:
        """Return the named dynamic object, or None when absent.

        Args:
            name: Dynamic object name, e.g. ``"fecal_boli"``.
        """
        ...

    def get_identity_convex_hulls(self, identity: int) -> list[BaseGeometry | None]:
        """Return the per-frame convex hull of an identity's body keypoints.

        Args:
            identity: Identity index.

        Returns:
            List of length F, None where there is no hull (fewer than 3 valid
            non-tail keypoints).
        """
        ...

    def compute_bearing(self, points: npt.NDArray[np.floating], use_nose: bool = False) -> float:
        """Return the bearing in degrees of one frame's ``(K, 2)`` points.

        Args:
            points: One frame's points.
            use_nose: Use the nose instead of the base of the neck.
        """
        ...

    def compute_all_bearings(self, identity: int) -> npt.NDArray[np.float32]:
        """Return the ``(F,)`` float32 bearing of an identity, NaN where absent.

        Args:
            identity: Identity index.
        """
        ...

    def identity_index_to_display(self, identity_index: int) -> str:
        """Return the display name of an identity.

        Args:
            identity_index: Identity index.
        """
        ...

    def get_bounding_boxes(self, identity: int) -> npt.NDArray[np.float32] | None:
        """Return the ``(F, 2, 2)`` bounding boxes of an identity, or None.

        Args:
            identity: Identity index.

        Returns:
            ``[[upper_left_x, upper_left_y], [lower_right_x, lower_right_y]]`` per
            frame, or None when the source has no bounding boxes.
        """
        ...

    def segmentation_window(self, identity: int, start: int, stop: int) -> SegmentationWindow:
        """Return an identity's segmentation contours for frames ``[start, stop)``.

        Valid contours are packed first on each frame (``contour_count`` of them,
        in source order), ``contour_length`` gives each contour's valid points,
        padding is unspecified. ``contours`` is ``(stop - start, C, P, 2)`` int32
        ``(x, y)``.

        Args:
            identity: Identity index.
            start: First frame, inclusive.
            stop: Last frame, exclusive.

        Raises:
            NotImplementedError: If the source has no segmentation.
            ValueError: If the identity or frame range is out of bounds.
        """
        ...

    def iter_segmentation(self, identity: int, block: int = 300) -> Iterator[SegmentationWindow]:
        """Yield consecutive segmentation windows covering every frame.

        ``NotImplementedError`` and the ``ValueError`` for a non-positive
        ``block`` are raised on the call, not on the first ``next()``.

        An out-of-range ``identity`` is a ``ValueError`` too, raised at a time
        that depends on the implementation: ``Pose`` checks it on the call, the
        legacy readers only when the first window is read.

        Args:
            identity: Identity index.
            block: Frames per window; the last window may be shorter.

        Raises:
            NotImplementedError: If the source has no segmentation.
            ValueError: If ``block`` is not positive, or (``Pose``, on the call;
                legacy readers, lazily) the identity is out of range.
        """
        ...
