from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DynamicObjectData:
    """Container for a single dynamic object read from a pose file.

    Dynamic objects are objects that may change position or count over time, but
    are not predicted every frame.  Only frames listed in sample_indices have
    valid predictions.  Coordinates are always stored in (x, y) order.

    points is always 4-D regardless of how many keypoints each object
    instance has.  Single-keypoint objects (e.g. fecal boli) have
    n_keypoints=1 after normalization on read.

    Attributes:
        points: Detected keypoint coordinates in (x, y) order, shape
            (n_predictions, max_count, n_keypoints, 2).
        counts: Number of valid detected objects for each prediction, shape
            (n_predictions,).
        sample_indices: Frame indices at which each prediction was made, shape
            (n_predictions,).
    """

    points: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]
    sample_indices: npt.NDArray[np.int64]


@dataclass(frozen=True)
class SegmentationData:
    """Container for instance segmentation contours read from a pose file.

    Segmentation is stored as polygon contours rather than pixel masks.  An identity may
    need more than one contour on a frame: an outer boundary plus one or more holes, or a
    body that an occluder splits into disjoint parts.  Because different frames need
    different numbers of contours and vertices, ``contours`` is padded out to the largest
    of each and ``vertex_counts`` records how much of every slot is real.

    Coordinates are in ``(x, y)`` order, matching :attr:`PoseData.points`.

    Attributes:
        contours: Contour vertex positions in (x, y) order, shape
            (num_identities, num_frames, num_contours, num_vertices, 2).  Only the first
            ``vertex_counts`` vertices of each contour slot hold a position; the rest is
            padding.
        vertex_counts: Number of valid vertices in each contour slot, shape
            (num_identities, num_frames, num_contours).  0 means the slot holds no
            contour on that frame.
        is_external: True where the contour slot is an external boundary (an outer edge
            of the animal), False where it is an internal boundary (a hole), shape
            (num_identities, num_frames, num_contours).  Has no meaning where
            ``vertex_counts`` is 0.
    """

    contours: npt.NDArray[np.int32]
    vertex_counts: npt.NDArray[np.uint32]
    is_external: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate that the three arrays describe the same contour slots.

        Raises:
            ValueError: If the arrays have the wrong number of dimensions, if the
                trailing axis of ``contours`` is not 2, or if the shapes disagree.
        """
        if self.contours.ndim != 5:
            raise ValueError(
                "contours must have 5 dimensions "
                f"(ident, frame, contour, vertex, axis), got {self.contours.ndim}"
            )
        if self.contours.shape[-1] != 2:
            raise ValueError(
                f"contours last dimension must be 2 (x, y), got {self.contours.shape[-1]}"
            )

        expected = self.contours.shape[:3]
        if self.vertex_counts.shape != expected:
            raise ValueError(
                f"vertex_counts shape {self.vertex_counts.shape} must match the first "
                f"three dimensions of contours {expected}"
            )
        if self.is_external.shape != expected:
            raise ValueError(
                f"is_external shape {self.is_external.shape} must match the first "
                f"three dimensions of contours {expected}"
            )


@dataclass(frozen=True)
class PoseData:
    """Canonical representation of pose estimation data.

    Attributes:
        points: Keypoint coordinates of shape (num_identities, num_frames, num_keypoints, 2).
        point_mask: Boolean mask indicating valid keypoints, shape (num_identities, num_frames, num_keypoints).
        identity_mask: Boolean mask indicating if identity is present in frame, shape (num_identities, num_frames).
        body_parts: List of names for the keypoints.
        edges: List of tuples defining connections between keypoints (e.g., for skeleton visualization).
        fps: Frames per second of the source video.
        cm_per_pixel: Optional scale factor for converting pixels to centimeters.
        bounding_boxes: Optional bounding boxes of shape (num_identities, num_frames, 2, 2).
            Format is [[upper_left_x, upper_left_y], [lower_right_x, lower_right_y]].
        confidence: Optional per-keypoint confidence scores, shape
            (num_identities, num_frames, num_keypoints).  None when unknown.
        segmentation_data: Optional instance segmentation contours, as a
            SegmentationData.  None when the source pose file has none (every
            version before v6, and v6+ files generated without segmentation).
        static_objects: Dictionary of static objects (e.g., 'lixit') and their positions.
        dynamic_objects: Dictionary of dynamic objects (e.g., 'fecal_boli') and their data.
        external_ids: Optional list of external identifiers for each identity.
            Maps an identity index to an external ID string.
        subjects: Optional per-animal biological metadata, keyed by identity name
            (matching the values in external_ids).  Each value is a free-form
            dict; standard keys are subject_id, sex, genotype,
            strain, age, weight, species, description.
        metadata: Dictionary for any additional provenance or experimental metadata.
    """

    points: np.ndarray
    point_mask: np.ndarray
    identity_mask: np.ndarray
    body_parts: list[str]
    edges: list[tuple[int, int]]
    fps: int
    cm_per_pixel: float | None = None
    bounding_boxes: np.ndarray | None = None
    confidence: np.ndarray | None = None
    segmentation_data: SegmentationData | None = None
    static_objects: dict[str, np.ndarray] = field(default_factory=dict)
    dynamic_objects: dict[str, DynamicObjectData] = field(default_factory=dict)
    external_ids: list[str] | None = None
    subjects: dict[str, dict] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate dimensions of initialized data."""
        if self.points.ndim != 4:
            raise ValueError(
                f"points must have 4 dimensions (ident, frame, keypoint, axis), got {self.points.ndim}"
            )

        num_idents, num_frames, num_keypoints, num_axes = self.points.shape

        if num_axes != 2:
            raise ValueError(f"points last dimension must be 2 (x, y), got {num_axes}")

        if self.point_mask.shape != (num_idents, num_frames, num_keypoints):
            raise ValueError(
                f"point_mask shape {self.point_mask.shape} must match points "
                f"dimensions {(num_idents, num_frames, num_keypoints)}"
            )

        if self.identity_mask.shape != (num_idents, num_frames):
            raise ValueError(
                f"identity_mask shape {self.identity_mask.shape} must match "
                f"points dimensions {(num_idents, num_frames)}"
            )

        if len(self.body_parts) != num_keypoints:
            raise ValueError(
                f"Length of body_parts ({len(self.body_parts)}) must match "
                f"number of keypoints ({num_keypoints})"
            )

        if self.bounding_boxes is not None and self.bounding_boxes.shape != (
            num_idents,
            num_frames,
            2,
            2,
        ):
            raise ValueError(
                f"bounding_boxes shape {self.bounding_boxes.shape} must "
                f"be {(num_idents, num_frames, 2, 2)}"
            )

        if self.confidence is not None and self.confidence.shape != (
            num_idents,
            num_frames,
            num_keypoints,
        ):
            raise ValueError(
                f"confidence shape {self.confidence.shape} must match points "
                f"dimensions {(num_idents, num_frames, num_keypoints)}"
            )

        if self.segmentation_data is not None:
            seg_shape = self.segmentation_data.contours.shape[:2]
            if seg_shape != (num_idents, num_frames):
                raise ValueError(
                    f"segmentation_data covers {seg_shape} (identities, frames) but points "
                    f"has {(num_idents, num_frames)}"
                )

        invalid_edges = [e for e in self.edges if e[0] >= num_keypoints or e[1] >= num_keypoints]
        if invalid_edges:
            raise ValueError(
                f"edges contain out-of-range keypoint indices (num_keypoints={num_keypoints}): "
                f"{invalid_edges}"
            )

        if self.external_ids is not None:
            if len(self.external_ids) != num_idents:
                raise ValueError(
                    f"external_ids length ({len(self.external_ids)}) must match "
                    f"num_identities ({num_idents})"
                )
            if len(set(self.external_ids)) != len(self.external_ids):
                duplicates = [x for x in self.external_ids if self.external_ids.count(x) > 1]
                raise ValueError(f"external_ids must be unique, found duplicates: {duplicates}")
