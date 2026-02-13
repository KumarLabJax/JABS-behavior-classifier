from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoseData:
    """Canonical representation of pose estimation data.

    Attributes:
        points: Keypoint coordinates of shape (num_identities, num_frames, num_keypoints, 2).
        point_mask: Boolean mask indicating valid keypoints, shape (num_identities, num_frames, num_keypoints).
        identity_mask: Boolean mask indicating if identity is present in frame, shape (num_identities, num_frames).
        body_parts: List of names for the keypoints.
        fps: Frames per second of the source video.
        cm_per_pixel: Optional scale factor for converting pixels to centimeters.
        bounding_boxes: Optional bounding boxes of shape (num_identities, num_frames, 2, 2).
            Format is [[upper_left_x, upper_left_y], [lower_right_x, lower_right_y]].
        segmentation_data: Optional segmentation masks or data.
        static_objects: Dictionary of static objects (e.g., 'lixit') and their positions.
        metadata: Dictionary for any additional provenance or experimental metadata.
    """

    points: np.ndarray
    point_mask: np.ndarray
    identity_mask: np.ndarray
    body_parts: list[str]
    fps: int
    cm_per_pixel: float | None = None
    bounding_boxes: np.ndarray | None = None
    segmentation_data: np.ndarray | None = None
    static_objects: dict[str, np.ndarray] = field(default_factory=dict)
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
