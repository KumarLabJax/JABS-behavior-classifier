from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameKeypointsData:
    """Collection of per-frame keypoint predictions.

    Attributes:
        frames: Per-frame keypoint predictions in video pixel space.
    """

    frames: list["FrameKeypoints"]


@dataclass(frozen=True)
class FrameKeypoints:
    """Keypoints for a single frame.

    Attributes:
        frame_index: Frame index within the video.
        keypoints: Keypoint coordinates in video pixel space (K, 2).
        confidence: Per-frame confidence score.
    """

    frame_index: int
    keypoints: np.ndarray
    confidence: np.ndarray | None = None


@dataclass(frozen=True)
class KeypointAnnotation:
    """Aggregated keypoint annotation for a video.

    Attributes:
        keypoints: Aggregated keypoints in video pixel space (K, 2).
        kept_frame_indices: Frame indices retained after filtering.
        mean_confidence: Mean confidence across kept frames.
    """

    keypoints: np.ndarray
    kept_frame_indices: list[int]
    mean_confidence: float | None = None

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints)
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError("KeypointAnnotation.keypoints must have shape (K, 2)")
