"""Estimate the video frame size a pose file was produced from."""

import logging
import math

import numpy as np

from jabs.pose_estimation import PoseEstimation

logger = logging.getLogger(__name__)

# Large enough for any camera JABS is used with, small enough to keep frames in memory
MAX_FRAME_DIMENSION = 8192

# Estimated sizes are rounded up to a multiple of this, so an arena whose animals never
# reach the frame edge still comes out at the video's size (e.g. 800x800).
FRAME_SIZE_STEP = 100


def estimate_frame_size(pose_est: PoseEstimation) -> tuple[int, int]:
    """Estimate the width and height of the video a pose file was produced from.

    Pose files do not record the frame size, so it is taken from the largest x and y
    coordinates of the identities' valid keypoints and of the arena landmarks (static
    objects), each rounded up to a multiple of :data:`FRAME_SIZE_STEP` and capped at
    :data:`MAX_FRAME_DIMENSION`, so a stray point far outside the frame can't size it
    past what the frame size dialog accepts.

    Args:
        pose_est: Pose estimation to size the frames for.

    Returns:
        Estimated (width, height) in pixels; at least ``FRAME_SIZE_STEP`` and at most
        ``MAX_FRAME_DIMENSION`` each, even for a pose file with no valid points.
    """
    max_x = 0.0
    max_y = 0.0

    for identity in pose_est.identities:
        points, mask = pose_est.get_identity_poses(identity)
        valid = points[mask.astype(bool)]
        if valid.size:
            max_x = max(max_x, float(valid[:, 0].max()))
            max_y = max(max_y, float(valid[:, 1].max()))

    for static_object in pose_est.static_objects.values():
        coords = np.asarray(static_object, dtype=np.float64).reshape(-1, 2)
        if coords.size:
            max_x = max(max_x, float(coords[:, 0].max()))
            max_y = max(max_y, float(coords[:, 1].max()))

    width, height = _round_up(max_x), _round_up(max_y)
    if width > MAX_FRAME_DIMENSION or height > MAX_FRAME_DIMENSION:
        logger.warning(
            "Estimated frame size %dx%d exceeds %d; capping it",
            width,
            height,
            MAX_FRAME_DIMENSION,
        )
    return min(width, MAX_FRAME_DIMENSION), min(height, MAX_FRAME_DIMENSION)


def _round_up(value: float) -> int:
    """Round a pixel coordinate up to a frame dimension that contains it."""
    # +1: a point at coordinate n needs a frame n + 1 pixels wide to be inside it
    return max(1, math.ceil((math.floor(value) + 1) / FRAME_SIZE_STEP)) * FRAME_SIZE_STEP
