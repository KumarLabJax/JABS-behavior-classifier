"""Work out which animal is nearest the subject on a frame.

Moved out of ``PlayerThread`` when the markers it fed became an overlay. It is kept
apart from the overlay that draws them because it is not drawing: it measures hull
distances and view angles, and an overlay repaints far more often than the frame
changes, so the result wants computing once per frame and reusing.
"""

import numpy as np

from jabs.core.utils import signed_angle_degrees
from jabs.feature_extraction.social_features.social_distance import ClosestIdentityInfo
from jabs.pose_estimation import PoseEstimation

# Beyond this half-angle either side of the subject's nose, an animal is behind it.
HALF_FOV_DEGREES = ClosestIdentityInfo.HALF_FOV_DEGREE


def closest_identity(
    pose: PoseEstimation,
    subject: int,
    frame_index: int,
    half_fov_degrees: float | None = None,
) -> int | None:
    """Return the identity nearest ``subject`` on this frame.

    Args:
        pose: Pose estimation data for the video.
        subject: Identity to measure distances from.
        frame_index: Frame to measure on.
        half_fov_degrees: When given, only consider animals within this half-angle of
            the subject's facing direction, which needs its nose and neck keypoints.
            ``None`` (or 180 and above) ignores which way the subject is facing.

    Returns:
        The nearest identity, or ``None`` when the subject has no pose on this frame,
        there is nobody else, or nobody qualifies.
    """
    idx = PoseEstimation.KeypointIndex

    ref_shape = pose.get_identity_convex_hulls(subject)[frame_index]
    if ref_shape is None:
        return None

    closest_id: int | None = None
    closest_dist: float | None = None

    for curr_id in pose.identities:
        if curr_id == subject:
            continue

        other_shape = pose.get_identity_convex_hulls(curr_id)[frame_index]
        if other_shape is None:
            continue

        curr_dist = ref_shape.distance(other_shape)

        if half_fov_degrees is None or half_fov_degrees >= 180:
            # Distance alone decides; which way the subject faces does not matter.
            if closest_dist is None or curr_dist < closest_dist:
                closest_id = curr_id
                closest_dist = curr_dist
            continue

        points, mask = pose.get_points(frame_index, subject)

        # The nose and neck are what give the subject a facing direction.
        if mask[idx.NOSE] != 1 or mask[idx.BASE_NECK] != 1:
            continue

        other_centroid = np.array((other_shape.centroid.x, other_shape.centroid.y))
        # Already wrapped to [-180, 180), which is the range the comparison needs.
        view_angle = signed_angle_degrees(
            points[idx.NOSE, :], points[idx.BASE_NECK, :], other_centroid
        )

        if abs(view_angle) <= half_fov_degrees and (
            closest_dist is None or curr_dist < closest_dist
        ):
            closest_id = curr_id
            closest_dist = curr_dist

    return closest_id
