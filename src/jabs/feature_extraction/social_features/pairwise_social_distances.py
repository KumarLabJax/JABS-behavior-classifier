import typing

import numpy as np

from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation

if typing.TYPE_CHECKING:
    from .social_distance import ClosestIdentityInfo

# For social interaction we will consider a subset
# of points to capture just the most important
# information for social.
_social_point_subset = [
    PoseEstimation.KeypointIndex.NOSE,
    PoseEstimation.KeypointIndex.BASE_NECK,
    PoseEstimation.KeypointIndex.BASE_TAIL,
]


class PairwiseSocialDistances(Feature):
    """Computes pairwise social distances between a subject and its closest other identity for a subset of keypoints.

    This feature extracts, for each frame, the distances between all pairs of keypoints in a predefined subset
    for the subject and the closest other identity. The distances are used to characterize social interactions
    based on pose estimation data.

    Args:
        poses (PoseEstimation): Pose estimation data for all subjects.
        pixel_scale (float): Scale factor to convert pixel distances to real-world units.
        social_distance_info (ClosestIdentityInfo): Object providing closest identity information.

    Methods:
        per_frame(identity): Computes per-frame pairwise social distances for a given identity.
    """

    _name = "social_pairwise_distances"
    _min_pose = 3

    # total number of values created by pairwise distances between the
    # subject and closest other identity for this subset of points
    _num_social_distances = len(_social_point_subset) ** 2

    def __init__(
        self,
        poses: PoseEstimation,
        pixel_scale: float,
        social_distance_info: "ClosestIdentityInfo",
    ):
        super().__init__(poses, pixel_scale)
        self._social_distance_info = social_distance_info
        self._poses = poses

    def per_frame(self, identity: int) -> dict:
        """compute the value of the per frame features for a specific identity

        Args:
            identity: identity to compute features for

        Returns:
            dict with feature values
        """
        return self._social_distance_info.compute_pairwise_social_distances(
            _social_point_subset, self._social_distance_info.closest_identities
        )


class PairwiseSocialFovDistances(PairwiseSocialDistances):
    """compute pairwise social distances between subject and closest other animal in field of view

    nearly the same as the PairwiseSocialDistances, except closest_fov_identities is passed to
    compute_pairwise_social_distances rather than closest_identities
    """

    _name = "social_pairwise_fov_distances"

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """compute the value of the per frame features for a specific identity

        Args:
            identity: identity to compute features for

        Returns:
            np.ndarray with feature values
        """
        return self._social_distance_info.compute_pairwise_social_distances(
            _social_point_subset, self._social_distance_info.closest_fov_identities
        )
