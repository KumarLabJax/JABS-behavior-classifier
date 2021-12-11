import typing

import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature

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


def _init_feature_names():
    """
    build a list of feature names that correspond to the columns returned
    by the PairwiseSocialDistances and PairwiseSocialFovDistances modules
    """
    dist_names = []
    for kpi1 in _social_point_subset:
        for kpi2 in _social_point_subset:
            dist_names.append(f"{kpi1.name}-{kpi2.name}")
    return [f"social dist. {sdn}" for sdn in dist_names]


class PairwiseSocialDistances(Feature):

    _name = 'social_pairwise_distances'
    _feature_names = _init_feature_names()
    _min_pose = 3


    # total number of values created by pairwise distances between the
    # subject and closest other identity for this subset of points
    _num_social_distances = len(_social_point_subset) ** 2

    def __init__(self, poses: PoseEstimation, pixel_scale: float,
                 social_distance_info: 'ClosestIdentityInfo'):
        super().__init__(poses, pixel_scale)
        self._social_distance_info = social_distance_info
        self._poses = poses

    def per_frame(self, identity: int) -> np.ndarray:
        """
        compute the value of the per frame features for a specific identity
        :param identity: identity to compute features for
        :return: np.ndarray with feature values
        """
        return self._social_distance_info.compute_pairwise_social_distances(
            _social_point_subset,
            self._social_distance_info.closest_identities
        )


class PairwiseSocialFovDistances(PairwiseSocialDistances):

    """
    PairwiseSocialFovDistances, nearly the same as the PairwiseSocialDistances,
    except closest_fov_identities is passed to compute_pairwise_social_distances
    rather than closest_identities
    """

    _name = 'social_pairwise_fov_distances'
    _feature_names = [
        n.replace('social dist.', 'social fov dist.')
        for n in PairwiseSocialDistances._feature_names
    ]

    def per_frame(self, identity: int) -> np.ndarray:
        """
                compute the value of the per frame features for a specific identity
                :param identity: identity to compute features for
                :return: np.ndarray with feature values
                """
        return self._social_distance_info.compute_pairwise_social_distances(
            _social_point_subset,
            self._social_distance_info.closest_fov_identities
        )
