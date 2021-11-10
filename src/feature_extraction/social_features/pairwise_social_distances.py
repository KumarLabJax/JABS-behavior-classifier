import typing

import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from .social_distance import ClosestIdentityInfo


class PairwiseSocialDistances(Feature):

    _name = 'social_pairwise_distances'

    # For social interaction we will consider a subset
    # of points to capture just the most important
    # information for social.
    _social_point_subset = [
        PoseEstimation.KeypointIndex.NOSE,
        PoseEstimation.KeypointIndex.BASE_NECK,
        PoseEstimation.KeypointIndex.BASE_TAIL,
    ]

    # total number of values created by pairwise distances between the
    # subject and closest other identity for this subset of points
    _num_social_distances = len(_social_point_subset) ** 2

    def __init__(self, poses: PoseEstimation, pixel_scale: float,
                 social_distance_info: 'ClosestIdentityInfo'):
        super().__init__(poses, pixel_scale)
        self._social_distance_info = social_distance_info
        self._poses = poses

    @property
    def feature_names(self) -> typing.List[str]:
        return [
            f"social dist. {sdn}"
            for sdn in self._get_social_distance_names()
        ]

    def per_frame(self, identity: int) -> np.ndarray:
        return self._social_distance_info.compute_pairwise_social_distances(
            self._social_point_subset,
            self._social_distance_info.closest_identities
        )

    @classmethod
    def _get_social_distance_names(cls):
        """
        get list of human readable names for each value computed by
        _compute_social_pairwise_distance
        :return: list of distance names where each is a string of the form
        "distance_name_1-distance_name_2"
        """
        dist_names = []
        for kpi1 in cls._social_point_subset:
            for kpi2 in cls._social_point_subset:
                dist_names.append(f"{kpi1.name}-{kpi2.name}")
        return dist_names


class PairwiseSocialFovDistances(PairwiseSocialDistances):

    """
    PairwiseSocialFovDistances, nearly the same as the PairwiseSocialDistances,
    except closest_fov_identities is passed to compute_pairwise_social_distances
    rather than closest_identities
    """

    @property
    def name(self) -> str:
        return 'social_pairwise_fov_distances'

    @property
    def feature_names(self) -> typing.List[str]:
        return [
            f"social fov dist. {sdn}"
            for sdn in self._get_social_distance_names()
        ]

    def per_frame(self, identity: int) -> np.ndarray:
        return self._social_distance_info.compute_pairwise_social_distances(
            self._social_point_subset,
            self._social_distance_info.closest_fov_identities
        )
