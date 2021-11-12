import typing

import numpy as np

from src.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from .social_distance import ClosestIdentityInfo
    from src.pose_estimation import PoseEstimation


class ClosestDistances(Feature):

    _name = 'closest_distances'
    _feature_names = ['closest social distance']

    def __init__(self, poses: 'PoseEstimation', pixel_scale: float,
                 social_distance_info: 'ClosestIdentityInfo'):
        super().__init__(poses, pixel_scale)
        self._social_distance_info = social_distance_info

    def per_frame(self, identity: int) -> np.ndarray:
        """
        compute the value of the per frame features for a specific identity
        :param identity: identity to compute features for
        :return: np.ndarray with feature values
        """
        return self._social_distance_info.compute_distances(
            self._social_distance_info.closest_identities)
