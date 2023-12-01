import typing

import numpy as np

from src.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from .social_distance import ClosestIdentityInfo
    from src.pose_estimation import PoseEstimation


class ClosestFovDistances(Feature):

    _name = 'closest_fov_distances'
    _min_pose = 3

    def __init__(self, poses: 'PoseEstimation', pixel_scale: float,
                 social_distance_info: 'ClosestIdentityInfo'):
        super().__init__(poses, pixel_scale)
        self._social_distance_info = social_distance_info

    def per_frame(self, identity: int) -> np.ndarray:
        """
        compute the value of the per frame features for a specific identity
        :param identity: identity to compute features for
        :return: dict with feature values
        """
        return {'closest social distance in FoV': self._social_distance_info.compute_distances(self._social_distance_info.closest_fov_identities)}
