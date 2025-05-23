import typing

import numpy as np

from jabs.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from .social_distance import ClosestIdentityInfo
    from jabs.pose_estimation import PoseEstimation


class ClosestDistances(Feature):
    _name = "closest_distances"
    _min_pose = 3

    def __init__(
        self,
        poses: "PoseEstimation",
        pixel_scale: float,
        social_distance_info: "ClosestIdentityInfo",
    ):
        super().__init__(poses, pixel_scale)
        self._social_distance_info = social_distance_info

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """compute the value of the per frame features for a specific identity

        Args:
            identity: identity to compute features for

        Returns:
            dict with feature values
        """
        return {
            "closest social distance": self._social_distance_info.compute_distances(
                self._social_distance_info.closest_identities
            )
        }
