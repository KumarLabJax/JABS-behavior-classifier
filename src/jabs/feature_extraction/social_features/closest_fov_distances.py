import typing

import numpy as np

from jabs.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation

    from .social_distance import ClosestIdentityInfo


class ClosestFovDistances(Feature):
    """Computes the closest distance between a subject and the nearest other identity within its field of view (FoV).

    Args:
        poses (PoseEstimation): Pose estimation data for one video.
        pixel_scale (float): Scale factor to convert pixel distances to cm.
        social_distance_info (ClosestIdentityInfo): Object providing closest identity and FoV information.
    """

    _name = "closest_fov_distances"
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
            "closest social distance in FoV": self._social_distance_info.compute_distances(
                self._social_distance_info.closest_fov_identities
            )
        }
