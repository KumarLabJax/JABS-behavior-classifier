import typing

import numpy as np

from jabs.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation

    from .social_distance import ClosestIdentityInfo


class ClosestDistances(Feature):
    """
    Computes the distance between a subject and the nearest other identity for each frame.

    This feature calculates, for each frame, the distance between the subject and the closest other identity,
    based on pose estimation data. The result is useful for analyzing proximity-based social interactions.

    Args:
        poses (PoseEstimation): Pose estimation data for a video.
        pixel_scale (float): Scale factor to convert pixel distances to real-world units (cm).
        social_distance_info (ClosestIdentityInfo): Object providing pre-computed closest identity information.
    """

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
