import typing

import numpy as np
import scipy.stats

from jabs.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation

    from .social_distance import ClosestIdentityInfo


class ClosestFovAngles(Feature):
    """
    Computes the angle between a subject and the closest other identity within its field of view (FoV) for each frame.

    This feature provides, for each frame, the angle (in degrees) from the subject to the closest other identity
    that is within its field of view, based on pose estimation data. The angles are treated as circular values for
    windowed operations.

    Args:
        poses (PoseEstimation): Pose estimation data for a video.
        pixel_scale (float): Scale factor to convert pixel distances to cm.
        social_distance_info (ClosestIdentityInfo): Object providing closest identity and FoV angle information.
    """

    _name = "closest_fov_angles"
    _min_pose = 3

    # override for circular values
    _window_operations: typing.ClassVar[dict[str, typing.Callable]] = {
        "mean": lambda x: scipy.stats.circmean(
            x, low=-180, high=180, nan_policy="omit"
        ),
        "std_dev": lambda x: scipy.stats.circstd(
            x, low=-180, high=180, nan_policy="omit"
        ),
    }

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
        # this is already computed
        return {
            "angle of closest social distance in FoV": self._social_distance_info.closest_fov_angles
        }

    def window(
        self, identity: int, window_size: int, per_frame_values: dict[str, np.ndarray]
    ) -> dict:
        """compute window feature values for a given identities per frame values"""
        # need to override to use special method for computing window features with circular values
        return self._window_circular(identity, window_size, per_frame_values)
