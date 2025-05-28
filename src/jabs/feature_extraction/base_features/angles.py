import typing

import numpy as np
import scipy.stats

from jabs.feature_extraction.angle_index import AngleIndex
from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation


class Angles(Feature):
    """this module computes joint angles the result is a dict of features of length #frames rows"""

    _name = "angles"

    # override for circular values
    _window_operations: typing.ClassVar[dict[str, typing.Callable]] = {
        "mean": lambda x: scipy.stats.circmean(x, high=360, nan_policy="omit"),
        "std_dev": lambda x: scipy.stats.circstd(x, high=360, nan_policy="omit"),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)
        self._num_angles = len(AngleIndex)

    def per_frame(self, identity: int) -> np.ndarray:
        """compute the value of the per frame features for a specific identity"""
        values = {}

        poses, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

        for named_angle in AngleIndex:
            angle_keypoints = AngleIndex.get_angle_indices(named_angle)
            values[f"angle {AngleIndex.get_angle_name(named_angle)}"] = (
                self._compute_angles(
                    poses[:, angle_keypoints[0]],
                    poses[:, angle_keypoints[1]],
                    poses[:, angle_keypoints[2]],
                )
            )
        return values

    def window(self, identity: int, window_size: int, per_frame_values: dict) -> dict:
        """compute window feature values.

        overrides the base class method to handle circular values.

        Args:
            identity (int): subject identity
            window_size (int): window size NOTE: (actual window size is 2 * window_size + 1)
            per_frame_values (dict): per frame values for this identity
        """
        return self._window_circular(identity, window_size, per_frame_values)

    @staticmethod
    def _compute_angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """compute angles for a set of points

        Args:
            a: array of point coordinates
            b: array of vertex point coordinates
            c: array of point coordinates

        Returns:
            array containing angles, in degrees, formed from the lines ab and ba for each row in a, b, and c with range [0, 360)
        """
        angles = np.degrees(
            np.arctan2(c[:, 1] - b[:, 1], c[:, 0] - b[:, 0])
            - np.arctan2(a[:, 1] - b[:, 1], a[:, 0] - b[:, 0])
        )
        return angles % 360
