import typing

import numpy as np
from scipy import stats

from jabs.feature_extraction.angle_index import AngleIndex
from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation


class Angles(Feature):
    """this module computes joint angles the result is a dict of features of length #frames rows"""

    _name = "angles"
    _use_circular = True

    # need to override to set the correct range for circular operations
    _circular_window_operations: typing.ClassVar[dict[str, typing.Callable]] = {
        "mean": lambda x: stats.circmean(x, low=0, high=360, nan_policy="omit"),
        "std_dev": lambda x: stats.circstd(x, low=0, high=360, nan_policy="omit"),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)
        self._num_angles = len(AngleIndex)

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """compute the value of the per frame features for a specific identity"""
        values = {}

        poses, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

        for named_angle in AngleIndex:
            angle_keypoints = AngleIndex.get_angle_indices(named_angle)
            values[f"angle {AngleIndex.get_angle_name(named_angle)}"] = self._compute_angles(
                poses[:, angle_keypoints[0]],
                poses[:, angle_keypoints[1]],
                poses[:, angle_keypoints[2]],
            )
            values[f"angle {AngleIndex.get_angle_name(named_angle)} sine"] = np.sin(
                np.deg2rad(values[f"angle {AngleIndex.get_angle_name(named_angle)}"])
            )
            values[f"angle {AngleIndex.get_angle_name(named_angle)} cosine"] = np.cos(
                np.deg2rad(values[f"angle {AngleIndex.get_angle_name(named_angle)}"])
            )

        return values

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
