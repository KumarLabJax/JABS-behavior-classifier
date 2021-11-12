import typing

import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature
from src.utils.utilities import smooth


class AngularVelocity(Feature):

    """ compute angular velocity of animal bearing """

    _name = 'angular_velocity'
    _feature_names = ['angular_velocity']

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        """
        compute the value of the per frame features for a specific identity
        :param identity: identity to compute features for
        :return: np.ndarray with feature values
        """
        fps = self._poses.fps

        bearings = self._poses.compute_all_bearings(identity)
        velocities = np.zeros_like(bearings)

        for i in range(len(bearings) - 1):

            angle1 = bearings[i]
            angle1 = angle1 % 360
            if angle1 < 0:
                angle1 += 360

            angle2 = bearings[i + 1]
            angle2 = angle2 % 360
            if angle2 < 0:
                angle2 += 360

            diff1 = angle2 - angle1
            abs_diff1 = abs(diff1)
            diff2 = (360 + angle2) - angle1
            abs_diff2 = abs(diff2)
            diff3 = angle2 - (360 + angle1)
            abs_diff3 = abs(diff3)

            if abs_diff1 <= abs_diff2 and abs_diff1 <= abs_diff3:
                velocities[i] = diff1
            elif abs_diff2 <= abs_diff3:
                velocities[i] = diff2
            else:
                velocities[i] = diff3
        velocities = velocities * fps

        return smooth(velocities, smoothing_window=self._SMOOTHING_WINDOW)
