import numpy as np
import scipy.stats

from src.feature_extraction.angle_index import AngleIndex
from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


class Angles(Feature):

    """
    this module computes joint angles
    the result is a 2D numpy array with #frames rows, and #angles columns
    (#angles different features for input to the classifier)
    """

    _name = 'angles'
    _feature_names = [f'angle {AngleIndex.get_angle_name(i.value)}' for i in AngleIndex]

    # override for circular values
    _window_operations = {
        "mean": lambda x: scipy.stats.circmean(x, high=360),
        "std_dev": lambda x: scipy.stats.circstd(x, high=360),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)
        self._num_angles = len(AngleIndex)

    def per_frame(self, identity: int) -> np.ndarray:
        nframes = self._poses.num_frames
        values = np.zeros((nframes, self._num_angles), dtype=np.float32)

        poses, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

        values[:, AngleIndex.NOSE_BASE_NECK_RIGHT_FRONT_PAW] = \
            self._compute_angles(
                poses[:, PoseEstimation.KeypointIndex.NOSE],
                poses[:, PoseEstimation.KeypointIndex.BASE_NECK],
                poses[:, PoseEstimation.KeypointIndex.RIGHT_FRONT_PAW]
            )

        values[:, AngleIndex.NOSE_BASE_NECK_LEFT_FRONT_PAW] = \
            self._compute_angles(
                poses[:, PoseEstimation.KeypointIndex.NOSE],
                poses[:, PoseEstimation.KeypointIndex.BASE_NECK],
                poses[:, PoseEstimation.KeypointIndex.LEFT_FRONT_PAW]
            )

        values[:, AngleIndex.RIGHT_FRONT_PAW_BASE_NECK_CENTER_SPINE] = \
            self._compute_angles(
                poses[:, PoseEstimation.KeypointIndex.RIGHT_FRONT_PAW],
                poses[:, PoseEstimation.KeypointIndex.BASE_NECK],
                poses[:, PoseEstimation.KeypointIndex.CENTER_SPINE]
            )

        values[:, AngleIndex.LEFT_FRONT_PAW_BASE_NECK_CENTER_SPINE] = \
            self._compute_angles(
                poses[:, PoseEstimation.KeypointIndex.LEFT_FRONT_PAW],
                poses[:, PoseEstimation.KeypointIndex.BASE_NECK],
                poses[:, PoseEstimation.KeypointIndex.CENTER_SPINE]
            )

        values[:, AngleIndex.BASE_NECK_CENTER_SPINE_BASE_TAIL] = \
            self._compute_angles(
                poses[:, PoseEstimation.KeypointIndex.BASE_NECK],
                poses[:, PoseEstimation.KeypointIndex.CENTER_SPINE],
                poses[:, PoseEstimation.KeypointIndex.BASE_TAIL]
            )

        values[:, AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_CENTER_SPINE] = \
            self._compute_angles(
                poses[:, PoseEstimation.KeypointIndex.RIGHT_REAR_PAW],
                poses[:, PoseEstimation.KeypointIndex.BASE_TAIL],
                poses[:, PoseEstimation.KeypointIndex.CENTER_SPINE]
            )

        values[:, AngleIndex.LEFT_REAR_PAW_BASE_TAIL_CENTER_SPINE] = \
            self._compute_angles(
                poses[:, PoseEstimation.KeypointIndex.LEFT_REAR_PAW],
                poses[:, PoseEstimation.KeypointIndex.BASE_TAIL],
                poses[:, PoseEstimation.KeypointIndex.CENTER_SPINE]
            )

        values[:, AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_MID_TAIL] = \
            self._compute_angles(
                poses[:, PoseEstimation.KeypointIndex.RIGHT_REAR_PAW],
                poses[:, PoseEstimation.KeypointIndex.BASE_TAIL],
                poses[:, PoseEstimation.KeypointIndex.MID_TAIL]
            )

        values[:, AngleIndex.LEFT_REAR_PAW_BASE_TAIL_MID_TAIL] = \
            self._compute_angles(
                poses[:, PoseEstimation.KeypointIndex.LEFT_REAR_PAW],
                poses[:, PoseEstimation.KeypointIndex.BASE_TAIL],
                poses[:, PoseEstimation.KeypointIndex.MID_TAIL]
            )

        values[:, AngleIndex.CENTER_SPINE_BASE_TAIL_MID_TAIL] = \
            self._compute_angles(
                poses[:, PoseEstimation.KeypointIndex.CENTER_SPINE],
                poses[:, PoseEstimation.KeypointIndex.BASE_TAIL],
                poses[:, PoseEstimation.KeypointIndex.MID_TAIL]
            )

        values[:, AngleIndex.BASE_TAIL_MID_TAIL_TIP_TAIL] = \
            self._compute_angles(
                poses[:, PoseEstimation.KeypointIndex.BASE_TAIL],
                poses[:, PoseEstimation.KeypointIndex.MID_TAIL],
                poses[:, PoseEstimation.KeypointIndex.TIP_TAIL]
            )

        return values

    def window(self, identity: int, window_size: int,
               per_frame_values: np.ndarray) -> dict:
        # need to override to use special method for computing window features
        # with circular values
        return self._window_circular(identity, window_size, per_frame_values)

    @staticmethod
    def _compute_angles(
            a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        """
        compute angles for a set of points
        :param a: array of point coordinates
        :param b: array of vertex point coordinates
        :param c: array of point coordinates
        :return: array containing angles, in degrees, formed from the lines
        ab and ba for each row in a, b, and c
        """
        angles = np.degrees(
            np.arctan2(c[:, 1] - b[:, 1], c[:, 0] - b[:, 0]) -
            np.arctan2(a[:, 1] - b[:, 1], a[:, 0] - b[:, 0])
        )
        return np.where(angles < 0, angles + 360, angles)
