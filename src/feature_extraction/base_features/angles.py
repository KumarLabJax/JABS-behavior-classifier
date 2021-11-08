import numpy as np
import scipy.stats

from src.feature_extraction.angle_index import AngleIndex
from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_group import FeatureGroup


class Angles(FeatureGroup):

    _window_operations = {
        "mean": lambda x: scipy.stats.circmean(x, high=360),
        "std_dev": lambda x: scipy.stats.circstd(x, high=360),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float = 1.0):
        super().__init__(poses, pixel_scale)
        self._num_angles = len(AngleIndex)

    @property
    def name(self) -> str:
        return 'angles'

    @classmethod
    def feature_names(cls) -> dict:
        """
        return angle names, where each name is formed from the
        three points used to compute the angle, where the middle point name is
        the vertex point. For example, given points a,b, and c the angle between
        ab anb bc would be named 'point-name-a_point-name-b_point-name-c'
        """
        return {
            'angles': [AngleIndex.get_angle_name(i.value) for i in AngleIndex]
        }

    def compute_per_frame(self, identity: int) -> np.ndarray:
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

        return {
            self.name: values
        }

    def compute_window(self, identity: int, window_size: int,
                       per_frame_values: np.ndarray) -> dict:

        values = {}

        for op_name, op in self._window_operations.items():
            values[op_name] = self._compute_window_features_circular(
                per_frame_values, self._poses.identity_mask(identity),
                window_size, op, op_name == 'std_dev')

        return {
            self.name: values
        }

    @staticmethod
    def _compute_angles(
            a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        angles = np.degrees(
            np.arctan2(c[:, 1] - b[:, 1], c[:, 0] - b[:, 0]) -
            np.arctan2(a[:, 1] - b[:, 1], a[:, 0] - b[:, 0])
        )
        return np.where(angles < 0, angles + 360, angles)
