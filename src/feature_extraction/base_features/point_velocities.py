import abc
import typing

import numpy as np
import scipy.stats

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


# TODO: merge this with point_speeds to reduce compute
# since they both use keypoint gradients

class PointVelocityDirs(Feature, abc.ABC):
    """ feature for the direction of the point velocity """

    # subclass must override this
    _name = 'point_velocity_dirs'
    _point_index = None

    # override for circular values
    _window_operations = {
        "mean": lambda x: scipy.stats.circmean(x, low=-180, high=180),
        "std_dev": lambda x: scipy.stats.circstd(x, low=-180, high=180),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        poses, point_masks = self._poses.get_identity_poses(identity, self._pixel_scale)

        bearings = self._poses.compute_all_bearings(identity)

        directions = {}

        for keypoint in PoseEstimation.KeypointIndex:
            # compute x,y velocities
            # pass indexes so numpy can figure out spacing
            points = np.ma.array(poses[:, keypoint, :], mask=np.stack([~point_masks[:, keypoint], ~point_masks[:, keypoint]]), dtype=np.float32)
            point_velocities = np.gradient(points, axis=0)

            # compute the orientation, and adjust based on the animal's bearing
            adjusted_angle = (((np.degrees(np.arctan2(point_velocities[:, 1], point_velocities[:, 0])) - bearings) + 360) % 360) - 180
            adjusted_angle.fill_value = 0
            directions[f"{keypoint.name} velocity direction"] = adjusted_angle.filled()

        return directions

    def window(self, identity: int, window_size: int,
               per_frame_values: np.ndarray) -> dict:
        # need to override to use special method for computing window features
        # with circular values
        return self._window_circular(identity, window_size, per_frame_values)
