import abc

import numpy as np
import scipy.stats

from jabs.pose_estimation import PoseEstimation
from jabs.feature_extraction.feature_base_class import Feature


# TODO: merge this with point_speeds to reduce compute
# since they both use keypoint gradients

class PointVelocityDirs(Feature, abc.ABC):
    """ feature for the direction of the point velocity """

    # subclass must override this
    _name = 'point_velocity_dirs'
    _point_index = None

    # override for circular values
    _window_operations = {
        "mean": lambda x: scipy.stats.circmean(x, low=-180, high=180, nan_policy='omit'),
        "std_dev": lambda x: scipy.stats.circstd(x, low=-180, high=180, nan_policy='omit'),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        poses, point_masks = self._poses.get_identity_poses(identity, self._pixel_scale)

        bearings = self._poses.compute_all_bearings(identity)

        directions = {}
        xy_deltas = np.gradient(poses, axis=0)
        angles = np.degrees(np.arctan2(xy_deltas[:, :, 1], xy_deltas[:, :, 0]))

        for keypoint in PoseEstimation.KeypointIndex:
            directions[f"{keypoint.name} velocity direction"] = ((angles[:, keypoint.value] - bearings + 360) % 360) - 180

        return directions


    def window(self, identity: int, window_size: int,
               per_frame_values: np.ndarray) -> dict:
        # need to override to use special method for computing window features
        # with circular values
        return self._window_circular(identity, window_size, per_frame_values)
