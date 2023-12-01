import typing

import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


class PointSpeeds(Feature):

    _name = 'point_speeds'

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        """
        compute the value of the per frame features for a specific identity
        :param identity: identity to compute features for
        :return: dict with feature values
        """
        fps = self._poses.fps
        poses, point_masks = self._poses.get_identity_poses(identity, self._pixel_scale)

        speeds = {}

        # calculate velocities for each point
        for keypoint in PoseEstimation.KeypointIndex:
            # grab all of the values for this point
            points = np.ma.array(poses[:, keypoint, :], mask=np.stack([~point_masks[:, keypoint], ~point_masks[:, keypoint]]), dtype=np.float32)
            point_velocities = np.gradient(points, axis=0)
            point_velocities.fill_value = 0
            speeds[f"{keypoint.name} speed"] = point_velocities

        # convert the velocities to speed and convert units
        for key, val in speeds.items():
            speeds[key] = np.linalg.norm(val, axis=-1) * fps

        return speeds
