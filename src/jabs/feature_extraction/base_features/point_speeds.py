import numpy as np

from jabs.pose_estimation import PoseEstimation
from jabs.feature_extraction.feature_base_class import Feature


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
        xy_deltas = np.gradient(poses, axis=0)
        point_velocities = np.linalg.norm(xy_deltas, axis=-1) * fps

        for keypoint in PoseEstimation.KeypointIndex:
            speeds[f"{keypoint.name} speed"] = point_velocities[:, keypoint.value]

        return speeds
