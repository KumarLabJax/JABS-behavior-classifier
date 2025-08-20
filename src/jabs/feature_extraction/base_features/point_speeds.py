import numpy as np

from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation


class PointSpeeds(Feature):
    """Feature extraction class for computing the speed of each keypoint per frame.

    This class calculates the instantaneous speed of each keypoint by computing the Euclidean norm of the
    frame-to-frame displacement, scaled by the video frame rate. The resulting speeds are provided as a dictionary
    mapping keypoint names to per-frame speed arrays.
    """

    _name = "point_speeds"

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """compute the value of the per frame features for a specific identity

        Args:
            identity: identity to compute features for

        Returns:
            dict with feature values
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
