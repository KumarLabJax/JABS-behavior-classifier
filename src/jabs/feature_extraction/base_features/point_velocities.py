import numpy as np

from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation

# TODO: merge this with point_speeds to reduce compute
#  since they both use keypoint gradients


class PointVelocityDirs(Feature):
    """feature for the direction of the point velocity"""

    # subclass must override this
    _name = "point_velocity_dirs"
    _point_index = None
    _use_circular = True

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """compute per-frame feature values

        Args:
            identity (int): subject identity

        Returns:
            dict[str, np.ndarray]: dictionary of per frame values for this identity
        """
        poses, point_masks = self._poses.get_identity_poses(identity, self._pixel_scale)

        bearings = self._poses.compute_all_bearings(identity)

        features = {}
        xy_deltas = np.gradient(poses, axis=0)
        angles = np.degrees(np.arctan2(xy_deltas[:, :, 1], xy_deltas[:, :, 0]))

        for keypoint in PoseEstimation.KeypointIndex:
            features[f"{keypoint.name} velocity direction"] = (
                (angles[:, keypoint.value] - bearings + 360) % 360
            ) - 180

            features[f"{keypoint.name} velocity direction sine"] = np.sin(
                np.deg2rad(features[f"{keypoint.name} velocity direction"])
            )
            features[f"{keypoint.name} velocity direction cosine"] = np.cos(
                np.deg2rad(features[f"{keypoint.name} velocity direction"])
            )

        return features
