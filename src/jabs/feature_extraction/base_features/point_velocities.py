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

    def window(self, identity: int, window_size: int, per_frame_values: dict) -> dict:
        """compute window feature values.

        Args:
            identity (int): subject identity
            window_size (int): window size NOTE: (actual window size is 2 *
                window_size + 1)
            per_frame_values (dict[str, np.ndarray]): dictionary of per frame values for this identity

        need to override to use special method for computing window features with circular values
        """
        # separate circular and non-circular values
        non_circular = {k: v for k, v in per_frame_values.items() if "sine" in k or "cosine" in k}
        circular = {k: v for k, v in per_frame_values.items() if k not in non_circular}

        circular_features = self._window_circular(identity, window_size, circular)
        non_circular_features = super().window(identity, window_size, non_circular)

        return circular_features | non_circular_features
