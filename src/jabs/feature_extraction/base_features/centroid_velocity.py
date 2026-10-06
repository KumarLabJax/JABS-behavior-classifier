import numpy as np

from jabs.core.utils.pose_util import identity_centroids
from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation

# TODO: consider merging CentroidVelocityMag and CentroidVelocityDir into a single feature
#   these are currently separate features in the features file, so we keep them
#   separate here for ease of implementation, but this results in duplicated
#   work computing each feature.


class CentroidVelocityDir(Feature):
    """feature for the direction of the center of mass velocity"""

    _name = "centroid_velocity_dir"
    _use_circular = True

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """compute the value of the per frame features for a specific identity"""
        bearings = self._poses.compute_all_bearings(identity)

        # compute the velocity of the center of mass
        centroid_centers = identity_centroids(self._poses, identity)
        v = np.gradient(centroid_centers, axis=0)

        # compute direction of velocities
        d = np.degrees(np.arctan2(v[:, 1], v[:, 0]))

        # subtract animal bearing from orientation
        # convert angle to range -180 to 180
        values = (((d - bearings) + 180) % 360) - 180

        return {
            "centroid_velocity_dir": values,
            "centroid_velocity_dir sine": np.sin(np.radians(values)),
            "centroid_velocity_dir cosine": np.cos(np.radians(values)),
        }


class CentroidVelocityMag(Feature):
    """feature for the magnitude of the center of mass velocity"""

    _name = "centroid_velocity_mag"

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """compute the value of the per frame features for a specific identity

        Args:
            identity: identity to compute features for

        Returns:
            np.ndarray with feature values
        """
        fps = self._poses.fps

        # compute the velocity of the center of mass
        centroid_centers = identity_centroids(self._poses, identity)

        # get change over frames
        v = np.gradient(centroid_centers, axis=0)
        values = np.linalg.norm(v, axis=-1) * fps * self._pixel_scale

        return {"centroid_velocity_mag": values}
