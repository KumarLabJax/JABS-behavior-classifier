import typing

import numpy as np
import scipy.stats

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


# TODO: merge CentroidVelocityMag and CentroidVelocityDir into a single feature
#  with a 2D numpy array of values
# these are currently separate features in the features file, so we keep them
# separate here for ease of implementation, but this results in duplicated
# work computing each feature. Fix at next update to feature h5 file format.

class CentroidVelocityDir(Feature):
    """ feature for the direction of the center of mass velocity """

    _name = 'centroid_velocity_dir'

    # override for circular values
    _window_operations = {
        "mean": lambda x: scipy.stats.circmean(x, low=-180, high=180),
        "std_dev": lambda x: scipy.stats.circstd(x, low=-180, high=180),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        values = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        bearings = self._poses.compute_all_bearings(identity)
        frame_valid = self._poses.identity_mask(identity)

        # compute the velocity of the center of mass.
        # first, grab convex hulls for this identity
        convex_hulls = self._poses.get_identity_convex_hulls(identity)

        # get an array of the indexes of valid frames only
        indexes = np.arange(self._poses.num_frames)[frame_valid == 1]

        # get centroids for all frames where this identity is present
        centroids = [convex_hulls[i].centroid for i in indexes]

        # convert to numpy array of x,y points of the centroids
        points = np.asarray([[p.x, p.y] for p in centroids])

        if points.shape[0] > 1:
            # compute x,y velocities
            # pass indexes so numpy can figure out spacing
            v = np.gradient(points, indexes, axis=0)

            # compute direction of velocities
            d = np.degrees(np.arctan2(v[:, 1], v[:, 0]))

            # subtract animal bearing from orientation
            # convert angle to range -180 to 180
            values[indexes] = (((d - bearings[indexes]) + 360) % 360) - 180

        return {'centroid_velocity_dir': values}

    def window(self, identity: int, window_size: int,
               per_frame_values: np.ndarray) -> dict:
        # need to override to use special method for computing window features
        # with circular values
        return self._window_circular(identity, window_size, per_frame_values)


class CentroidVelocityMag(Feature):
    """ feature for the magnitude of the center of mass velocity """

    _name = 'centroid_velocity_mag'

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        """
        compute the value of the per frame features for a specific identity
        :param identity: identity to compute features for
        :return: np.ndarray with feature values
        """
        values = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        fps = self._poses.fps
        frame_valid = self._poses.identity_mask(identity)

        # compute the velocity of the center of mass.
        # first, grab convex hulls for this identity
        convex_hulls = self._poses.get_identity_convex_hulls(identity)

        # get an array of the indexes of valid frames only
        indexes = np.arange(self._poses.num_frames)[frame_valid == 1]

        # get centroids for all frames where this identity is present
        centroids = [convex_hulls[i].centroid for i in indexes]

        # convert to numpy array of x,y points of the centroids
        points = np.asarray([[p.x, p.y] for p in centroids])

        if points.shape[0] > 1:
            # compute x,y velocities
            # pass indexes so numpy can figure out spacing
            v = np.gradient(points, indexes, axis=0)

            # compute magnitude of velocities
            values[indexes] = np.sqrt(
                np.square(v[:, 0]) + np.square(v[:, 1])) * fps

        return {'centroid_velocity_mag': values}
