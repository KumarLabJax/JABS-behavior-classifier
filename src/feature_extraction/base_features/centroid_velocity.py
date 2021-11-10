import typing

import numpy as np
import scipy.stats

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


class CentroidVelocityMag(Feature):

    _name = 'centroid_velocity_mag'

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    @property
    def feature_names(self) -> typing.List[str]:
        return ['centroid_velocity_mag']

    def per_frame(self, identity: int) -> np.ndarray:
        values = np.zeros(self._poses.num_frames, dtype=np.float32)
        fps = self._poses.fps
        frame_valid = self._poses.get_identity_point_mask(identity)

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
            # compute x,y velocities, pass indexes so numpy can figure out spacing
            v = np.gradient(points, indexes, axis=0)

            # compute magnitude and direction of velocities
            values[indexes] = np.sqrt(
                np.square(v[:, 0]) + np.square(v[:, 1])) * fps

        return values


class CentroidVelocityDir(Feature):

    _name = 'centroid_velocity_dir'

    # override for circular values
    _window_operations_circular_2 = {
        "mean": lambda x: scipy.stats.circmean(x, low=-180, high=180),
        "std_dev": lambda x: scipy.stats.circstd(x, low=-180, high=180),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    @property
    def feature_names(self) -> typing.List[str]:
        return ['centroid_velocity_dir']

    def per_frame(self, identity: int) -> np.ndarray:
        values = np.zeros(self._poses.num_frames, dtype=np.float32)
        bearings = self._poses.compute_all_bearings(identity)
        frame_valid = self._poses.get_identity_point_mask(identity)

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
            # compute x,y velocities, pass indexes so numpy can figure out spacing
            v = np.gradient(points, indexes, axis=0)

            # compute direction of velocities
            d = np.degrees(np.arctan2(v[:, 1], v[:, 0]))

            # subtract animal bearing from orientation
            # convert angle to range -180 to 180
            values[indexes][indexes] = (((d - bearings[
                indexes]) + 360) % 360) - 180

        return values
