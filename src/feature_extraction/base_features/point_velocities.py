import abc
import typing

import numpy as np
import scipy.stats

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


# TODO: merge each of these pairs into a single feature with a 2D numpy array of values
# these are currently separate features in the features file, so we keep them
# separate here for ease of implementation, but this results in duplicated
# work computing each feature. Fix at next update to feature h5 file format.

class PointVelocityDir(Feature, abc.ABC):
    """ feature for the direction of the point velocity """

    # subclass must override this
    _point_index = None

    # override for circular values
    _window_operations = {
        "mean": lambda x: scipy.stats.circmean(x, low=-180, high=180),
        "std_dev": lambda x: scipy.stats.circstd(x, low=-180, high=180),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        points, mask = self._poses.get_identity_poses(identity, self._pixel_scale)

        bearings = self._poses.compute_all_bearings(identity)

        # get an array of the indexes where this point exists
        indexes = np.arange(self._poses.num_frames)[mask[:, self._point_index] == 1]

        # get points where this point exists
        points = points[indexes, self._point_index]

        values = np.zeros(self._poses.num_frames, dtype=np.float32)

        if indexes.shape[0] > 1:
            # compute x,y velocities
            # pass indexes so numpy can figure out spacing
            v = np.gradient(points, indexes, axis=0)

            # compute the orientation, and adjust based on the animal's bearing
            values[indexes] = (((np.degrees(np.arctan2(v[:, 1], v[:, 0])) - bearings[
                indexes]) + 360) % 360) - 180

        return values

    def window(self, identity: int, window_size: int,
               per_frame_values: np.ndarray) -> dict:
        # need to override to use special method for computing window features
        # with circular values
        return self._window_circular(identity, window_size, per_frame_values)


class PointVelocityMag(Feature, abc.ABC):
    """ feature for the magnitude of point velocity """

    # subclass must override this
    _point_index = None

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        """
        compute the value of the per frame features for a specific identity
        :param identity: identity to compute features for
        :return: np.ndarray with feature values
        """
        fps = self._poses.fps
        points, mask = self._poses.get_identity_poses(identity,
                                                      self._pixel_scale)

        # get an array of the indexes where this point exists
        indexes = np.arange(self._poses.num_frames)[mask[:, self._point_index] == 1]

        # get points where this point exists
        points = points[indexes, self._point_index]

        values = np.zeros(self._poses.num_frames, dtype=np.float32)

        if indexes.shape[0] > 1:
            # compute x,y velocities
            # pass indexes so numpy can figure out spacing
            v = np.gradient(points, indexes, axis=0)

            # compute magnitude of velocities
            values[indexes] = np.sqrt(
                np.square(v[:, 0]) + np.square(v[:, 1])) * fps

        return values


class NoseVelocityDir(PointVelocityDir):
    """ feature for the direction of the nose velocity """

    _name = 'nose_velocity_dir'
    _feature_names = ['nose velocity direction']
    _point_index = PoseEstimation.KeypointIndex.NOSE


class NoseVelocityMag(PointVelocityMag):
    """ feature for the magnitude of the nose velocity """

    _name = 'nose_velocity_mag'
    _feature_names = ['nose velocity magnitude']
    _point_index = PoseEstimation.KeypointIndex.NOSE

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)


class BaseTailVelocityDir(PointVelocityDir):
    """ feature for the direction of the base_tail velocity """

    _name = 'base_tail_velocity_dir'
    _feature_names = ['base tail velocity direction']
    _point_index = PoseEstimation.KeypointIndex.BASE_TAIL

    # override for circular values
    _window_operations = {
        "mean": lambda x: scipy.stats.circmean(x, low=-180, high=180),
        "std_dev": lambda x: scipy.stats.circstd(x, low=-180, high=180),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)


class BaseTailVelocityMag(PointVelocityMag):
    """ feature for the magnitude of the base_tail velocity """

    _name = 'base_tail_velocity_mag'
    _feature_names = ['base tail velocity magnitude']
    _point_index = PoseEstimation.KeypointIndex.BASE_TAIL

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)


class LeftFrontPawVelocityDir(PointVelocityDir):
    """ feature for the direction of the left front paw velocity """

    _name = 'left_front_paw_velocity_dir'
    _feature_names = ['left front paw velocity direction']
    _point_index = PoseEstimation.KeypointIndex.LEFT_FRONT_PAW

    # override for circular values
    _window_operations_circular_2 = {
        "mean": lambda x: scipy.stats.circmean(x, low=-180, high=180),
        "std_dev": lambda x: scipy.stats.circstd(x, low=-180, high=180),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)


class LeftFrontPawVelocityMag(PointVelocityMag):
    """ feature for the magnitude of the left front paw velocity """

    _name = 'left_front_paw_velocity_mag'
    _feature_names = ['left front paw velocity magnitude']
    _point_index = PoseEstimation.KeypointIndex.LEFT_FRONT_PAW

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)


class RightFrontPawVelocityDir(PointVelocityDir):
    """ feature for the direction of the right front paw velocity """

    _name = 'right_front_paw_velocity_dir'
    _feature_names = ['right front paw velocity direction']
    _point_index = PoseEstimation.KeypointIndex.RIGHT_FRONT_PAW

    # override for circular values
    _window_operations_circular_2 = {
        "mean": lambda x: scipy.stats.circmean(x, low=-180, high=180),
        "std_dev": lambda x: scipy.stats.circstd(x, low=-180, high=180),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)


class RightFrontPawVelocityMag(PointVelocityMag):
    """ feature for the magnitude of the right front paw velocity """

    _name = 'right_front_paw_velocity_mag'
    _feature_names = ['right front paw velocity magnitude']
    _point_index = PoseEstimation.KeypointIndex.RIGHT_FRONT_PAW

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)
