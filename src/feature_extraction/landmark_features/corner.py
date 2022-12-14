import math
import typing

import numpy as np
import scipy.stats
from shapely.geometry import Point


from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


class CornerDistanceInfo:
    """
    because we have two features that both need to know which corner is the
    closest, we compute that information once in this helper class and then
    pass it to both of the features
    The features are not merged into a single feature because one (bearing to
    corner) needs to use the circular window feature methods, and the other
    does not, so it can't easily be implemented as one multi-column Feature
    class
    """
    def __init__(self, poses: PoseEstimation, pixel_scale: float):

        self._poses = poses
        self._pixel_scale = pixel_scale
        self._cached_distances = {}

    def get_distances(self, identity: int) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        get corner distances and bearings for a given identity
        :param identity: integer identity to get distances for
        :return: tuple containing distances to nearest corner and bearing to
        nearest corner
        """

        if identity in self._cached_distances:
            return self._cached_distances[identity]
        else:
            distances = np.zeros(self._poses.num_frames, dtype=np.float32)
            bearings = np.zeros(self._poses.num_frames, dtype=np.float32)
            self_convex_hulls = self._poses.get_identity_convex_hulls(identity)
            idx = PoseEstimation.KeypointIndex

            try:
                corners = self._poses.static_objects['corners']
            except KeyError:
                return distances, bearings

            for frame in range(self._poses.num_frames):

                # don't scale the point coordinates by the pixel_scale value,
                # since the corners are in pixel space. we'll adjust the
                # distance units later
                points, mask = self._poses.get_points(frame, identity)

                if points is None:
                    continue

                # find distance to closest corner
                self_shape = self_convex_hulls[frame]
                distance = float('inf')
                corner_coordinates = (0, 0)
                for i in range(4):
                    d = self_shape.distance(Point(corners[i, 0], corners[i, 1]))
                    if d < distance:
                        distance = d
                        corner_coordinates = (corners[i, 0], corners[i, 1])

                self_base_neck_point = points[idx.BASE_NECK, :]
                self_nose_point = points[idx.NOSE, :]

                bearing = self.compute_angle(
                    self_nose_point,
                    self_base_neck_point,
                    corner_coordinates)

                # make angle in range [180, -180)
                if bearing > 180:
                    bearing -= 360

                distances[frame] = distance * self._pixel_scale
                bearings[frame] = bearing
            self._cached_distances[identity] = distances, bearings
            return distances, bearings

    @staticmethod
    def compute_angle(a, b, c):
        """
        compute angle created by three connected points
        :param a: point
        :param b: vertex point
        :param c: point
        :return: angle between AB and BC
        """

        # most of the point types are unsigned short integers
        # cast to signed types to avoid underflow issues during subtraction
        angle = math.degrees(
            math.atan2(int(c[1]) - int(b[1]), int(c[0]) - int(b[0])) -
            math.atan2(int(a[1]) - int(b[1]), int(a[0]) - int(b[0]))
        )
        return angle + 360 if angle < 0 else angle


class DistanceToCorner(Feature):
    _name = 'distance_to_corner'
    _feature_names = ['distance to corner']
    _min_pose = 5
    _static_objects = ['corners']

    def __init__(self, poses: PoseEstimation, pixel_scale: float,
                 distances: CornerDistanceInfo):
        super().__init__(poses, pixel_scale)

        self._cached_distances = distances

    def per_frame(self, identity: int) -> np.ndarray:
        """
        get the per frame distance to the nearest corner values
        :param identity: identity to get feature values for
        :return: numpy ndarray of values with shape (nframes,)
        """

        distances, _ = self._cached_distances.get_distances(identity)
        return distances


class BearingToCorner(Feature):
    _name = 'bearing_to_corner'
    _feature_names = ['bearing to corner']
    _min_pose = 5
    _static_objects = ['corners']

    # override for circular values
    _window_operations = {
        "mean": lambda x: scipy.stats.circmean(x, low=-180, high=180),
        "std_dev": lambda x: scipy.stats.circstd(x, low=-180, high=180),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float,
                 distances: CornerDistanceInfo):
        super().__init__(poses, pixel_scale)

        self._cached_distances = distances

    def per_frame(self, identity: int) -> np.ndarray:
        """
        get the per frame bearing to the nearest corner values
        :param identity: identity to get feature values for
        :return: numpy ndarray of values with shape (nframes,)
        """

        _, bearings = self._cached_distances.get_distances(identity)
        return bearings

    def window(self, identity: int, window_size: int,
               per_frame_values: np.ndarray) -> typing.Dict:
        # need to override to use special method for computing window features
        # with circular values
        return self._window_circular(identity, window_size, per_frame_values)
