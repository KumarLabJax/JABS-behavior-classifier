import numpy as np
import typing
import scipy.stats

from jabs.pose_estimation import PoseEstimation
from jabs.feature_extraction.feature_base_class import Feature


class LixitDistanceInfo:
    """
    because we have two feature groups that both need to know which lixit is the
    closest, we compute that information once in this helper class and then
    pass it to both of the features
    The features cannot be merged into a single feature because one requires a
    a different set of window feature methods.
    """
    def __init__(self, poses: PoseEstimation, pixel_scale: float):

        # Identify closest lixit
        self._closest_key = PoseEstimation.KeypointIndex.NOSE
        # Distances include all keypoints
        self._keypoint_indices = list(PoseEstimation.KeypointIndex)

        self._poses = poses
        self._pixel_scale = pixel_scale
        self._closest_lixit_idx = {}
        self._cached_distances = {}
        self._cached_bearings = {}

    def cache_features(self, identity: int):
        if identity in self._cached_distances and identity in self._cached_bearings:
            return

        self._closest_lixit_idx[identity] = None
        self._cached_distances[identity] = {f'distance to lixit {keypoint.name}': np.full(self._poses.num_frames, np.nan, dtype=np.float32) for keypoint in self._keypoint_indices}
        self._cached_bearings[identity] = {'bearing to lixit': np.full(self._poses.num_frames, np.nan, dtype=np.float32)}

        if 'lixit' in self._poses.static_objects:
            closest_lixits = np.full(self._poses.num_frames, -1, dtype=np.int8)
            lixit = self._poses.static_objects['lixit']

            # there might be more than one: shape of lixit is <number of lixit> x 2
            num_lixit = lixit.shape[0]

            # convert lixit coordinates from pixels to cm if we can
            if self._pixel_scale is not None:
                lixit = lixit * self._pixel_scale

            alignment_distances = np.full((self._poses.num_frames, num_lixit), np.nan, dtype=np.float32)

            points, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

            # if there are multiple lixit, we compute the distance from nose to
            # each one, resulting in a numpy array of shape #frames, #lixit
            for i in range(num_lixit):
                pts = points[:, self._closest_key, :]
                ref = lixit[i]
                alignment_distances[:, i] = np.sqrt(np.sum((pts - ref) ** 2, axis=1))

            self._closest_lixit_idx[identity] = np.argmin(alignment_distances, axis=1)
            closest_lixit_vec = lixit[self._closest_lixit_idx[identity]]

            for keypoint in self._keypoint_indices:
                pts = points[:, keypoint, :]
                dists = pts - closest_lixit_vec
                kpt_dist_vector = np.hypot(dists[:, 0], dists[:, 1])
                self._cached_distances[identity][f'distance to lixit {keypoint.name}'] = kpt_dist_vector

            nose_points = points[:, PoseEstimation.KeypointIndex.NOSE, :]
            base_neck_points = points[:, PoseEstimation.KeypointIndex.BASE_NECK, :]
            self._cached_bearings[identity]['bearing to lixit'] = self.compute_angles(nose_points, base_neck_points, closest_lixit_vec)

    def get_distances(self, identity: int) -> typing.Dict:
        """
        get lixit distance feature for a given identity
        :param identity: integer identity to get distances foor
        :return: dict containing keyed distances
        """
        if identity not in self._cached_distances:
            self.cache_features(identity)
        return self._cached_distances[identity]

    def get_bearings(self, identity: int) -> typing.Dict:
        """
        get lixit bearing features for a given identity
        :param identity: integer identity to get bearings for
        :return: dict containing keyed bearings
        """
        if identity not in self._cached_bearings:
            self.cache_features(identity)
        return self._cached_bearings[identity]

    def get_closest_lixit(self, identity: int) -> np.ndarray:
        """
        get the closest lixit index
        :param identity: integer identity to get the closest lixit
        :return: np.ndarray of the lixit index
        """
        if identity not in self._closest_lixit_idx:
            self.cache_features(identity)
        return self._closest_lixit_idx[identity]

    @staticmethod
    def compute_angles(
            a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        """
        compute angles for a set of points
        :param a: array of point coordinates
        :param b: array of vertex point coordinates
        :param c: array of point coordinates
        :return: array containing angles, in degrees, formed from the lines
        ab and ba for each row in a, b, and c with range [-180, 180)
        """
        angles = np.degrees(
            np.arctan2(c[:, 1] - b[:, 1], c[:, 0] - b[:, 0]) -
            np.arctan2(a[:, 1] - b[:, 1], a[:, 0] - b[:, 0])
        )
        return ((angles + 180) % 360) - 180


class DistanceToLixit(Feature):
    _name = 'lixit_distances'
    _min_pose = 5
    _static_objects = ['lixit']
    
    def __init__(self, poses: PoseEstimation, pixel_scale: float,
                 distances: LixitDistanceInfo):
        super().__init__(poses, pixel_scale)

        self._cached_distances = distances

    def per_frame(self, identity: int) -> typing.Dict:
        """
        get the per frame distance to nearest lixit
        :param identity: identity to get feature values for
        :return: dict of numpy ndarray of values with shape (nframes,)
        """
        distances = self._cached_distances.get_distances(identity)
        return distances


class BearingToLixit(Feature):
    _name = 'lixit_bearings'
    _min_pose = 5
    _static_objects = ['lixit']

    # override for circular values
    _window_operations = {
        "mean": lambda x: scipy.stats.circmean(x, low=-180, high=180, nan_policy='omit'),
        "std_dev": lambda x: scipy.stats.circstd(x, low=-180, high=180, nan_policy='omit'),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float,
                 distances: LixitDistanceInfo):
        super().__init__(poses, pixel_scale)

        self._cached_distances = distances

    def per_frame(self, identity: int) -> typing.Dict:
        """
        get the per frame bearing to the nearest lixit values
        :param identity: identity to get the feature values for
        :return: dict of numpy ndarray values with shape (nframes,)
        """
        bearings = self._cached_distances.get_bearings(identity)
        return bearings

    def window(self, identity: int, window_size: int,
               per_frame_values: dict) -> typing.Dict:
        # override for circular values
        return self._window_circular(identity, window_size, per_frame_values)
        
