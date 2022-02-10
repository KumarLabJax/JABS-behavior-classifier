import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


class DistanceToLixit(Feature):
    _name = 'distance_to_lixit'
    _feature_names = ['distance to lixit']
    _min_pose = 5
    _static_objects = ['lixit']
    _nose_index = PoseEstimation.KeypointIndex.NOSE

    def per_frame(self, identity: int) -> np.ndarray:
        """
        get the per frame distance to nearest lixit
        :param identity: identity to get feature values for
        :return: numpy ndarray of values with shape (nframes,)
        """

        lixit = self._poses.static_objects['lixit']
        if self._pixel_scale is not None:
            lixit = lixit * self._pixel_scale

        # first compute the distance to lixit
        # there might be more than one: shape of lixit is <number of lixit> x 2
        distances = np.zeros((self._poses.num_frames, lixit.shape[0]),
                             dtype=np.float32)

        num_lixit = lixit.shape[0]
        points, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

        for i in range(num_lixit):
            distances[:, i] = self._compute_distance(
                points[:, self._nose_index, :], lixit[i])

        return distances.min(axis=1)

    @staticmethod
    def _compute_distance(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        compute the distance between an array of points and a reference point
        :param points: numpy array of points with shape = (npoints, 2)
        :param reference: point (numpy array of length 2)
        :return: numpy array of distances
        """
        return np.sqrt(np.sum((points - reference) ** 2, axis=1))
