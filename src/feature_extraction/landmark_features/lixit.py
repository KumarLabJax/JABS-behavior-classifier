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

        # there might be more than one: shape of lixit is <number of lixit> x 2
        num_lixit = lixit.shape[0]

        # convert lixit coordinates from pixels to cm if we can
        if self._pixel_scale is not None:
            lixit = lixit * self._pixel_scale

        distances = np.zeros((self._poses.num_frames, num_lixit),
                             dtype=np.float32)

        points, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

        # if there are multiple lixit, we compute the distance from nose to
        # each one, resulting in a numpy array of shape #frames, #lixit
        for i in range(num_lixit):
            pts = points[:, self._nose_index, :]
            ref = lixit[i]
            distances[:, i] = np.sqrt(np.sum((pts - ref) ** 2, axis=1))

        # return the min of each row, to give us a numpy array with a shape
        # (#nframes,) containing the distance from the nose to the closest lixit
        # for each frame
        return distances.min(axis=1)
