import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


class DistanceToLixit(Feature):
    _name = 'distance_to_lixit'
    _min_pose = 5
    _static_objects = ['lixit']
    # Identify closest lixit
    _closest_key = PoseEstimation.KeypointIndex.NOSE
    # Distances include all keypoints
    _keypoint_indices = list(PoseEstimation.KeypointIndex)

    def per_frame(self, identity: int) -> dict:
        """
        get the per frame distance to nearest lixit
        :param identity: identity to get feature values for
        :return: dict of numpy ndarray of values with shape (nframes,)
        """

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

        closest_lixit = np.argmin(alignment_distances, axis=1)
        closest_lixit_vec = lixit[closest_lixit]

        return_dict = {}
        for keypoint in self._keypoint_indices:
            pts = points[:, keypoint, :]
            dists = pts - closest_lixit_vec
            kpt_dist_vector = np.hypot(dists[:, 0], dists[:, 1])
            return_dict[f'distance to lixit {keypoint.name}'] = kpt_dist_vector

        return return_dict
