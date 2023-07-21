import typing

import numpy as np


from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature
from src.utils.utilities import n_choose_r


def _init_feature_names() -> typing.List[str]:
    """
    "distance_name_1-distance_name_2"
    """
    distances = []
    point_names = [p.name for p in PoseEstimation.KeypointIndex]
    for i in range(0, len(point_names)):
        p1 = point_names[i]
        for p2 in point_names[i + 1:]:
            distances.append(f"{p1}-{p2}")
    return distances


class PairwisePointDistances(Feature):

    _name = 'pairwise_distances'
    _feature_names = _init_feature_names()

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)
        self._num_distances = n_choose_r(len(PoseEstimation.KeypointIndex), 2)

    def per_frame(self, identity: int) -> np.ndarray:
        """
                compute the value of the per frame features for a specific identity
                :param identity: identity to compute features for
                :return: np.ndarray with feature values
                """
        values = np.zeros(
            (self._poses.num_frames, self._num_distances), dtype=np.float32)

        points, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

        npoints = len(PoseEstimation.KeypointIndex)
        d = 0
        for i in range(0, npoints):
            for j in range(i+1, npoints):
                # compute euclidean distance between ith and jth points
                values[:, d] = np.sqrt(
                    np.square(points[:, i, 0] - points[:, j, 0]) +
                    np.square(points[:, i, 1] - points[:, j, 1])
                )
                d += 1

        return values
