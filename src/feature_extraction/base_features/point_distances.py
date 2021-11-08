import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_group import FeatureGroup
from src.utils.utilities import n_choose_r


class PairwisePointDistances(FeatureGroup):

    _window_operations = {
        "mean": np.ma.mean,
        "median": np.ma.median,
        "std_dev": np.ma.std,
        "max": np.ma.amax,
        "min": np.ma.amin
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float = 1.0):
        super().__init__(poses, pixel_scale)
        self._num_distances = n_choose_r(len(PoseEstimation.KeypointIndex), 2)

    @property
    def name(self) -> str:
        return 'pairwise_distances'

    @classmethod
    def feature_names(cls) -> dict:
        """
        "distance_name_1-distance_name_2"
        """
        distances = []
        point_names = [p.name for p in PoseEstimation.KeypointIndex]
        for i in range(0, len(point_names)):
            p1 = point_names[i]
            for p2 in point_names[i + 1:]:
                distances.append(f"{p1}-{p2}")

        return {
            'pairwise_distances': distances
        }

    def compute_per_frame(self, identity: int) -> np.ndarray:
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

        return {
            self.name: values
        }

    def compute_window(self, identity: int, window_size: int,
                       per_frame_values: np.ndarray) -> dict:

        values = {}
        for op in self._window_operations:
            values[op] = self._compute_window_feature(
                per_frame_values, self._poses.identity_mask(identity),
                window_size, self._window_operations[op]
            )

        return {
            self.name: values
        }
