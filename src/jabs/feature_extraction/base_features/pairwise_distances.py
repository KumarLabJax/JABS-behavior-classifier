import numpy as np

from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation


class PairwisePointDistances(Feature):
    """Feature extraction class for computing pairwise Euclidean distances between all keypoints per frame.

    This class calculates the Euclidean distance between every unique pair of keypoints, returning a dictionary
    mapping each keypoint pair to an array of per-frame distances.
    """

    _name = "pairwise_distances"

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """compute the value of the per frame features for a specific identity

        Args:
            identity: identity to compute features for

        Returns:
            dict with feature values
        """
        points, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

        values = {}
        point_names = [p.name for p in PoseEstimation.KeypointIndex]

        for i in range(0, len(point_names)):
            p1_name = point_names[i]
            for j in range(i + 1, len(point_names)):
                p2_name = point_names[j]
                # compute euclidean distance between ith and jth points
                euclidean_dist = np.sqrt(
                    np.square(points[:, i, 0] - points[:, j, 0])
                    + np.square(points[:, i, 1] - points[:, j, 1])
                )
                values[f"{p1_name}-{p2_name}"] = euclidean_dist

        return values
