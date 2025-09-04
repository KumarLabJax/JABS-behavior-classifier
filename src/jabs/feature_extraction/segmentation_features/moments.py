import typing

import numpy as np

from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation

if typing.TYPE_CHECKING:
    from .moment_cache import MomentInfo


class Moments(Feature):
    """feature for the image moments of the contours."""

    _name = "moments"
    # These are all the opencv moments
    # _feature_names = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
    # However, we only want to look at egocentric (translational invariant)
    # mu (central or relative to centroid) moments and nu (normalized central) moments meet this translational invariance criteria
    # nu moments are also scale-invariant
    _moments_to_use: typing.ClassVar[list[str]] = [
        "m00",
        "mu20",
        "mu11",
        "mu02",
        "mu30",
        "mu21",
        "mu12",
        "mu03",
        "nu20",
        "nu11",
        "nu02",
        "nu30",
        "nu21",
        "nu12",
        "nu03",
    ]

    def __init__(self, poses: PoseEstimation, pixel_scale: float, moment_cache: "MomentInfo"):
        super().__init__(poses, pixel_scale)
        self._moment_cache = moment_cache

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """Computes per-frame image moment features for a specific identity.

        For each frame, extracts selected translational and scale-invariant image moments
        (central and normalized central moments) from the cached segmentation data.

        Args:
            identity (int): The identity index for which to compute moment features.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping moment names to per-frame arrays of values.
        """
        values = {}

        for cur_moment in self._moments_to_use:
            vector = np.zeros([self._poses.num_frames], dtype=np.float32)
            for frame in range(self._poses.num_frames):
                vector[frame] = self._moment_cache.get_moment(frame, cur_moment)
            values[cur_moment] = vector

        return values
