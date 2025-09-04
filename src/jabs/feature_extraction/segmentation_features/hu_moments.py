import typing

import cv2
import numpy as np

from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation

if typing.TYPE_CHECKING:
    from .moment_cache import MomentInfo


class HuMoments(Feature):
    """Feature for the hu image moments of the segmentation contours."""

    _name = "hu_moments"
    _feature_names: typing.ClassVar[list[str]] = [f"hu{i}" for i in range(1, 8)]

    def __init__(self, poses: PoseEstimation, pixel_scale: float, moment_cache: "MomentInfo"):
        super().__init__(poses, pixel_scale)
        self._moment_cache = moment_cache

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """Computes per-frame Hu image moment features for a specific identity.

        For each frame, calculates the seven Hu invariant moments from the cached image moments
        of the segmentation contours.

        Args:
            identity (int): The identity index for which to compute Hu moments.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping Hu moment names ("hu1" to "hu7") to per-frame arrays of values.
        """
        values = {
            name: np.zeros([self._poses.num_frames], dtype=np.float32)
            for name in self._feature_names
        }

        for frame in range(self._poses.num_frames):
            # Skip calculation if m00 is 0
            if self._moment_cache.get_moment(frame, "m00") == 0:
                continue
            hu_moments = cv2.HuMoments(self._moment_cache.get_all_moments(frame))
            for i, name in enumerate(self._feature_names):
                values[name][frame] = hu_moments[i, 0]

        return values
