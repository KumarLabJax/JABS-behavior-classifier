import typing

import cv2
import numpy as np

from src.jabs.pose_estimation import PoseEstimation
from src.jabs.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from .moment_cache import MomentInfo


class HuMoments(Feature):
    """
    Feature for the hu image moments of the segmentation contours.
    """

    _name = 'hu_moments'
    _feature_names = [f"hu{i}" for i in range(1, 8)]

    def __init__(self, poses: PoseEstimation, pixel_scale: float,
                 moment_cache: 'MomentInfo'):
        super().__init__(poses, pixel_scale)
        self._moment_cache = moment_cache

    def per_frame(self, identity: int) -> np.ndarray:
        values = {name: np.zeros([self._poses.num_frames], dtype=np.float32) for name in self._feature_names}
        
        for frame in range(self._poses.num_frames):
            # Skip calculation if m00 is 0
            if self._moment_cache.get_moment(frame, 'm00') == 0:
                continue
            hu_moments = cv2.HuMoments(self._moment_cache.get_all_moments(frame))
            for i, name in enumerate(self._feature_names):
                values[name][frame] = hu_moments[i]

        return values
