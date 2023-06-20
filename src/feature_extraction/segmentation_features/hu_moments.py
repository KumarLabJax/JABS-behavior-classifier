import typing

import cv2
import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from .moment_cache import MomentInfo


class HuMoments(Feature):
    """Feature for the hu image moments of the segmentation contours.
    """

    _name = 'hu_moments'
    _feature_names = [f"hu{i}" for i in range(1, 8)]

    def __init__(self, poses: PoseEstimation, pixel_scale: float,
                 moment_cache: 'MomentInfo'):
        super().__init__(poses, pixel_scale)
        self._moment_cache = moment_cache

    def per_frame(self, identity: int) -> np.ndarray:
        values = np.zeros((self._poses.num_frames, len(self._feature_names)), dtype=np.float32)
        
        for frame in range(values.shape[0]):

            values[frame, :] = cv2.HuMoments(self._moment_cache.get_all_moments(frame)).T

        return values
