import typing

import cv2
import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from .moment_cache import MomentInfo

class EllipseFit(Feature):
    """Feature for the best fit ellipse from the segmentation contours.
    """

    _name = 'ellipse_fit'
    _feature_names = ['x', 'y', 'a', 'b', 'c', 'w', 'l', 'theta']

    def __init__(self, poses: PoseEstimation, pixel_scale: float,
                 moment_cache: 'MomentInfo'):
        super().__init__(poses, pixel_scale)
        self._moment_cache = moment_cache

    def per_frame(self, identity: int) -> np.ndarray:
        values = np.zeros((self._poses.num_frames, len(self._feature_names)), dtype=np.float32)

        x = self._feature_names.index('x')
        y = self._feature_names.index('y')
        a = self._feature_names.index('a')
        b = self._feature_names.index('b')
        c = self._feature_names.index('c')
        w = self._feature_names.index('w')
        ln = self._feature_names.index('l')
        t = self._feature_names.index('theta')
        for frame in range(values.shape[0]):
            # Safety for division by 0 (no segmentation to calculate on)
            if self._moment_cache.get_moment(frame, 'm00')==0:
                continue
            values[frame, x] = self._moment_cache.get_moment(frame, 'm10') / self._moment_cache.get_moment(frame, 'm00')
            values[frame, y] = self._moment_cache.get_moment(frame, 'm01') / self._moment_cache.get_moment(frame, 'm00')
            values[frame, a] = self._moment_cache.get_moment(frame, 'm20') / self._moment_cache.get_moment(frame, 'm00') - np.square(values[frame, x])
            values[frame, b] = 2*(self._moment_cache.get_moment(frame, 'm11') / self._moment_cache.get_moment(frame, 'm00') - values[frame, x] * values[frame, y])
            values[frame, c] = self._moment_cache.get_moment(frame, 'm02') / self._moment_cache.get_moment(frame, 'm00') - np.square(values[frame, y])
            values[frame, w] = 0.5 * np.sqrt(
                8*(values[frame, a] + values[frame, c] -
                    np.sqrt(np.square(values[frame, b]) +
                            np.square(values[frame, a] - values[frame, c])))
                )
            values[frame, ln] = 0.5 * np.sqrt(
                8*(values[frame, a] + values[frame, c] +
                    np.sqrt(np.square(values[frame, b]) +
                            np.square(values[frame, a] - values[frame, c])))
                )
            values[frame, t] = 0.5 * np.arctan(
                2 * values[frame, b] / (values[frame, a] - values[frame, c])
                )

        return values
