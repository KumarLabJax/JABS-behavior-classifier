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
    # TODO: we're discarding centroid angle and ellipse-fit angle (theta)
    # These need to be handled similar to other angle terms (circular statistics)
    _feature_names = ['centroid_speed', 'w', 'l']

    def __init__(self, poses: PoseEstimation, pixel_scale: float,
                 moment_cache: 'MomentInfo'):
        super().__init__(poses, pixel_scale)
        self._moment_cache = moment_cache

    def per_frame(self, identity: int) -> np.ndarray:
        x = np.zeros((self._poses.num_frames), dtype=np.float32)
        y = np.zeros((self._poses.num_frames), dtype=np.float32)
        values = np.zeros((self._poses.num_frames, len(self._feature_names)), dtype=np.float32)

        fps = self._poses.fps
        cs = self._feature_names.index('centroid_speed')
        w = self._feature_names.index('w')
        ln = self._feature_names.index('l')
        for frame in range(values.shape[0]):
            # Safety for division by 0 (no segmentation to calculate on)
            if self._moment_cache.get_moment(frame, 'm00')==0:
                continue
            x[frame] = self._moment_cache.get_moment(frame, 'm10') / self._moment_cache.get_moment(frame, 'm00')
            y[frame] = self._moment_cache.get_moment(frame, 'm01') / self._moment_cache.get_moment(frame, 'm00')
            a = self._moment_cache.get_moment(frame, 'm20') / self._moment_cache.get_moment(frame, 'm00') - np.square(x[frame])
            b = 2*(self._moment_cache.get_moment(frame, 'm11') / self._moment_cache.get_moment(frame, 'm00') - x[frame] * y[frame])
            c = self._moment_cache.get_moment(frame, 'm02') / self._moment_cache.get_moment(frame, 'm00') - np.square(y[frame])
            values[frame, w] = 0.5 * np.sqrt(8*(a + c - np.sqrt(np.square(b) + np.square(a - c))))
            values[frame, ln] = 0.5 * np.sqrt(8*(a + c + np.sqrt(np.square(b) + np.square(a - c))))
            # Theta needs the be handled uniquely because it's only 0-pi and needs circular statistics
            # theta = 0.5 * np.arctan(2 * b / (a - c))
        # Calculate the centroid speeds
        centroid_speeds = np.hypot(np.gradient(x), np.gradient(y))
        values[:,cs] = centroid_speeds

        return values
