import abc
import typing
from abc import ABC

import numpy as np

from src.utils.utilities import rolling_window
from src.pose_estimation import PoseEstimation


class FeatureSet(ABC):

    def __init__(self, poses: PoseEstimation, pixel_scale: float = 1.0):
        super().__init__()
        self._poses = poses
        self._pixel_scale = pixel_scale

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def column_names(cls) -> [str]:
        pass

    @abc.abstractmethod
    def compute_per_frame(self, identity: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def compute_window(self, identity: int, window_size: int,
                       per_frame_values: np.ndarray) -> dict:
        pass

    @staticmethod
    def window_width(window_size: int) -> int:
        return 2 * window_size + 1

    def _window_masks(self, frame_mask: np.ndarray, window_size: int) -> np.ndarray:

        window_width = self.window_width(window_size)

        # generate a numpy mask array to mask out invalid frames
        mask = np.full(self._poses.num_frames, 1)
        mask[frame_mask == 1] = 0

        # generate masks for all of the rolling windows
        return rolling_window(
            np.pad(mask, window_size, 'constant', constant_values=1),
            window_width
        )

    def _compute_window_feature(self, feature_values: np.ndarray,
                                frame_mask: np.ndarray, window_size: int,
                                op: typing.Callable) -> np.ndarray:

        window_masks = self._window_masks(frame_mask, window_size)

        window_width = self.window_width(window_size)
        values = np.zeros_like(feature_values)
        if feature_values.ndim == 1:
            windows = rolling_window(
                np.pad(feature_values, window_size),
                window_width
            )
            mx = np.ma.masked_array(windows, window_masks)
            values[:] = op(mx, axis=1)
        else:
            # if the feature is 2D, for example 'pairwise_distances',
            # compute the window features for each column
            for j in range(feature_values.shape[1]):
                windows = rolling_window(
                    np.pad(feature_values[:, j], window_size),
                    window_width
                )
                mx = np.ma.masked_array(windows, window_masks)
                values[:, j] = op(mx, axis=1)

        return values

    def _compute_window_features_circular(self, feature_values: np.ndarray,
                                frame_mask: np.ndarray, window_size: int,
                                operations: {}) -> dict:
        """
        special case compute_window_features for circular measurements

        :param feature_values: numpy array containing per-frame feature values
        :param frame_mask: numpy array that indicates if the frame is valid or
        not for the specific identity we are computing features for
        :param window_size:
        :param operations:
        :return: dictionary with the following form:
        {
          'operation_name': np.ndarray,
          'operation_name2': np.ndarray,
        }
        """
        nframes = self._poses.num_frames
        values = {}

        for op in operations:
            values[op] = np.zeros_like(feature_values)

        def func_wrapper(_op, _values):
            # XXX
            # scipy.stats.circstd has a bug that can result in nan
            # and a warning message to stderr if passed an array of
            # nearly identical values
            # this will be fixed when 1.6.0 is released, so this
            # work-around can be removed once we can upgrade to
            # scipy 1.6.0
            # our work around is to suppress the warning and replace
            # the nan with 0
            with np.errstate(invalid='ignore'):
                v = _op(_values)
            if np.isnan(v):
                return 0.0
            return v

        # unfortunately the scipy.stats.circmean/circstd functions don't work
        # with numpy masked arrays, so we need to iterate over each window and
        # create a view with only the valid values

        for i in range(nframes):

            # identity doesn't exist for this frame don't bother to compute
            if not frame_mask[i]:
                continue

            slice_start = max(0, i - window_size)
            slice_end = min(i + window_size + 1, nframes)

            slice_frames_valid = frame_mask[slice_start:slice_end]

            if feature_values.ndim == 1:
                window_values = feature_values[slice_start:slice_end][
                    slice_frames_valid == 1]

                for op_name, op in operations.items():
                    if op_name == 'std_dev':
                        values[op_name][i] = func_wrapper(op, window_values)
                    else:
                        values[op_name][i] = op(window_values)
            else:
                for j in range(feature_values.shape[1]):
                    window_values = feature_values[slice_start:slice_end, j][slice_frames_valid == 1]

                    for op_name, op in operations.items():
                        if op_name == 'std_dev':
                            values[op_name][i, j] = func_wrapper(op, window_values)
                        else:
                            values[op_name][i, j] = op(window_values)

        return values
