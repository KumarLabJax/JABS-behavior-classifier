import abc
import typing
from abc import ABC

import numpy as np

from src.utils.utilities import rolling_window
from src.pose_estimation import PoseEstimation


class FeatureGroup(ABC):
    """
    Abstract Base Class to define a common interface for classes that implement
    one or more related features
    """

    def __init__(self, poses: PoseEstimation, pixel_scale: float = 1.0):
        super().__init__()
        self._poses = poses
        self._pixel_scale = pixel_scale

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """ return a string name of the feature group """
        pass

    @classmethod
    @abc.abstractmethod
    def feature_names(cls) -> dict:
        """
        return a dict of feature names, where each key in the dictionary is a
        feature set name (for example 'pairwise_distances') and the value is
        a list of strings containing the names of the features for that
        feature set (for example, for pairwise distances
        ['LEFT_EAR-LEFT_FRONT_PAW', ...])
        """
        pass

    @abc.abstractmethod
    def compute_per_frame(self, identity: int) -> np.ndarray:
        """
        each FeatureGroup subclass will implement this to compute the
        features in the group

        returns a dictionary where each key is a feature set in the group. A
        feature set could be a single feature, where it would be a 1D numpy
        ndarray, or it could be a set of related features (for example the
        pairwise point distances, which is a 2D ndarray where each row
        corresponds to the frame index, and each column is one of the point
        distances)
        """
        pass

    @abc.abstractmethod
    def compute_window(self, identity: int, window_size: int,
                       per_frame_values: np.ndarray) -> dict:
        pass

    @staticmethod
    def window_width(window_size: int) -> int:
        return 2 * window_size + 1

    def _window_masks(self, frame_mask: np.ndarray, window_size: int) -> np.ndarray:
        """
        helper function for generating masks for all of the windows to be used
        to compute window feature values
        """

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
        """
        helper function to compute window feature values

        :param feature_values: per frame feature values. Can be a 1D ndarray
        for a single feature, or a 2D array for a set of related features
        (e.g. pairwise point distances are stored as a 2D array)
        :param frame_mask: array indicating which frames are valid for the
        current identity
        :param window_size: number of frames (in each direction) to include
        in the window. The actual number of frames is 2 * window_size + 1
        :param op: function to perform the actual computation
        :return: numpy nd array containing feature values
        """
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

    def _compute_window_features_circular(
            self, feature_values: np.ndarray, frame_mask: np.ndarray,
            window_size: int, op: typing.Callable,
            scipy_workaround: bool = False
    ) -> dict:
        """
        special case compute_window_features for circular measurements

        :param feature_values: numpy array containing per-frame feature values
        :param frame_mask: numpy array that indicates if the frame is valid or
        not for the specific identity we are computing features for
        :param window_size:
        :param op:
        :param scipy_workaround:

        # scipy.stats.circstd has a bug that can result in nan
        # and a warning message to stderr if passed an array of
        # nearly identical values
        #
        # our work around is to suppress the warning and replace
        # the nan with 0
        #
        # this will be fixed as of scipy 1.6.0, so this work-around can be
        # removed once we can upgrade to scipy 1.6.0

        :return: numpy nd array with circular feature values
        """
        nframes = self._poses.num_frames
        values = np.zeros_like(feature_values)

        def func_wrapper(_values):
            """
            implements work-around described in docstring
            :param _values: values to use for computing window feature value
            for a single frame
            :return: window feature value
            """
            with np.errstate(invalid='ignore'):
                v = op(_values)
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

                if scipy_workaround:
                    values[i] = func_wrapper(window_values)
                else:
                    values[i] = op(window_values)
            else:
                for j in range(feature_values.shape[1]):
                    window_values = feature_values[slice_start:slice_end, j][slice_frames_valid == 1]

                    if scipy_workaround:
                        values[i, j] = func_wrapper(window_values)
                    else:
                        values[i, j] = op(window_values)

        return values
