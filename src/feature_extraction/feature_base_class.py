import abc
import typing

import numpy as np

from src.utils.utilities import rolling_window
from src.pose_estimation import PoseEstimation


class Feature(abc.ABC):
    """
    Abstract Base Class to define a common interface for classes that implement
    one or more related features
    """

    # each subclass needs to define this name and feature_names
    _name = None
    # list of feature names, correspond to columns of feature values
    _feature_names = None

    # requirements for this feature to be available
    _min_pose = 2
    _static_objects = []

    _SMOOTHING_WINDOW = 5

    # _compute_window_feature uses numpy masked arrays, so we
    # need to use the np.ma.* versions of these functions
    # NOTE: Circular values need to override this as well as the window()
    _window_operations = {
        "mean": np.ma.mean,
        "median": np.ma.median,
        "std_dev": np.ma.std,
        "max": np.ma.amax,
        "min": np.ma.amin
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__()
        self._poses = poses
        self._pixel_scale = pixel_scale
        if self._name is None:
            raise NotImplementedError(
                "Base class must override _name class member")
        if self._feature_names is None:
            raise NotImplementedError(
                "Base class must override _feature_names class member")

    @classmethod
    def name(cls) -> str:
        """ return a string name of the feature """
        return cls._name

    @classmethod
    def feature_names(cls) -> typing.List[str]:
        """
        return a list of strings containing the names of the features for the
        feature set
        """
        return cls._feature_names

    @classmethod
    def is_supported(
            cls, pose_version: int, static_objects: typing.List[str]) -> bool:
        """
        check that a feature is supported by a pose file
        :param pose_version: pose file version
        :param static_objects: list of static object available in pose file
        :return: True if the pose file supports the feature, false otherwise
        """

        # check that the minimum pose version is met
        if cls._min_pose > pose_version:
            return False

        # check that any static objects required by the feature are
        # available
        for obj in cls._static_objects:
            if obj not in static_objects:
                return False

        return True

    @abc.abstractmethod
    def per_frame(self, identity: int) -> np.ndarray:
        """
        each FeatureSet subclass will implement this to compute the
        features in the set

        returns an ndarray containing the feature values.
        The feature set could be a single feature, where this would be a 1D
        numpy ndarray, or it could be a 2D ndarray for a set of related
        features (for example the pairwise point distances, which is a 2D
        ndarray where each row corresponds to the frame index, and each column
        is one of the pairwise point distances)
        """
        pass

    def window(self, identity: int, window_size: int,
               per_frame_values: np.ndarray) -> typing.Dict:
        """
        standard method for computing window feature values

        NOTE: some features may need to override this (for example, those with
        circular values such as angles)
        """
        values = {}
        for op in self._window_operations:
            values[op] = self._compute_window_feature(
                per_frame_values, self._poses.identity_mask(identity),
                window_size, self._window_operations[op]
            )
        return values

    def _window_circular(self, identity: int, window_size: int,
                         per_frame_values: np.ndarray) -> typing.Dict:

        values = {}
        for op_name, op in self._window_operations.items():
            values[op_name] = self._compute_window_features_circular(
                per_frame_values, self._poses.identity_mask(identity),
                window_size, op, op_name == 'std_dev')
        return values

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
    ) -> typing.Dict:
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
