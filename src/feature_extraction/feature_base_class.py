import abc
import typing
import numpy as np
from scipy import signal

from src.feature_extraction.window_operations import window_stats
from src.feature_extraction.window_operations import signal_stats
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

    # need to use the np.ma.* versions of these functions
    # NOTE: Circular values need to override this as well as the window()
    _window_operations = {
        "mean": window_stats.mean,
        "median": window_stats.median,
        "std_dev": window_stats.std_dev,
        # "kurtosis": window_stats.kurtosis,
        # "skew": window_stats.skew,
        "max": window_stats.max,
        "min": window_stats.min,
    }
    _nan_fill_value = 0

    # signal processing operations
    _signal_operations = {
        "fft_band": signal_stats.psd_mean_band,
        "psd_sum": signal_stats.psd_sum,
        "psd_max": signal_stats.psd_max,
        "psd_min": signal_stats.psd_min,
        "psd_mean": signal_stats.psd_mean,
        "psd_std_dev": signal_stats.psd_std_dev,
        "psd_kurtosis": signal_stats.psd_kurtosis,
        "psd_skew": signal_stats.psd_skew,
        "psd_median": signal_stats.psd_median,
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__()
        self._poses = poses
        self._pixel_scale = pixel_scale
        self._fps = poses.fps
        self._signal_bands = [
            {'band_low': 0.1, 'band_high': 1.0},
            {'band_low': 1.0, 'band_high': 3.0},
            {'band_low': 3.0, 'band_high': 5.0},
            {'band_low': 5.0, 'band_high': 8.0},
            {'band_low': 8.0, 'band_high': 15.0},
        ]

        if self._name is None:
            raise NotImplementedError(
                "Base class must override _name class member")

    @staticmethod
    def window_width(window_size: int) -> int:
        return 2 * window_size + 1

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
               per_frame_values: dict) -> typing.Dict:
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

    def window_signal(
        self,
        identity: int,
        window_size: int,
        per_frame_values: dict
    ) -> typing.Dict:
        """
        The standard method for computing signal processing window features.

        :param identity: The identity of the mouse.
        :param window_size: The window size used for signal formation.
        :param per_frame_values: The values for a particular feature.
        :return: a dictionary of the signal processing features.
        """
        values = {}

        psd_data = {}
        # Obtain the PSD once
        for key, values in per_frame_values.items():
            freqs, ts, Zxx = signal.stft(values, fs=self._fps, nperseg=window_size * 2 + 1, noverlap=window_size * 2, window='hann', scaling='psd', detrend='linear')
            psd = np.abs(Zxx)
            psd_data[key] = psd

        # Summarize the signal features
        for op_name, op in self._signal_operations.items():
            if op_name == 'fft_band':
                for band in self._signal_bands:
                    values[f"{op_name}-{band['band_low']}Hz-{band['band_high']}Hz"] = self._compute_signal_features(freqs, psd, self._poses.identity_mask(identity), op, band)
            else:
                values[op_name] = self._compute_signal_features(freqs, psd, self._poses.identity_mask(identity), op)

        return values

    def _window_circular(self, identity: int, window_size: int,
                         per_frame_values: dict) -> typing.Dict:
        """
        helper function for overriding window features to be circular

        :param identity: The identity of the mouse.
        :param window_size: The window size used for signal formation.
        :param per_frame_values: The values for a particular feature.
        :return: a dictionary of the circular window features.
        """
        values = {}
        for op_name, op in self._window_operations.items():
            values[op_name] = self._compute_window_features_circular(
                per_frame_values, self._poses.identity_mask(identity),
                window_size, op)
        return values

    def _compute_window_feature(self, feature_values: dict,
                                frame_mask: np.ndarray, window_size: int,
                                op: typing.Callable) -> np.ndarray:
        """
        helper function to compute window feature values

        :param feature_values: dict of per frame feature values
        :param frame_mask: array indicating which frames are valid for the
        current identity
        :param window_size: number of frames (in each direction) to include
        in the window. The actual number of frames is 2 * window_size + 1
        :param op: function to perform the actual computation
        :return: dict containing feature values
        """
        values = {}
        for key, val in feature_values.items():
            values[f"{key}"] = op(val, window=window_size)

        return values

    def _compute_signal_features(
            self, freqs: np.ndarray, psd: dict,
            frame_mask: np.ndarray, op: typing.Callable, **kwargs) -> np.ndarray:
        """
        helper function to compute signal window feature values.

        :param freqs: frequency values for psd matrices
        :param psd: dict of power spectral density
        :param frame_mask: array indicating which frames are valid for the
        current identity
        :param op: function to perform the actual computation. Operation must
        accept frequencies and psd as input
        :param kwargs: additional keyword args used by op
        :return: numpy nd array containing feature values
        """
        values = {}
        for key, value in psd.items():
            values[key] = op(freqs, psd, **kwargs)

        return values

    def _compute_window_features_circular(
            self, feature_values: dict, frame_mask: np.ndarray,
            window_size: int, op: typing.Callable
    ) -> typing.Dict:
        """
        special case compute_window_features for circular measurements

        :param feature_values: dict of per-frame feature values
        :param frame_mask: numpy array that indicates if the frame is valid or
        not for the specific identity we are computing features for
        :param window_size:
        :param op:
        :return: dict with circular feature values
        """
        nframes = self._poses.num_frames
        values = {}

        for key, val in feature_values.items():

            op_result = np.zeros(val.shape)

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

                window_values = val[slice_start:slice_end][
                    slice_frames_valid == 1]

                op_result[i] = op(window_values)

            values[f"{key}"] = op_result

        return values
