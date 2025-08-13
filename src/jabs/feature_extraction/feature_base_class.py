import abc
import typing
import warnings

import numpy as np
from scipy import signal, stats

from jabs.feature_extraction.window_operations import signal_stats, window_stats
from jabs.pose_estimation import PoseEstimation


class Feature(abc.ABC):
    """Abstract Base Class to define a common interface for classes that implement one or more related features"""

    # each subclass needs to define this name and feature_names
    _name = None
    # list of feature names, correspond to columns of feature values
    _feature_names = None

    # requirements for this feature to be available
    _min_pose = 2
    _static_objects: typing.ClassVar[list[str]] = []

    # does this feature use circular window operations
    # Typically this is set to true for features that are angles
    _use_circular = False

    # standard window operations
    _window_operations: typing.ClassVar[dict[str, typing.Callable]] = {
        "mean": window_stats.window_mean,
        "median": window_stats.window_median,
        "std_dev": window_stats.window_std_dev,
        "skew": window_stats.window_skew,
        "kurtosis": window_stats.window_kurtosis,
        "max": window_stats.window_max,
        "min": window_stats.window_min,
    }
    _nan_fill_value = 0

    # signal processing operations
    _signal_operations: typing.ClassVar[dict[str, typing.Callable]] = {
        "fft_band": signal_stats.psd_mean_band,
        "psd_sum": signal_stats.psd_sum,
        "psd_max": signal_stats.psd_max,
        "psd_min": signal_stats.psd_min,
        "psd_mean": signal_stats.psd_mean,
        "psd_std_dev": signal_stats.psd_std_dev,
        "psd_skew": signal_stats.psd_skew,
        "psd_kurtosis": signal_stats.psd_kurtosis,
        "psd_median": signal_stats.psd_median,
        "psd_top_freq": signal_stats.psd_peak_freq,
    }

    # most angles are bearings in the range [-180, 180)
    # if an angle feature is in a different range, the subclass needs to override this
    _circular_window_operations: typing.ClassVar[dict[str, typing.Callable]] = {
        "mean": lambda x: stats.circmean(x, low=-180, high=180, nan_policy="omit"),
        "std_dev": lambda x: stats.circstd(x, low=-180, high=180, nan_policy="omit"),
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__()
        self._poses = poses
        self._pixel_scale = pixel_scale
        self._fps = poses.fps
        self._signal_bands = [
            {"band_low": 0.1, "band_high": 1.0},
            {"band_low": 1.0, "band_high": 3.0},
            {"band_low": 3.0, "band_high": 5.0},
            {"band_low": 5.0, "band_high": 8.0},
            {"band_low": 8.0, "band_high": 15.0},
        ]

        if self._name is None:
            raise NotImplementedError("Base class must override _name class member")

    @staticmethod
    def window_width(window_size: int) -> int:
        """return the width of the window used for computing features"""
        return 2 * window_size + 1

    @classmethod
    def name(cls) -> str:
        """return a string name of the feature"""
        return cls._name

    @classmethod
    def feature_names(cls) -> list[str]:
        """return a list of strings containing the names of the features for the feature set"""
        return cls._feature_names

    @classmethod
    def is_supported(cls, pose_version: int, static_objects: set[str], **kwargs) -> bool:
        """check that a feature is supported by a pose file

        Args:
            pose_version: pose file version
            static_objects: list of static object available in pose file
            **kwargs: additional attributes that can be used to
                determine if a feature is supported by the project, not
                used by the default method but might be used by
                subclasses

        Returns:
            True if the pose file supports the feature, false otherwise
        """
        # check that the minimum pose version is met
        if cls._min_pose > pose_version:
            return False

        # check that any static objects required by the feature are
        # available
        return all(obj in static_objects for obj in cls._static_objects)

    @abc.abstractmethod
    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """each FeatureSet subclass will implement this to compute the features in the set

        returns a ndarray containing the feature values.
        The feature set could be a single feature, where this would be a 1D
        numpy ndarray, or it could be a 2D ndarray for a set of related
        features (for example the pairwise point distances, which is a 2D
        ndarray where each row corresponds to the frame index, and each column
        is one of the pairwise point distances)
        """
        pass

    def window(self, identity: int, window_size: int, per_frame_features: dict) -> dict:
        """compute window feature values.

        Args:
            identity (int): subject identity
            window_size (int): window size NOTE: (actual window size is 2 * window_size + 1)
            per_frame_features (dict): dictionary of per frame values for this identity

        Returns:
            dict with feature values
        """
        circular_features = {}
        if self._use_circular:
            # This feature uses circular window operations. Some features that use the circular window operations
            # also include non-circular values, so we need to separate them out and compute them separately using
            # the standard window operations.
            # right now we assume that any feature that ends with " sine" or " cosine" is not circular.
            # (our angle features include sine and cosine values for each angle, and we want to compute standard
            # window features on those)
            non_circular = {
                k: v
                for k, v in per_frame_features.items()
                if k.endswith(" sine") or k.endswith(" cosine")
            }
            circular = {k: v for k, v in per_frame_features.items() if k not in non_circular}
            circular_features = self._window_circular(identity, window_size, circular)
        else:
            # feature does not use circular window operations, so we can compute standard window features on all values
            non_circular = per_frame_features

        # standard method for computing window features on non-circular values
        non_circular_features = self._window_standard(identity, window_size, non_circular)

        # non_circular_features and circular_features are both dicts of dicts, but they may have overlapping *top level* keys.
        merged = dict(non_circular_features)  # start with dict1's keys/values
        for k, v in circular_features.items():
            if k in merged:
                merged[k] = {**merged[k], **v}  # merge the inner dicts
            else:
                merged[k] = v

        return merged

    def _window_standard(self, identity: int, window_size: int, per_frame_values: dict) -> dict:
        """standard method for computing standard window feature values

        NOTE: some features may need to override this (for example, those with
        circular values such as angles)
        """
        values = {}
        for op in self._window_operations:
            values[op] = self._compute_window_feature(
                per_frame_values,
                self._poses.identity_mask(identity),
                window_size,
                self._window_operations[op],
            )
        # Also include signal features
        signal_features = self._window_signal(identity, window_size, per_frame_values)
        values.update(signal_features)
        return values

    def _window_signal(self, identity: int, window_size: int, per_frame_values: dict) -> dict:
        """The standard method for computing signal processing window features.

        Args:
            identity: The identity of the mouse.
            window_size: The window size used for signal formation.
            per_frame_values: The values for a particular feature.

        Returns:
            a dictionary of the signal processing features.
        """
        values = {}

        psd_data = {}
        # Obtain the PSD once
        for per_frame_key, per_frame in per_frame_values.items():
            adjusted_feature = np.nan_to_num(per_frame, nan=0)
            if len(adjusted_feature) < 2 * window_size + 1:
                adjusted_feature = np.pad(
                    adjusted_feature, (0, (2 * window_size + 1) - len(adjusted_feature))
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                freqs, ts, Zxx = signal.stft(
                    adjusted_feature,
                    fs=self._fps,
                    nperseg=window_size * 2 + 1,
                    noverlap=window_size * 2,
                    window="hann",
                    scaling="psd",
                    detrend="linear",
                )
            psd = np.abs(Zxx)
            psd_data[per_frame_key] = psd

        # Summarize the signal features
        for op_name, op in self._signal_operations.items():
            if op_name == "fft_band":
                for band in self._signal_bands:
                    values[f"{op_name}-{band['band_low']}Hz-{band['band_high']}Hz"] = (
                        self._compute_signal_features(
                            freqs,
                            psd_data,
                            self._poses.identity_mask(identity),
                            op,
                            **band,
                        )
                    )
            else:
                values[op_name] = self._compute_signal_features(
                    freqs, psd_data, self._poses.identity_mask(identity), op
                )

        return values

    def _window_circular(self, identity: int, window_size: int, per_frame_values: dict) -> dict:
        """helper function for overriding window features to be circular

        Args:
            identity: The identity of the mouse.
            window_size: The window size used for signal formation.
            per_frame_values: The values for a particular feature.

        Returns:
            a dictionary of the circular window features.
        """
        values = {}
        for op_name, op in self._circular_window_operations.items():
            values[op_name] = self._compute_window_features_circular(
                per_frame_values, self._poses.identity_mask(identity), window_size, op
            )
        return values

    def _compute_window_feature(
        self,
        feature_values: dict,
        frame_mask: np.ndarray,
        window_size: int,
        op: typing.Callable,
    ) -> dict:
        """helper function to compute window feature values

        Args:
            feature_values: dict of per frame feature values
            frame_mask: array indicating which frames are valid for the current identity
            window_size: number of frames (in each direction) to include in the window. The actual number of frames is 2 * window_size + 1
            op: function to perform the actual computation

        Returns:
            dict containing feature values
        """
        values = {}
        for key, val in feature_values.items():
            values[f"{key}"] = op(val, window=window_size)

        return values

    def _compute_signal_features(
        self,
        freqs: np.ndarray,
        psd: dict,
        frame_mask: np.ndarray,
        op: typing.Callable,
        **kwargs,
    ) -> dict:
        """helper function to compute signal window feature values.

        Args:
            freqs: frequency values for psd matrices
            psd: dict of power spectral density
            frame_mask: array indicating which frames are valid for the current identity
            op: function to perform the actual computation. Operation must accept frequencies and psd as input
            **kwargs: additional keyword args used by op



        Returns:
            numpy nd array containing feature values
        """
        values = {}
        for key, value in psd.items():
            values[f"{key}"] = op(freqs, value, **kwargs)

        return values

    def _compute_window_features_circular(
        self,
        feature_values: dict,
        frame_mask: np.ndarray,
        window_size: int,
        op: typing.Callable,
    ) -> dict:
        """special case compute_window_features for circular measurements

        Args:
            feature_values: dict of per-frame feature values
            frame_mask: numpy array that indicates if the frame is valid for the specific identity we are computing features for
            window_size: number of frames (in each direction) to include in the window. The actual number of frames is 2 * window_size + 1
            op: function to perform the actual computation, such as scipy.stats.circmean or scipy.stats.circstd

        Returns:
            dict with circular feature values
        """
        nframes = self._poses.num_frames
        values = {}

        for key, val in feature_values.items():
            op_result = np.full(val.shape, np.nan)

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

                window_values = val[slice_start:slice_end][slice_frames_valid == 1]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    op_result[i] = op(window_values)

            values[f"{key}"] = op_result

        return values
