import typing

import numpy as np
import scipy.stats
from scipy import signal
from scipy.stats import kurtosis, skew

from src.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from .social_distance import ClosestIdentityInfo # not sure if needed
    from src.pose_estimation import PoseEstimation


# All of my signals can come from moments, hu_moments, and ellipse_fitting.
# I think perimeter can be computed from the moments.  dx, dy can be computed
# using x & y from the ellipse fitting (technically computed from the moments).
# All of this should be available from the pose_est_v6 object.
signals = ['m00', 'perimeter', 'w', 'l', "wl_ratio", "dx", "dy", "dx2_plus_dy2", 'hu0', 'hu1',
    'hu2', 'hu3', 'hu4', 'hu5', 'hu6']

# Broken up into 3 sublists because __MPL_<i> columns have been changed to become dynamic.  
# I am waiting to figure out how to design this.  Maybe mpl colnames will grow dynamically
# as a response to the user interface.
colnames_surfix = ["__k", "__k_psd", "__s_psd"]
mpl_colnames = ["__MPL_1", "__MPL_3", "__MPL_5", "__MPL_8", "__MPL_15"]
colnames_surfix += mpl_colnames
colnames_surfix += ["__Tot_PSD", "__Max_PSD", "__Min_PSD", "__Ave_PSD", "__Std_PSD", "__Ave_Signal", 
            "__Std_Signal", "__Max_Signal", "__Min_Signal", "__Top_Signal", "__Med_Signal", "__Med_PSD"]

colnames = ["unique_epoch_id", "video", "Stage"] + [s + c for s in signals for c in colnames_surfix]
features = ['m00', 'perimeter', 'w', 'l', "wl_ratio", "dx", "dy", "dx2_plus_dy2", 'hu0', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6']

# hard-coded values from sleep paper:
samplerate = 30.
filterOrder = 5
criticalFrequencies = [1. / samplerate, 29. / samplerate]
a, b = signal.butter(filterOrder, criticalFrequencies, 'bandpass')


class FFT(Feature):
    ''' This class contains the signal processing features outlined in:
    High-throughput visual assessment of sleep stages in mice using machine learning, Geuther, et al.
    '''
    _name = "fft"
    _feature_names = ['m00', 'perimeter', 'w', 'l', "wl_ratio", "dx", "dy", "dx2_plus_dy2", 'hu0', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6']
    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        """
        compute the value of the per frame features for a specific identity
        :param identity: identity to compute features for
        :return: np.ndarray with feature values
        """
        return np.ndarray((self._poses.num_frames, len(FFT._feature_names)))

    # this is the standard way of working with window features outlined in the parent abstract Feature
    # class.
    '''
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
    '''