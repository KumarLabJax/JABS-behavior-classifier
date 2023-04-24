import typing

import cv2
import pandas as pd
import numpy as np
import scipy.stats
from scipy import signal
from scipy.stats import kurtosis, skew

from src.utils.utilities import rolling_window
from src.feature_extraction.feature_base_class import Feature
from ..base_features import hu_moments, ellipse_fitting, moments

assert hu_moments.HuMoments._name == 'hu moments'

if typing.TYPE_CHECKING or __name__== "__main__":
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

colnames = [s + c for s in signals for c in colnames_surfix]

# hard-coded values from sleep paper:
samplerate = 30.
filterOrder = 5
criticalFrequencies = [1. / samplerate, 29. / samplerate]
b, a = signal.butter(filterOrder, criticalFrequencies, 'bandpass')


class FFT(Feature):
    ''' This class contains the signal processing features outlined in:
    High-throughput visual assessment of sleep stages in mice using machine learning, Geuther, et al.
    '''
    _name = "fft"
    _feature_names = [s + c for s in signals for c in colnames_surfix]
    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

        # probably not the best way to do this.
        # basically i need to interact with the ellipse fit & moments features
        self.ellipse = ellipse_fitting.EllipseFit(poses, pixel_scale)
        self.humoments = hu_moments.HuMoments(poses, pixel_scale)
        self.moments = moments.Moments(poses, pixel_scale)

        # create a callable operation to be used by window.
        self._op = lambda wave: self.get_frequency_feature(wave, a, b, samplerate)
    
    
    def get_signals(self, identity: int) -> np.ndarray:
        ''' Compute all of the required signals for signal processing and aggregate them into a 
        numpy array.
        :param identity: identity to compute features for
        :return: np.ndarray with feature values
        '''
        ellipse_features = self.ellipse.per_frame(identity)
        moments_features = self.moments.per_frame(identity)
        humoments_features = self.hu_moments.per_frame(identity)

        # a) Moments features
        # since we only want m00
        m00 = moments_features[:, 0]

        # b) Ellipse Features
        # perimeter?
        seg_data = self._poses.get_segmentation_data(identity)

        # total perimeter for each frame
        perimeter_feature = np.zeros((self._poses.num_frames, 1))

        for frame in range(perimeter_feature.shape[0]):
            contours = seg_data[frame, ...]
            perimeter = 0
            for contour_idx in range(contours.shape[0]):
                contour = contours[contour_idx, ...].astype(np.int32)

                contour = contour[(contour[..., 0] > -1) & (contour[..., 1] > -1)]
                if contour.shape[0] == 0:
                    # we have hit an empty contour.
                    continue
                perimeter += cv2.arcLength(contour, True)
            perimeter_feature[frame] = perimeter

        # w, l, wl_ratio
        w = ellipse_features[:, 5]
        l = ellipse_features[:, 6]
        wl_ratio = w / l

        # dx, dy, dx2_plus_dy2
        diff = pd.DataFrame(ellipse_features[:, 0:2], columns=['x', 'y']).diff()

        dx = diff['x'].to_numpy()
        dy = diff['y'].to_numpy()

        dx2_plus_dy2 = dx ** 2 + dy ** 2

        # c) Hu moments
        # we want all hu moments

        # combine features into a single numpy array
        signals = np.hstack((m00, perimeter_feature, w, l, wl_ratio, dx, dy, dx2_plus_dy2, humoments_features))
        
        return signals


    def get_frequency_feature(self, wave: np.ndarray, a, b, samplerate)->dict:
        '''
        :param wave: an array representing the signals for each frame for a given identity for a particular window size.
        :param a: The denominator coefficient vector of the filter. 
        :param b: The numerator coefficient vector of the filter.
        :param samplerate: average number of samples per unit time.
        :return: np.ndarray with feature values
        '''
        wave = signal.filtfilt(b, a, wave)
        freqs, psd = signal.welch(wave)
        freqs = freqs * samplerate
        idx_1 = np.logical_and(freqs >= 0.1, freqs < 1)
        idx_3 = np.logical_and(freqs >= 1, freqs < 3)
        idx_5 = np.logical_and(freqs >= 3, freqs < 5)
        idx_8 = np.logical_and(freqs >= 5, freqs < 8)
        idx_15 = np.logical_and(freqs >= 8, freqs < 15)
        return {
            "__k": kurtosis(wave),
            "__k_psd": kurtosis(psd), 
            "__s_psd": skew(psd),
            "__MPL_1": np.mean(psd[idx_1]), 
            "__MPL_3": np.mean(psd[idx_3]),
			"__MPL_5": np.mean(psd[idx_5]), 
            "__MPL_8": np.mean(psd[idx_8]), 
            "__MPL_15": np.mean(psd[idx_15]), 
            "__Tot_PSD": np.sum(psd), 
            "__Max_PSD": max(psd), 
            "__Min_PSD": min(psd),
			"__Ave_PSD": np.average(psd), 
            "__Std_PSD": np.average(psd), 
            "__Ave_Signal": np.average(wave), 
            "__Std_Signal": np.std(wave),
			"__Max_Signal": max(wave), 
            "__Min_Signal": min(wave), 
            "__Top_Signal": freqs[psd == max(psd)][0], 
            "__Med_Signal": np.median(wave), 
            "__Med_PSD": np.median(psd)
        }


    def per_frame(self, identity: int) -> np.ndarray:
        """
        compute the value of the per frame features for a specific identity
        :param identity: identity to compute features for
        :return: np.ndarray with feature values
        """
        return self.get_signals(identity)


    # this is the standard way of working with window features outlined in the parent abstract Feature
    # class.
    # _compute_window_features should work for my 2d feature array.  
    # [x] I will however need an operation callable which represents the applied fft.
    # [x] What is this? self._poses.identity_mask(identity)
    #   a mask for each identity that indicates if it exists or not in the frame.


    def window(self, identity: int, window_size: int, per_frame_values: np.ndarray):
        """
        Overriding the standard method for computing window feature values.

        NOTE: Alternatively, I could make each dictionary entry from get_frequency_feature
        an 'operation', which is in turn invoked via _compute_window_feature.
        """
        return self._compute_window_feature(
                per_frame_values, self._poses.identity_mask(identity),
                window_size, self._op
            )
    

    def _compute_window_feature(self, feature_values: np.ndarray,
                            frame_mask: np.ndarray, window_size: int,
                            op: typing.Callable) -> np.ndarray:
        """
        Overriding the helper function to compute window feature values.  

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

        # The feature is 2D, the window features are computed for each column.
        i = 0
        for j in range(feature_values.shape[1]):
            windows = rolling_window(
                np.pad(feature_values[:, j], window_size),
                window_width
            )
            mx = np.ma.masked_array(windows, window_masks)

            fft_features = self._op(mx, axis=1)

            for feature_name in fft_features:
                values[:, i] = fft_features[feature_name]
                i += 1

        return values


if __name__=="__main__": assert True