import typing

import numpy as np
import scipy.stats
from scipy import signal
from scipy.stats import kurtosis, skew

from src.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from .social_distance import ClosestIdentityInfo # not sure if needed
    from src.pose_estimation import PoseEstimation


signals = ['m00', 'perimeter', 'w', 'l', "wl_ratio", "dx", "dy", "dx2_plus_dy2", 'hu0', 'hu1',
    'hu2', 'hu3', 'hu4', 'hu5', 'hu6']
colnames_surfix = ["__k", "__k_psd", "__s_psd", "__MPL_1", "__MPL_3",
                    "__MPL_5", "__MPL_8", "__MPL_15", "__Tot_PSD", "__Max_PSD", "__Min_PSD",
                    "__Ave_PSD", "__Std_PSD", "__Ave_Signal", "__Std_Signal",
                    "__Max_Signal", "__Min_Signal", "__Top_Signal", "__Med_Signal", "__Med_PSD"]
colnames = ["unique_epoch_id", "video", "Stage"] + [s + c for s in signals for c in colnames_surfix]
features = ['m00', 'perimeter', 'w', 'l', "wl_ratio", "dx", "dy", "dx2_plus_dy2", 'hu0', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6']



# hard-coded values from sleep paper:
samplerate = 30.
filterOrder = 5
criticalFrequencies = [1. / samplerate, 29. / samplerate]
a, b = signal.butter(filterOrder, criticalFrequencies, 'bandpass')

