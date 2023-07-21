import unittest

import pandas as pd
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# project imports
from .seg_test_utils import SegDataBaseClass as SBC
from src.feature_extraction.social_features.signal_processing import colnames


class TestTemporalFeatures(unittest.TestCase):
    @unittest.skip("")
    def test_1(self):
        """ verify colnames values.
        """
        res = set(["unique_epoch_id", "video", "Stage"])
        for s in ['m00', 'perimeter', 'w', 'l', "wl_ratio", "dx", "dy", "dx2_plus_dy2", 'hu0', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6']:
            for c in ["__k", "__k_psd", "__s_psd", "__MPL_1", "__MPL_3",
                "__MPL_5", "__MPL_8", "__MPL_15", "__Tot_PSD", "__Max_PSD", "__Min_PSD",
                "__Ave_PSD", "__Std_PSD", "__Ave_Signal", "__Std_Signal",
                "__Max_Signal", "__Min_Signal", "__Top_Signal", "__Med_Signal", "__Med_PSD"]:
                res.add(s + c)

        assert len(res - set(colnames)) == 0

    def test_data_frame_features(self):
        ''' A simple testing ground for gaining an intuition about the pandas dataframe from the 
        sleep paper code.
        '''

        data = {
            "x": [1, 2.5, 3.77, 2.4],
            "y": [-1, -2, -1/2, 3]
        }
        df = pd.DataFrame(data)
        df["dx"] = df["x"].diff()
        df["dy"] = df["y"].diff()
        df["dx2_plus_dy2"] = df["dx"]**2 + df["dy"]**2

        df = df.dropna()

        print(df)

        assert True
