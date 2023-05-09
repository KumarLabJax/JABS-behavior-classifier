import numpy as np
import h5py
import unittest
import os
from typing import List 
import gzip
import shutil
import tempfile
import unittest
from pathlib import Path
from time import time

import src.pose_estimation

# test command: python -m unittest tests.test_social_features.test_fft

# Bring in base features of interest.
from src.feature_extraction.base_features import moments, ellipse_fitting

# Signal Processing Tests
class TestSignalProcessing(unittest.TestCase):
    """
    This test class will attempt to create an instance of a base feature.  After that it will attempt to 
    generate the signal processing features for various features.
    """
    
    _tmpdir = None
    _test_file = Path(__file__).parent.parent / 'data' / 'sample_pose_est_v6.h5.gz'

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._tmpdir_path = Path(cls._tmpdir.name)

        pose_path = cls._tmpdir_path / 'sample_pose_est_v6.h5'
        print(cls._test_file)

        with gzip.open(cls._test_file, 'rb') as f_in:
            with open(pose_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        if os.path.isfile(pose_path):
            cls._pose_est_v6 = src.pose_estimation.open_pose_file(pose_path)
            cls._poses = cls._pose_est_v6

        else: 
            raise(ValueError("Not a valid pose file name."))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_access_poses(self):
        """
        This tests simply checks that I can access basic info from the pose file.
        """
        # print(self._poses, self._poses.num_frames)
        self.assertEqual(self._poses.num_frames, 3600)
        self.assertTrue('seg_data' in self._poses._segmentation_dict)
        self.assertGreater(len(self._poses._segmentation_dict['seg_data'].shape), 1)
        self.assertGreaterEqual(abs(self._poses._cm_per_pixel-0.07928075), 1e-9)
    
    def test_initialize_feature(self):
        """
        This test simply validates that I can create instances of the base features.
        """
        moment = moments.Moments(self._poses, self._poses.cm_per_pixel)
        ellipse_fit = ellipse_fitting.EllipseFit(self._poses, self._poses.cm_per_pixel)

        self.assertEqual(moment._name, 'moments')
        self.assertEqual(ellipse_fit._name, 'ellipse fit')

        test_identity = 1

        moment_features = moment.per_frame(test_identity)
        ellipse_fit_features = ellipse_fit.per_frame(test_identity)

        self.assertEqual(moment_features.shape[0], self._poses.num_frames)
        
        # BUG [RESOLVED] Attribute Error:
        # x = self.feature_names.index('x') AttributeError: 'function' object has no attribute 'index'.
        # feature names was missing a _ prefix.

        self.assertEqual(ellipse_fit_features.shape[0], self._poses.num_frames)


    def test_generate_signal_processing_attributes(self):
        """
        For some random identity, attempt to generate the signal processing features for an arbitrary base feature.
        """
        # print(self._poses._identities)

        test_identity = np.random.choice(self._poses._identities)

        moment = moments.Moments(self._poses, self._poses.cm_per_pixel)
        ellipse_fit = ellipse_fitting.EllipseFit(self._poses, self._poses.cm_per_pixel)
        
        # check that I can access base feature signal processing attributes
        self.assertEqual(moment._samplerate, ellipse_fit._samplerate)
 
    def test_generate_window_features(self):
        """
        For some random identity, attempt to generate the signal processing features for an arbitrary base feature.
        """

        test_identity = np.random.choice(self._poses._identities)
        window_size = 5
        moment = moments.Moments(self._poses, self._poses.cm_per_pixel)
        per_frame = moment.per_frame(test_identity)
        t0 = time()
        window_values = moment.window(test_identity, window_size, per_frame)
        tf = time()
        
        random_key = np.random.choice([i for i in window_values.keys()])

        if False:
            print(f"window values [{random_key}]:", window_values[random_key])
            print(f"window time: {(tf-t0) // 60} m {(tf-t0) % 60:.4f} s")

        self.assertEqual((3600, 24), window_values[random_key].shape)
        self.assertIsInstance(window_values, dict)

    # @unittest.skip("paused")
    def test_generate_signal_processing_features(self):
        """
        For some random identity, attempt to generate the signal processing features for an arbitrary base feature.
        """

        test_identity = np.random.choice(self._poses._identities)
        window_size = 5

        moment = moments.Moments(self._poses, self._poses.cm_per_pixel)
        per_frame = moment.per_frame(test_identity)

        t0 = time()
        signal_processing_values = moment.signal_processing(test_identity, window_size, per_frame)
        tf = time()
        print(f"signal processing time: {(tf-t0) // 60} m {(tf-t0) % 60} s")

        self.assertIsInstance(signal_processing_values, dict)
        self.assertIn("__Max_Signal", signal_processing_values)
        for key in signal_processing_values:
            self.assertEqual(signal_processing_values[key].shape, (3600, 24))


# helper functions
def get_file_nth_parent_directory(n: int, file:str=__file__, target_path:List=[""], 
                                  _sep='\\' if '\\' in __file__ else '/')->str:
    """
    Return the path of the nth parent directory of a file.
    [deprecated] the Path module offers a built-in method for this.
    """
    return get_file_nth_parent_directory(n - 1, os.path.dirname(file), target_path=target_path) \
        if n > 0 else f"{file}{_sep}{_sep.join(target_path)}"

# 1. Sanity checks
if False:
    class TestScientificComputingBasics(unittest.TestCase):
        pass


if False:
    class TestRollingWindow(unittest.TestCase):
        '''
        This test is designed for the utils/utilities.py rolling_window
        method.  This is used in the Feature base class object to compute
        window features.
        '''

        log = True 

        @classmethod
        def setUpClass(cls) -> None:
            import src.utils.utilities as utils
            cls.rolling_window = utils.rolling_window

        @unittest.skip("resolved")
        def test_rolling_window(self):
            '''
            segmentation data has the following format:
            (frames, identities, contours, contour length, xy-point)

            Spent too much time here.  
            Basically I am curious about the difference in invoking feature_base_class/_compute_window_feature/rolling_window
            on 1D vs 2D features.  It is not obvious to me that the 2D rolling is correct.

            1D: (frames - window_size, window_size) vs 2D: (frames, features - window_size, window_size) 
            '''
            step_size = 1
            window_size = 6
            num_frames = 3600

            data = [
                ('points', (num_frames, 5, 12, 2)),       # ignore
                ('seg data', (num_frames, 5, 4, 319, 2)), # ignore
                ('1D', (num_frames, )),
                ('2D', (num_frames, 8))
            ]

            # The way this is used in code is for feature_values arrays.  Let me check the shape of these arrays.
            # It appears to be (number of frames,), which makes intuitive sense for 1D arrays, but what about 2D 
            # feature values arrays such as hu_moments, moments, and ellipse_fitting which all have shapes of the
            # form: (num_frames, len(self._feature_names)).  I checked and pairwise_distances also uses this 
            # 2D structure.  Next I will check if pairwise_distances is ever used with rolling_window. 

            # resolved:
            # I overlooked something obvious, 
            # The rolling_window code is used with: if feature_values.ndim == 1: 
            # thus it is not applicable for 2D features like I was worried about.
            
            rolling_shape = lambda shape: shape[:-1] + (shape[-1] - window_size + 1 - step_size + 1, window_size)
            
            for name, shape in data[2:]:

                A = np.random.randint(0, 1000, shape)
                roll_shape = rolling_shape(shape)
    
                window = TestRollingWindow.rolling_window(A, window_size)

                if TestRollingWindow.log:
                    print(f"{name} shape: ", shape, "roll shape:", roll_shape)

                assert len(roll_shape) - 1 == len(shape)
                assert window.shape == roll_shape
                assert isinstance(window, np.ndarray)


if __name__ == "__main__":

    for n in range(0, 4):
            p = get_file_nth_parent_directory(n, target_path=["data","sample_pose_est_v6.h5.gz"])
            print(n, p)
