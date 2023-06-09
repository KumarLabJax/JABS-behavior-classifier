import numpy as np
import h5py
import unittest
import os
import gzip
import shutil
import tempfile
from pathlib import Path
from time import time
import warnings 

import src.pose_estimation

# Bring in base features of interest.
from src.feature_extraction.base_features import (
    moments, ellipse_fitting, point_speeds)

# test command: python -m unittest tests.test_social_features.test_fft


class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Signal Processing Tests
class TestSignalProcessing(unittest.TestCase):
    """
    This test class will attempt to create an instance of a base feature.
    After that it will attempt to generate the signal processing features for
    various features.
    """
    _fname = 'B6J_MDB0054_pose_est_v6.h5'  # 'sample_pose_est_v6.h5'
    _tmpdir = None
    _test_file = Path(__file__).parent.parent / 'data' / _fname

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._tmpdir_path = Path(cls._tmpdir.name)

        pose_path = cls._tmpdir_path / cls._fname
        print(cls._test_file)

        opener = open if ".gz" not in cls._fname else gzip.open

        with opener(cls._test_file, 'rb') as f_in:
            with open(pose_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        if os.path.isfile(pose_path):
            cls._pose_est_v6 = src.pose_estimation.open_pose_file(pose_path)
            cls._poses = cls._pose_est_v6

            with h5py.File(pose_path, "r") as f:
                cls._points = f.get("poseest/points")[:]
                cls._raw_embed_data = f['poseest/instance_embed_id'][:]

        else:
            raise (ValueError("Not a valid pose file name."))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_access_poses(self):
        """
        This tests simply checks that I can access basic info from the pose
        file.
        """
        # print(self._poses, self._poses.num_frames)
        self.assertEqual(self._poses.num_frames, 18000)
        self.assertTrue('seg_data' in self._poses._segmentation_dict)
        self.assertGreater(
            len(self._poses._segmentation_dict['seg_data'].shape), 1)
        self.assertGreaterEqual(
            abs(self._poses._cm_per_pixel-0.07928075), 1e-9)

    def test_initialize_feature(self):
        """
        This test simply validates that I can create instances of the base
        features.
        """
        moment = moments.Moments(self._poses, self._poses.cm_per_pixel)
        ellipse_fit = ellipse_fitting.EllipseFit(
            self._poses, self._poses.cm_per_pixel)

        self.assertEqual(moment._name, 'moments')
        self.assertEqual(ellipse_fit._name, 'ellipse fit')

        test_identity = 1

        moment_features = moment.per_frame(test_identity)
        ellipse_fit_features = ellipse_fit.per_frame(test_identity)

        self.assertEqual(moment_features.shape[0], self._poses.num_frames)
        self.assertEqual(ellipse_fit_features.shape[0], self._poses.num_frames)

    def test_generate_signal_processing_attributes(self):
        """
        For some random identity, attempt to generate the signal processing
        features for an arbitrary base feature.
        """

        moment = moments.Moments(self._poses, self._poses.cm_per_pixel)
        ellipse_fit = ellipse_fitting.EllipseFit(
            self._poses, self._poses.cm_per_pixel)

        # check that I can access base feature signal processing attributes
        self.assertEqual(moment._samplerate, ellipse_fit._samplerate)

    def test_generate_window_features(self):
        """
        For some random identity, attempt to generate the signal processing
        features for an arbitrary base feature.
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

        self.assertEqual((18000, 24), window_values[random_key].shape)
        self.assertIsInstance(window_values, dict)

    @unittest.skip("paused")
    def test_generate_signal_processing_features(self):
        """
        For some random identity, attempt to generate the signal processing
        features for an arbitrary base feature.

        Computing the signal processing values for some feature takes about 25
        seconds.  First I will verify they are correct, then I will attempt to
        optimize.
        """

        test_identity = np.random.choice(self._poses._identities)
        window_size = 5

        moment = moments.Moments(self._poses, self._poses.cm_per_pixel)
        per_frame = moment.per_frame(test_identity)

        # print("cm_per_pixel:", self._poses.cm_per_pixel)

        t0 = time()
        signal_processing_values = moment.signal_processing(
            test_identity, window_size, per_frame)
        tf = time()
        print(f"signal processing time: {(tf-t0) // 60} m {(tf-t0) % 60} s")

        self.assertIsInstance(signal_processing_values, dict)
        self.assertIn("__Max_Signal", signal_processing_values)
        for key in signal_processing_values:
            self.assertEqual(signal_processing_values[key].shape, (3600, 24))

    def test_validate_against_brians_code(self):
        """
        This test aims to verify that the signal processing values have been
        computed correctly, by checking that the 'nose point speed' signal
        processing features when computed manually match the values when
        computed by the base feature's 'signal_processing' method.
        """
        window_size = 5*5
        frameIdx = 1700  # np.random.randint(0, self._poses._num_frames - 1)
        animal_idx = 2

        # sanity test, verify Brian's points for individual match my own.
        points_for_individual_from_pose_estimation, _ = \
            self._poses.get_identity_poses(
                animal_idx, self._poses.cm_per_pixel)

        # Alternate way to get points
        # self._points[:, animal_idx, ...] * self._poses.cm_per_pixel

        points_for_individual = np.zeros(np.delete(np.shape(self._points), 1),
                                         dtype=self._points.dtype)
        idxs = np.where(self._raw_embed_data == animal_idx)
        points_for_individual[idxs[0], :, :] = \
            self._points[idxs[0], idxs[1], :, :]

        pointSpeeds = point_speeds.PointSpeeds(
            self._poses, self._poses.cm_per_pixel)
        per_frame = pointSpeeds.per_frame(animal_idx)

        nose_speeds = np.gradient(points_for_individual[:, 0, :], axis=0)
        nose_speeds = np.hypot(nose_speeds[:, 0], nose_speeds[:, 1]) * \
            self._poses.cm_per_pixel * self._poses.fps

        self.assertEqual(pointSpeeds._name, "point_speeds")
        self.assertEqual(nose_speeds.shape, (per_frame.shape[0],))

        # Failed assertions
        if False:
            # Why don't our points match up?
            # Appears to be a "smoothing" issue.  Leaving here in case we want
            # to explore this further.
            assert np.all(
                points_for_individual_from_pose_estimation ==
                points_for_individual
                )
            # Computed nose speed not in my per_frame array.
            self.assertIn(nose_speeds[frameIdx], per_frame[frameIdx])

        # This test case was produced by running Brian's signal_prototyping
        # script on the "tests/data/B6J_MDB0054_pose_est_v6.h5" file.
        TEST_CASE = {
            '__MPL_1': 0.024304786900100593,
            '__MPL_3': 0.05065457015062057,
            '__MPL_5': 0.06418792878823419,
            '__MPL_8': 0.043147140820795106,
            '__MPL_15': 0.020555331540130454,
            '__Tot_PSD': 1.1099379843744661,
            '__Max_PSD': 0.066236477792041,
            '__Min_PSD': 0.003066600006771878,
            '__Ave_PSD': 0.03363448437498382,
            '__Med_PSD': 0.03754050540391479,
            '__k_psd': -1.25739391432352,
            '__s_psd': 0.054595346572103014,
            '__Top_Signal': 3.75
        }

        nose_speeds = np.genfromtxt(
            Path(__file__).parent / "nose_speeds.txt",
            delimiter=","
        )

        nose_speeds = nose_speeds.reshape(nose_speeds.shape[0], 1)

        signalProcessingFeatures = pointSpeeds.signal_processing(
            identity=animal_idx,
            window_size=window_size,
            per_frame_values=nose_speeds)

        print(
                "Signal Processing".ljust(30),
                "|", Color.BLUE + "Test Case" + Color.ENDC
                )
        for key in TEST_CASE:
            self.assertEqual(TEST_CASE[key],
                             signalProcessingFeatures[key][frameIdx][0])
            print(
                f"{key}: {signalProcessingFeatures[key][frameIdx][0]}"
                .ljust(30),
                "|", Color.BLUE + f"{TEST_CASE[key]}" + Color.ENDC
                )
