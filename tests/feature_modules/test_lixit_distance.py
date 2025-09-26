# TODO these tests need to be fixed, they were broken during a change in how features were stored/retrieved
import unittest

import numpy as np

import jabs.feature_extraction.landmark_features.lixit as lixit
from tests.feature_modules.base import TestFeatureBase


class TestCornerFeatures(TestFeatureBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        pixel_scale = cls._pose_est_v5.cm_per_pixel
        cls.distance_info = lixit.LixitDistanceInfo(cls._pose_est_v5, pixel_scale)
        cls.lixit_distance = lixit.DistanceToLixit(
            cls._pose_est_v5, pixel_scale, cls.distance_info
        )

    @unittest.skip("")
    def test_dimensions(self):
        # check dimensions of per frame feature values
        for i in range(self._pose_est_v5.num_identities):
            distances = self.lixit_distance.per_frame(i)
            self.assertEqual(distances.shape, (self._pose_est_v5.num_frames,))

            # check dimensions of window feature values
            dist_window_values = self.lixit_distance.window(i, 5, distances)
            for op in dist_window_values:
                self.assertEqual(dist_window_values[op].shape, (self._pose_est_v5.num_frames,))

    @unittest.skip("")
    def test_distances_greater_equal_zero(self):
        for i in range(self._pose_est_v5.num_identities):
            distances = self.lixit_distance.per_frame(i)
            # check distances are >= 0
            self.assertTrue((distances >= 0).all())

    @unittest.skip("")
    def test_computation(self):
        # spot check some distance values for identity 0
        expected = np.asarray(
            [
                10.44161892,
                10.51272678,
                10.60188961,
                10.67293644,
                10.7262249,
                10.70890999,
                10.70890999,
                10.7802515,
                10.7802515,
                10.90649891,
            ],
            dtype=np.float32,
        )
        actual = self.lixit_distance.per_frame(0)
        print(actual)
        for i in range(expected.shape[0]):
            self.assertAlmostEqual(expected[i], actual[i])
