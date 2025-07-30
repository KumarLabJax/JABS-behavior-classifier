import unittest

import numpy as np

# project imports
from jabs.feature_extraction.segmentation_features import Moments

from .seg_test_utils import SegDataBaseClass as SBC


class TestClassInstantiation(SBC, unittest.TestCase):
    """This test simply attempts instantiation of the Moments feature class."""

    def test_data(self):
        """Simple test of data integrity."""
        seg_data = self._pose_est_v6._segmentation_dict["seg_data"]

        assert len(seg_data.shape) == 5
        assert sum(seg_data.shape) > 0

    def test_posev6_instantiation(self):
        """Test that the posev6 class can be instantiated."""
        # test that pose estimation object was created, and that a segmentation_dict attribute is present.
        assert hasattr(self, "_pose_est_v6")
        assert hasattr(self._pose_est_v6, "_segmentation_dict")

        seg_data = self._pose_est_v6._segmentation_dict["seg_data"]

        # non-empty segmentation data
        assert sum(seg_data.shape) > 0

        # non-trivial data
        assert len(np.unique(seg_data)) > 1

        # test get_segmentation data for each identity
        for i in range(seg_data.shape[1]):
            assert np.array_equal(seg_data[:, i, ...], self._pose_est_v6.get_segmentation_data(i))

    def test_create_moment(self):
        """The moments can be initialized properly."""
        momentsFeature = Moments(self._pose_est_v6, self.pixel_scale, self._moment_cache)

        assert momentsFeature._name == "moments"

        moment_keys = {
            "m00",
            "mu20",
            "mu11",
            "mu02",
            "mu30",
            "mu21",
            "mu12",
            "mu03",
            "nu20",
            "nu11",
            "nu02",
            "nu30",
            "nu21",
            "nu12",
            "nu03",
        }

        assert moment_keys - set(momentsFeature._moments_to_use) == set()

    def test_moments_per_frame(self):
        """Test that moments can be computed per frame."""
        # initialize moments Feature for first identity
        momentsFeature = self.feature_mods["moments"]
        momentValues = momentsFeature.per_frame(1)
        # check that number of moments generated is same as number of frames in pose file
        assert len(momentValues["m00"]) == self._pose_est_v6.num_frames
