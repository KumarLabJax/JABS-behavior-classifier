import unittest
import gzip
import numpy as np
from pathlib import Path
import tempfile
import shutil

# project imports
from src.jabs.feature_extraction.segmentation_features import SegmentationFeatureGroup, Moments
from .seg_test_utils import SegDataBaseClass as SBC
import src.jabs.pose_estimation as pose_est


class TestImportSrc(unittest.TestCase):
    @unittest.skip("")
    def test(self):
        """
        # error: FileNotFoundError: Could not find module '<path>\geos_c.dll' (or one of its dependencies). 
        # Try using the full path with constructor syntax.
        # description: A bug in my shapely distro resulted in a nested import failure.
        # fix: pip unistall shapely; pip install shapely
        """

        assert True
        print("...test 1 complete")


class TestFeatureNameLength(unittest.TestCase):
    """Simple validation of the length of the feature_names list."""
    @unittest.skip("")
    def test(self):
        from src.jabs.utils.utilities import n_choose_r
        import src.jabs.pose_estimation as P
        import src.jabs.feature_extraction.base_features.pairwise_distances as pd

        assert len(pd._init_feature_names()) == n_choose_r(len(P.PoseEstimation.KeypointIndex), 2)

        print("...test 2 complete")


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

        moment_keys = {'m00', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03'}

        assert moment_keys - set(momentsFeature._moments_to_use) == set()

    def test_moments_per_frame(self):
        # initialize moments Feature for first identity
        momentsFeature = self.feature_mods['moments']
        momentValues = momentsFeature.per_frame(1)
        # check that number of moments generated is same as number of frames in pose file
        assert len(momentValues['m00']) == self._pose_est_v6.num_frames
