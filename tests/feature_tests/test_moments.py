import unittest
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# project imports
from .seg_test_utils import SegDataBaseClass as SBC
from src.feature_extraction.base_features.moments import Moments

class TestImportSrc(unittest.TestCase):
    @unittest.skip("")
    def test(self):
        """
        # error: FileNotFoundError: Could not find module '<path>\geos_c.dll' (or one of its dependencies). 
        # Try using the full path with constructor syntax.
        # description: A bug in my shapely distro resulted in a nested import failure.
        # fix: pip unistall shapely; pip install shapely
        """

        import src.pose_estimation
        assert True
        print("...test 1 complete")


class TestFeatureNameLength(unittest.TestCase):
    '''Simple validation of the length of the feature_names list.'''
    @unittest.skip("")
    def test(self):   
        from src.utils.utilities import n_choose_r
        import src.pose_estimation as P
        import src.feature_extraction.base_features.pairwise_distances as pd         
        
        assert len(pd._init_feature_names()) == n_choose_r(len(P.PoseEstimation.KeypointIndex), 2)

        print("...test 2 complete")
    

class TestClassInstantiation(SBC, unittest.TestCase):
    '''This test simply attempts instantiation of the Moments feature class.'''

    pixel_scale = 1.0
    
    def test_data(self):
        ''' Simple test of data integrity.
        '''
        assert len(self.seg_data.shape) == 5
        assert sum(self.seg_data.shape) > 0

    def test_posev6_instantiation(self):
        '''Test that the posev6 class can be instantiated.
        '''
        # test that pose estimation object was created, and that a segmentation_dict attribute is present.
        assert hasattr(self, "_pose_est_v6")
        assert hasattr(self._pose_est_v6, "_segmentation_dict")
        
        seg_data = self._pose_est_v6._segmentation_dict["seg_data"]

        # non-empty segmentation data
        assert sum(seg_data.shape) > 0
        
        # non-trivial data
        assert len(np.unique(seg_data)) > 1

        # test get_segmentation data for each identity
        for i in range(self.seg_data.shape[1]):
            assert np.array_equal(self.seg_data[:, i, ...], self._pose_est_v6.get_segmentation_data(i))
    
    def test_create_moment(self):
        '''The moments can be initialized properly.
        '''
        momentsFeature = Moments(self._pose_est_v6, self.pixel_scale)

        assert momentsFeature._name == "moments"

        moment_keys = {'m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 
            'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03'}

        assert moment_keys - set(momentsFeature._feature_names) == set()
    
    def test_moments_per_frame(self):

        # initialize moments Feature
        momentsFeature = Moments(self._pose_est_v6, self.pixel_scale)

        # now check the centroid location of each identity using the moments
        # for frame k
        SHOW_PLOT = True
        k = 1

        for i in range(self.seg_data.shape[1]):
            frames = momentsFeature.per_frame(i)
            kframe = frames[k, ...]
            # toggle numpy division warning:
            np.seterr(divide='ignore', invalid='ignore')
            cx  = np.divide(kframe[1], kframe[0])
            cy =  np.divide(kframe[2], kframe[0])
            np.seterr(divide='warn', invalid='warn')
            plt.plot(cx, cy, "*")

        if SHOW_PLOT:
            # plot actual segmentation data and verify by visual inspection that the centroids
            # appear in the center of each blob.
            for i in range(self.seg_data.shape[1]):
                for cnt in range(self.seg_data.shape[2]):
                    C = self.seg_data[k, i, cnt, ...]
                    y = C[:, 1]
                    x = C[:, 0]
                    plt.plot(x[x > 0], y[y > 0])

            plt.show()
        




        



