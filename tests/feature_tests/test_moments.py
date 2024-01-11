import unittest
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
import gzip

# project imports
from .seg_test_utils import SegDataBaseClass as SBC
from src.feature_extraction.base_features.moments import Moments
import src.pose_estimation as pose_est

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
    

class TestClassInstantiation(unittest.TestCase):
    '''This test simply attempts instantiation of the Moments feature class.'''

    pixel_scale = 1.0
    dataPath = Path("data")
    dataFileName = "B6J_MDB0054_pose_est_v6.h5"

    @classmethod
    def setUpClass(cls) -> None:
        with h5py.File("data/B6J_MDB0054_pose_est_v6.h5", "r") as f:
            cls.seg_data = f.get("poseest/seg_data")[:]
        
        # create pose estimation v6 file, which also contains segmentation data
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._tmpdir_path = Path(cls._tmpdir.name)

        with open(cls.dataPath / cls.dataFileName, 'rb') as f_in:
            with open(cls._tmpdir_path / cls.dataFileName.replace(".gz", ""),
                        'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        cls._pose_est_v6 = pose_est.open_pose_file(
            cls._tmpdir_path / cls.dataFileName.replace(".gz", ""))

    @ classmethod
    def tearDown(cls):
        if cls._tmpdir:
            cls._tmpdir.cleanup()
    
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
        """
        converting from pixel to cm before computing moments.

        not scaled:
        [3.69500000e+03 1.15126733e+06 3.84717333e+05 3.59679382e+08 
        1.20012430e+08 4.19945385e+07 1.12674928e+11 3.75437007e+10 
        1.31269494e+10 4.80189055e+09 9.73976808e+05 1.44365888e+05 
        1.93840140e+06 1.11498633e+06 4.51252837e+06 1.24648152e+07 
        2.58418039e+07 7.13378030e-02 1.05739122e-02 1.41975965e-01 
        1.34348689e-03 5.43730680e-03 1.50193015e-02 3.11377136e-02]

        scaled:
        [2.08673524e+01 4.88602273e+02 1.63275507e+02 1.14715272e+04 
        3.82764728e+03 1.33936356e+03 2.70059449e+05 8.99847992e+04 
        3.14626914e+04 1.15091780e+04 3.10637964e+01 4.60437084e+00 
        6.18229070e+01 2.67240616e+00 1.08156491e+01 2.98756944e+01 
        6.19376694e+01 7.13378194e-02 1.05739096e-02 1.41975930e-01 
        1.34348892e-03 5.43731147e-03 1.50192979e-02 3.11376966e-02]
        """
        

        # initialize moments Feature

        # so find out where I create the moments object and be sure to initialize with self._pose_est_v6.cm_per_pixel.
        momentsFeature = Moments(self._pose_est_v6, self._pose_est_v6.cm_per_pixel)
        #per_frame = momentsFeature.per_frame(1)
        #print(per_frame[20])
         
        # now check the centroid location of each identity using the moments
        # for frame k
        SHOW_PLOT = False
        k = 1

        # toggle numpy division warning:
        np.seterr(divide='ignore', invalid='ignore')

        for i in range(self.seg_data.shape[1]):
            frames = momentsFeature.per_frame(i)
            kframe = frames[k, ...]
            print(kframe); break
            cx  = np.divide(kframe[1], kframe[0])
            cy =  np.divide(kframe[2], kframe[0])

            plt.plot(cx, cy, "*")
        
        np.seterr(divide='warn', invalid='warn')
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
        




        



