import gzip
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import src.pose_estimation

_TEST_FILES = [
    'sample_pose_est_v3.h5.gz',
    'sample_pose_est_v4.h5.gz',
    'sample_pose_est_v5.h5.gz'
]


class TestOpenPose(unittest.TestCase):
    _tmpdir = None
    _test_data_dir = Path(__file__).parent / 'data'

    @classmethod
    def setUpClass(cls) -> None:
        '''11/14/2022 - This method throws an error: gzip.BadGzipFile: Not a gzipped file (x89H).  Noticed this when 
        I tried to create my own base feature tests.  Is gzip.open() necessary, simply reading the file without gzip 
        appears to work.  Please see my file test_pose_ancillary.py
        '''

        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._tmpdir_path = Path(cls._tmpdir.name)

        # decompress pose file into tempdir

        for f in _TEST_FILES:
            with gzip.open(cls._test_data_dir / f, 'rb') as f_in:
                with open(cls._tmpdir_path / f.replace('.h5.gz', '.h5'),
                          'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        cls._pose_est_v3 = src.pose_estimation.open_pose_file(
            cls._tmpdir_path / 'sample_pose_est_v3.h5')

        cls._pose_est_v4 = src.pose_estimation.open_pose_file(
            cls._tmpdir_path / 'sample_pose_est_v4.h5')

        cls._pose_est_v5 = src.pose_estimation.open_pose_file(
            cls._tmpdir_path / 'sample_pose_est_v5.h5')

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_open_pose_est_v3(self) -> None:
        """ test that open_pose_file can open a V3 pose file """
        self.assertIsInstance(self._pose_est_v3,
                              src.pose_estimation.PoseEstimationV3)
        self.assertEqual(self._pose_est_v3.format_major_version, 3)

    def test_open_pose_est_v4(self) -> None:
        """ test that open_pose_file can open a V4 pose file """
        self.assertIsInstance(self._pose_est_v4,
                              src.pose_estimation.PoseEstimationV4)
        self.assertEqual(self._pose_est_v4.format_major_version, 4)

    def test_open_pose_est_v5(self) -> None:
        """ test that open_pose_file can open a V5 pose file """
        self.assertIsInstance(self._pose_est_v5,
                              src.pose_estimation.PoseEstimationV5)
        self.assertEqual(self._pose_est_v5.format_major_version, 5)

        # the test v5 pose file has 'corners' in static objects dataset
        static_objs = self._pose_est_v5.static_objects
        self.assertTrue('corners' in static_objs)
        self.assertEqual(static_objs['corners'].shape, (4, 2))

    def test_get_points(self) -> None:
        """ test getting pose points from PoseEstimation instance """
        points, point_mask = self._pose_est_v4.get_identity_poses(0)
        nframes = self._pose_est_v4.num_frames

        self.assertEqual(points.shape, (nframes, 12, 2))
        self.assertEqual(point_mask.shape, (nframes, 12))

    def test_get_points_single_frame(self) -> None:
        """
        test getting pose points for single frame from PoseEstimation instance
        """
        # get points for the 10th frame for identity 0
        points, point_mask = self._pose_est_v4.get_points(10, 0)

        self.assertEqual(points.shape, (12, 2))
        self.assertEqual(point_mask.shape, (12,))

        # compare to getting all points
        points_all_frame, point_mask_all_frames = \
            self._pose_est_v4.get_identity_poses(0)
        self.assertTrue((points == points_all_frame[10, :]).all())
        self.assertTrue((point_mask == point_mask_all_frames[10, :]).all())

    def test_get_points_out_of_range(self) -> None:
        with self.assertRaises(IndexError):
            _, _ = self._pose_est_v4.get_points(1000000, 0)

    def test_scaling_points(self) -> None:
        """ test scaling points """
        points, _ = self._pose_est_v4.get_points(10, 0)
        scaled_points, _ = self._pose_est_v4.get_points(10, 0, 0.03)
        self.assertTrue((points * 0.03 == scaled_points).all())

    def test_v4_read_from_cache(self) -> None:
        """
        test that we can open a V4 pose file from its cached h5 file

        Converting v4/v5 pose files to the format used by JABS is expensive
        enough that we don't want to have to do it every time a pose file is
        opened. The JABS UI will cache data after it's been structured in the
        way JABS expects as an h5 file. This tests that we can open the file
        after it's been cached and get the expected results.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir_path = Path(temp_dir)

            # this will be uncached, so it will read raw data from the pose file
            # and manipulate it to generate data in the form we need, and then
            # will write it back out to the cache directory
            pose_v4 = src.pose_estimation.open_pose_file(
                self._tmpdir_path / 'sample_pose_est_v4.h5',
                cache_dir=cache_dir_path)

            # open it again, this time it should be read from the cached
            # file
            pose_v4_from_cache = src.pose_estimation.open_pose_file(
                self._tmpdir_path / 'sample_pose_est_v4.h5',
                cache_dir=cache_dir_path)


        # make sure the list of identities is the same
        self.assertListEqual(pose_v4.identities,
                             pose_v4_from_cache.identities)

        # they should have the same number of frames
        self.assertEqual(pose_v4.num_frames,
                         pose_v4_from_cache.num_frames)

        # make sure the points and point masks are equal for all identities
        for ident in pose_v4.identities:
            poses, mask = pose_v4.get_identity_poses(ident)
            poses_cached, mask_cached = pose_v4_from_cache.get_identity_poses(ident)
            self.assertTrue(np.alltrue(poses == poses_cached))
            self.assertTrue(np.alltrue(mask == mask_cached))
