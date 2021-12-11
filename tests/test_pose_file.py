import gzip
import shutil
import tempfile
import unittest
from pathlib import Path

import src.pose_estimation


class TestOpenPose(unittest.TestCase):
    _tmpdir = None
    _test_data_dir = Path(__file__).parent / 'data'

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._tmpdir_path = Path(cls._tmpdir.name)

        # decompress v3 pose file in tempdir
        with gzip.open(cls._test_data_dir / 'sample_pose_est_v3.h5.gz',
                       'rb') as f_in:
            with open(cls._tmpdir_path / 'sample_pose_est_v3.h5',
                      'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # decompress v4 pose file in tempdir
        with gzip.open(cls._test_data_dir / 'sample_pose_est_v4.h5.gz',
                       'rb') as f_in:
            with open(cls._tmpdir_path / 'sample_pose_est_v4.h5',
                      'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        cls._pose_est_v3 = src.pose_estimation.open_pose_file(
            cls._tmpdir_path / 'sample_pose_est_v3.h5')

        cls._pose_est_v4 = src.pose_estimation.open_pose_file(
            cls._tmpdir_path / 'sample_pose_est_v4.h5')

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_open_pose_est_v3(self) -> None:
        """
        test that open_pose_file can open a V3 pose file and it returns the
        correct type
        """
        self.assertIsInstance(self._pose_est_v3,
                              src.pose_estimation.PoseEstimationV3)
        self.assertEqual(self._pose_est_v3.format_major_version, 3)

    def test_open_pose_est_v4(self) -> None:
        """
        test that open_pose_file can open a V4 pose file and it returns the
        correct type
        """
        self.assertIsInstance(self._pose_est_v4,
                              src.pose_estimation.PoseEstimationV4)
        self.assertEqual(self._pose_est_v4.format_major_version, 4)

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

    def test_scaling_points(self) -> None:
        """ test scaling points """
        points, _ = self._pose_est_v4.get_points(10, 0)
        scaled_points, _ = self._pose_est_v4.get_points(10, 0, 0.03)
        self.assertTrue((points * 0.03 == scaled_points).all())
