import gzip
import shutil
import tempfile
import unittest
from pathlib import Path

import src.pose_estimation
import src.feature_extraction.landmark_features.corner as corner_module


class TestCornerFeatures(unittest.TestCase):
    _tmpdir = None
    _test_file = Path(__file__).parent.parent / 'data' / 'sample_pose_est_v5.h5.gz'

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._tmpdir_path = Path(cls._tmpdir.name)

        pose_path = cls._tmpdir_path / 'sample_pose_est_v5.h5'

        # decompress pose file into tempdir
        with gzip.open(cls._test_file, 'rb') as f_in:
            with open(pose_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        cls._pose_est_v5 = src.pose_estimation.open_pose_file(
            cls._tmpdir_path / 'sample_pose_est_v5.h5')

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_compute_corner_distances(self):
        pixel_scale = self._pose_est_v5.cm_per_pixel
        dist = corner_module.CornerDistanceInfo(self._pose_est_v5, pixel_scale)
        dist_to_corner = corner_module.DistanceToCorner(self._pose_est_v5,
                                                        pixel_scale, dist)
        bearing_to_corner = corner_module.BearingToCorner(self._pose_est_v5,
                                                          pixel_scale, dist)

        # check dimensions of per frame feature values
        for i in range(self._pose_est_v5.num_identities):
            dist_per_frame = dist_to_corner.per_frame(i)
            self.assertEqual(dist_per_frame.shape,
                             (self._pose_est_v5.num_frames,))

            bearing_per_frame = bearing_to_corner.per_frame(i)
            self.assertEqual(bearing_per_frame.shape,
                             (self._pose_est_v5.num_frames,))

            # check dimensions of window feature values
            dist_window_values = dist_to_corner.window(i, 5, dist_per_frame)
            for op in dist_window_values:
                self.assertEqual(dist_window_values[op].shape,
                                 (self._pose_est_v5.num_frames,))

            bearing_window_values = bearing_to_corner.window(i, 5,
                                                             bearing_per_frame)
            for op in bearing_window_values:
                self.assertEqual(bearing_window_values[op].shape,
                                 (self._pose_est_v5.num_frames,))

        # check range of bearings, should be in the range [180, -180)
        for i in range(self._pose_est_v5.num_identities):
            values = bearing_to_corner.per_frame(i)
            self.assertTrue(((values <= 180) & (values > -180)).all())

        # check distances are >= 0
        for i in range(self._pose_est_v5.num_identities):
            values = dist_to_corner.per_frame(i)
            self.assertTrue((values >= 0).all())
