import gzip
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import src.jabs.pose_estimation


class TestFeatureBase(unittest.TestCase):
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

        cls._pose_est_v5 = src.jabs.pose_estimation.open_pose_file(
            cls._tmpdir_path / 'sample_pose_est_v5.h5')

        # V5 pose file in the data directory does not currently have "lixit"
        # as one of its static objects, so we'll manually add it
        cls._pose_est_v5.static_objects['lixit'] = np.asarray([[62, 166]],
                                                              dtype=np.uint16)

        # V5 pose file also doesn't have the food hopper static object
        cls._pose_est_v5.static_objects['food_hopper'] = np.asarray(
            [[7, 291], [7, 528], [44, 296], [44, 518]]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()
