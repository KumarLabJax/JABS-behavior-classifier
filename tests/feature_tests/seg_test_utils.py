"""
This file contains the base class SegTest.  This class can be inherited by tests written that include segmentation
data in the pose file (v6+).
"""

import h5py
import os
from pathlib import Path
import tempfile
import shutil
import gzip

from src.jabs.feature_extraction.segmentation_features import SegmentationFeatureGroup
import src.jabs.pose_estimation as pose_est


class SegDataBaseClass(object):
    """Common setup and teardown for segmentation tests."""

    pixel_scale = 1.0
    dataPath = Path(__file__).parent / "../data"
    dataFileName = "sample_pose_est_v6.h5.gz"
    _tmpdir = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._tmpdir_path = Path(cls._tmpdir.name)

        with gzip.open(cls.dataPath / cls.dataFileName, "rb") as f_in:
            with open(
                cls._tmpdir_path / cls.dataFileName.replace(".h5.gz", ".h5"), "wb"
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)

        cls._pose_est_v6 = pose_est.open_pose_file(
            cls._tmpdir_path / cls.dataFileName.replace(".h5.gz", ".h5")
        )

        cls._moment_cache = SegmentationFeatureGroup(cls._pose_est_v6, cls.pixel_scale)
        cls.feature_mods = cls._moment_cache._init_feature_mods(1)

    @classmethod
    def tearDown(cls):
        if cls._tmpdir:
            cls._tmpdir.cleanup()


def setUpModule():
    """Use if code should be executed once for all tests."""
    pass
