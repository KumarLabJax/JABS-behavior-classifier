"""Shared fixtures for feature tests."""

import shutil
import tempfile
from pathlib import Path

import pytest

import jabs.pose_estimation as pose_est
from jabs.feature_extraction.segmentation_features import SegmentationFeatureGroup


@pytest.fixture(scope="module")
def seg_data():
    """Common setup for segmentation tests.

    Provides pose estimation v6 data with segmentation features initialized.

    Returns:
        dict: Dictionary with 'pose_est_v6', 'moment_cache', and 'feature_mods'.
    """
    pixel_scale = 1.0
    data_path = Path(__file__).parent.parent.parent / "data"
    data_filename = "sample_pose_est_v6.h5"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        shutil.copy(data_path / data_filename, tmpdir_path / data_filename)

        pose_est_v6 = pose_est.open_pose_file(tmpdir_path / data_filename)

        moment_cache = SegmentationFeatureGroup(pose_est_v6, pixel_scale)
        feature_mods = moment_cache._init_feature_mods(1)

        yield {
            "pose_est_v6": pose_est_v6,
            "moment_cache": moment_cache,
            "feature_mods": feature_mods,
        }
