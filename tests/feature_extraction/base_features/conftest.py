"""Shared fixtures for base feature tests."""

import shutil
import tempfile
from pathlib import Path

import pytest

import jabs.pose_estimation as pose_est


@pytest.fixture(scope="module")
def pose_est_v5():
    """Fixture to create a pose estimation v5 object.

    Creates a temporary directory with a copy of the pose file.

    Yields:
        PoseEstimationV5: Pose estimation object.
    """
    test_file = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v5.h5"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        pose_path = tmpdir_path / "sample_pose_est_v5.h5"

        # Copy pose file to tempdir
        shutil.copy(test_file, pose_path)

        pose_est_v5 = pose_est.open_pose_file(pose_path)

        yield pose_est_v5
