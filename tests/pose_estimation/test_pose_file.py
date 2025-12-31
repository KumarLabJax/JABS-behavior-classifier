import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import jabs.pose_estimation

_TEST_FILES = [
    "sample_pose_est_v3.h5",
    "sample_pose_est_v4.h5",
    "sample_pose_est_v5.h5",
]


@pytest.fixture(scope="module")
def tmpdir_with_pose_files():
    """Setup temporary directory and copy pose files to it."""
    test_data_dir = Path(__file__).parent.parent / "data"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # copy pose files to tempdir
        for f in _TEST_FILES:
            shutil.copy(test_data_dir / f, tmpdir_path / f)

        yield tmpdir_path


@pytest.fixture(scope="module")
def pose_est_v3(tmpdir_with_pose_files):
    """Fixture for PoseEstimationV3 object."""
    return jabs.pose_estimation.open_pose_file(tmpdir_with_pose_files / "sample_pose_est_v3.h5")


@pytest.fixture(scope="module")
def pose_est_v4(tmpdir_with_pose_files):
    """Fixture for PoseEstimationV4 object."""
    return jabs.pose_estimation.open_pose_file(tmpdir_with_pose_files / "sample_pose_est_v4.h5")


@pytest.fixture(scope="module")
def pose_est_v5(tmpdir_with_pose_files):
    """Fixture for PoseEstimationV5 object."""
    return jabs.pose_estimation.open_pose_file(tmpdir_with_pose_files / "sample_pose_est_v5.h5")


def test_open_pose_est_v3(pose_est_v3):
    """test that open_pose_file can open a V3 pose file"""
    assert isinstance(pose_est_v3, jabs.pose_estimation.PoseEstimationV3)
    assert pose_est_v3.format_major_version == 3


def test_open_pose_est_v4(pose_est_v4):
    """test that open_pose_file can open a V4 pose file"""
    assert isinstance(pose_est_v4, jabs.pose_estimation.PoseEstimationV4)
    assert pose_est_v4.format_major_version == 4


def test_open_pose_est_v5(pose_est_v5):
    """test that open_pose_file can open a V5 pose file"""
    assert isinstance(pose_est_v5, jabs.pose_estimation.PoseEstimationV5)
    assert pose_est_v5.format_major_version == 5

    # the test v5 pose file has 'corners' in static objects dataset
    static_objs = pose_est_v5.static_objects
    assert "corners" in static_objs
    assert static_objs["corners"].shape == (4, 2)


def test_get_points(pose_est_v4):
    """test getting pose points from PoseEstimation instance"""
    points, point_mask = pose_est_v4.get_identity_poses(0)
    nframes = pose_est_v4.num_frames

    assert points.shape == (nframes, 12, 2)
    assert point_mask.shape == (nframes, 12)


def test_get_points_single_frame(pose_est_v4):
    """test getting pose points for single frame from PoseEstimation instance"""
    # get points for the 10th frame for identity 0
    points, point_mask = pose_est_v4.get_points(10, 0)

    assert points.shape == (12, 2)
    assert point_mask.shape == (12,)

    # compare to getting all points
    points_all_frame, point_mask_all_frames = pose_est_v4.get_identity_poses(0)
    # May contain NaNs which assert_equal handles
    np.testing.assert_equal(points, points_all_frame[10, :])
    assert (point_mask == point_mask_all_frames[10, :]).all()


def test_get_points_out_of_range(pose_est_v4):
    """test that get_points raises IndexError when frame index is out of range"""
    with pytest.raises(IndexError):
        _, _ = pose_est_v4.get_points(1000000, 0)


def test_scaling_points(pose_est_v4):
    """test scaling points"""
    points, _ = pose_est_v4.get_points(10, 0)
    scaled_points, _ = pose_est_v4.get_points(10, 0, 0.03)
    np.testing.assert_equal(points * 0.03, scaled_points)


def test_v4_read_from_cache(tmpdir_with_pose_files):
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
        pose_v4 = jabs.pose_estimation.open_pose_file(
            tmpdir_with_pose_files / "sample_pose_est_v4.h5", cache_dir=cache_dir_path
        )

        # open it again, this time it should be read from the cached
        # file
        pose_v4_from_cache = jabs.pose_estimation.open_pose_file(
            tmpdir_with_pose_files / "sample_pose_est_v4.h5", cache_dir=cache_dir_path
        )

    # make sure the list of identities is the same
    assert pose_v4.identities == pose_v4_from_cache.identities

    # they should have the same number of frames
    assert pose_v4.num_frames == pose_v4_from_cache.num_frames

    # make sure the points and point masks are equal for all identities
    # nans need to be handled differently
    for ident in pose_v4.identities:
        poses, mask = pose_v4.get_identity_poses(ident)
        poses[np.isnan(poses)] = 0
        poses_cached, mask_cached = pose_v4_from_cache.get_identity_poses(ident)
        poses_cached[np.isnan(poses_cached)] = 0
        assert np.all(poses == poses_cached)
        assert np.all(mask == mask_cached)
