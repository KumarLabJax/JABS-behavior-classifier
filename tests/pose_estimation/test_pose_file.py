import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

import jabs.pose_estimation
from jabs.pose_estimation import PoseOpenError

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
def pose_est_v4(tmpdir_with_pose_files):
    """The sample v4 file, opened with open_pose_file (as Pose: legacy files are converted)."""
    return jabs.pose_estimation.open_pose_file(tmpdir_with_pose_files / "sample_pose_est_v4.h5")


@pytest.fixture(scope="module")
def pose_est_v5(tmpdir_with_pose_files):
    """The sample v5 file, opened with open_pose_file (as Pose: legacy files are converted)."""
    return jabs.pose_estimation.open_pose_file(tmpdir_with_pose_files / "sample_pose_est_v5.h5")


def test_v5_static_corners(pose_est_v5) -> None:
    """the sample v5 file's corners come back as (4, 2) (spec §2 row "Static objects")"""
    static_objs = pose_est_v5.static_objects
    assert "corners" in static_objs
    assert static_objs["corners"].shape == (4, 2)


@pytest.mark.parametrize(
    "filename",
    [
        "sample.h5",
        "sample_pose_est.h5",
        "samplev3.h5",
        "sample_pose_est_v6.h5.bak",
        "sample_pose_est_v1.h5",
        "sample_pose_est_v99.h5",
        "._sample_pose_est_v6.h5",
    ],
    ids=[
        "no-suffix",
        "no-version",
        "no-underscore",
        "trailing-extension",
        "version-too-old",
        "version-too-new",
        "dotfile",
    ],
)
def test_open_pose_file_rejects_non_pose_file_names(filename: str) -> None:
    """open_pose_file refuses a name that is neither a unified nor a supported legacy name

    The name is checked before the file is read, so the file need not exist.
    """
    with pytest.raises(PoseOpenError, match="not a pose file name"):
        jabs.pose_estimation.open_pose_file(Path(filename))


def test_get_points(pose_est_v4):
    """test getting pose points from a Pose"""
    points, point_mask = pose_est_v4.get_identity_poses(0)
    nframes = pose_est_v4.num_frames

    assert points.shape == (nframes, 12, 2)
    assert point_mask.shape == (nframes, 12)


def test_get_points_single_frame(pose_est_v4):
    """test getting pose points for single frame from a Pose"""
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


def _corrupt_v4_embed_id(tmpdir_with_pose_files: Path) -> tuple[Path, int, int]:
    """Copy the sample v4 file with one unmasked instance_embed_id set to 0.

    Returns:
        The corrupt copy, the corrupted frame, and the identity the instance had.
    """
    corrupt_path = tmpdir_with_pose_files / "corrupt_pose_est_v4.h5"
    shutil.copy(tmpdir_with_pose_files / "sample_pose_est_v4.h5", corrupt_path)

    with h5py.File(corrupt_path, "r+") as f:
        id_mask = f["poseest/id_mask"][:]
        instance_embed_id = f["poseest/instance_embed_id"][:]
        # find an unmasked entry and corrupt it to 0
        frame, instance = np.argwhere(id_mask == 0)[0]
        identity = int(instance_embed_id[frame, instance]) - 1
        instance_embed_id[frame, instance] = 0
        f["poseest/instance_embed_id"][:] = instance_embed_id
    return corrupt_path, int(frame), identity


def test_v4_zero_instance_embed_id_is_a_tail_slot(tmpdir_with_pose_files, tmp_path) -> None:
    """an unmasked instance_embed_id of 0 no longer refuses the file

    The converter places such an instance in a tail slot, and tail slots never
    become JABS identities (spec §2 row "Slots iterated"); the legacy v4 reader
    raised PoseIdEmbeddingException instead.
    """
    corrupt_path, frame, identity = _corrupt_v4_embed_id(tmpdir_with_pose_files)
    pose = jabs.pose_estimation.open_pose_file(corrupt_path, tmp_path)
    clean = jabs.pose_estimation.open_pose_file(
        tmpdir_with_pose_files / "sample_pose_est_v4.h5", tmp_path
    )
    assert pose.identities == clean.identities
    assert pose.num_frames == clean.num_frames
    # the instance left its identity on that frame and went nowhere JABS iterates
    assert clean.identity_mask(identity)[frame]
    assert not pose.identity_mask(identity)[frame]
    for other in pose.identities:
        if other != identity:
            np.testing.assert_array_equal(pose.identity_mask(other), clean.identity_mask(other))


def test_v4_read_from_cache(tmpdir_with_pose_files):
    """a v4 file opened a second time reads its cached conversion and gets the same data

    Converting a legacy pose file to the unified format is expensive enough that
    it is done once: the conversion is cached in the cache directory and reused
    by later opens of the same, unchanged file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir_path = Path(temp_dir)

        # uncached: converts the legacy file into the cache directory
        pose_v4 = jabs.pose_estimation.open_pose_file(
            tmpdir_with_pose_files / "sample_pose_est_v4.h5", cache_dir=cache_dir_path
        )

        # open it again: the cached conversion is reused
        pose_v4_from_cache = jabs.pose_estimation.open_pose_file(
            tmpdir_with_pose_files / "sample_pose_est_v4.h5", cache_dir=cache_dir_path
        )
        assert len(list((cache_dir_path / "converted").glob("*.h5"))) == 1

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


@pytest.mark.parametrize(
    ("lixit_shape", "expected"),
    [
        ((2, 3, 2), 3),
        ((2, 2), 1),
        ((0, 3, 2), 0),
        ((0, 2), 0),
    ],
    ids=["three-keypoint", "one-keypoint", "empty-three-keypoint", "empty-one-keypoint"],
)
def test_scan_lixit_keypoints(tmp_path: Path, lixit_shape: tuple[int, ...], expected: int) -> None:
    """An empty lixit dataset counts as no lixit, as it does when the file is opened.

    Args:
        tmp_path: Temporary directory to write the pose file copy into.
        lixit_shape: Shape of the ``static_objects/lixit`` dataset to store.
        expected: Keypoint count the scan should report.
    """
    pose_path = tmp_path / "vid_pose_est_v5.h5"
    shutil.copy(Path(__file__).parent.parent / "data" / "sample_pose_est_v5.h5", pose_path)
    with h5py.File(pose_path, "r+") as pose_h5:
        pose_h5.create_dataset("static_objects/lixit", data=np.zeros(lixit_shape, dtype=np.uint16))

    assert jabs.pose_estimation.scan_pose_metadata(pose_path).lixit_keypoints == expected


def test_scan_lixit_keypoints_without_a_lixit(tmpdir_with_pose_files) -> None:
    """A pose file with no lixit static object reports zero keypoints."""
    pose_path = tmpdir_with_pose_files / "sample_pose_est_v5.h5"
    assert jabs.pose_estimation.scan_pose_metadata(pose_path).lixit_keypoints == 0
