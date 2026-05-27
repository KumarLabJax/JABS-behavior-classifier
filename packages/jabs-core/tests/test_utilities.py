"""Unit tests for jabs.core.utils.utilities module."""

from pathlib import Path

import pytest

from jabs.core.utils import pose_file_stem


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("video_pose_est_v6.h5", "video"),
        ("video_pose_est_v2.h5", "video"),
        ("video_pose_est_v12.h5", "video"),
        ("/some/dir/video_pose_est_v6.h5", "video"),
        ("video.mp4", "video"),
        ("video.avi", "video"),
        ("nested_name_pose_est_v8.h5", "nested_name"),
        ("no_suffix.h5", "no_suffix"),
        ("plain_name", "plain_name"),
    ],
    ids=[
        "pose-v6",
        "pose-v2",
        "pose-v12",
        "pose-with-dir",
        "video-mp4",
        "video-avi",
        "nested-underscore-name",
        "h5-no-pose-suffix",
        "no-extension",
    ],
)
def test_pose_file_stem(path: str, expected: str) -> None:
    """Pose-file suffix is stripped and other names are returned unchanged."""
    assert pose_file_stem(path) == expected


def test_pose_file_stem_accepts_path() -> None:
    """``pathlib.Path`` inputs are supported."""
    assert pose_file_stem(Path("/a/b/video_pose_est_v6.h5")) == "video"


def test_pose_file_stem_video_and_pose_match() -> None:
    """A video file and its matching pose file produce the same stem.

    This is what guarantees feature-cache directories are consistent whether the
    caller passes the pose file (jabs-classify, jabs-cli compute-features) or the
    video file (jabs-init, GUI).
    """
    assert pose_file_stem("video.mp4") == pose_file_stem("video_pose_est_v6.h5")
