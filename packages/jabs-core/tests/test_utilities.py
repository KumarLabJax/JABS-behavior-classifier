"""Unit tests for jabs.core.utils.utilities module."""

import os
from pathlib import Path

import pytest

from jabs.core.utils import copy_file_atomic, pose_file_stem


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


def test_copy_file_atomic_replaces_existing_file(tmp_path: Path) -> None:
    """The destination's contents should be overwritten by the source's contents."""
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("new contents")
    destination.write_text("old contents")

    copy_file_atomic(source, destination)

    assert destination.read_text() == "new contents"


def test_copy_file_atomic_creates_missing_parent_dirs(tmp_path: Path) -> None:
    """Missing parent directories of the destination should be created."""
    source = tmp_path / "source.txt"
    destination = tmp_path / "nested" / "dir" / "destination.txt"
    source.write_text("payload")

    copy_file_atomic(source, destination)

    assert destination.read_text() == "payload"


def test_copy_file_atomic_does_not_leave_temp_file_on_success(tmp_path: Path) -> None:
    """A successful copy must leave no ``.tmp`` sibling behind."""
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.json"
    source.write_text("payload")

    copy_file_atomic(source, destination)

    assert not (tmp_path / "destination.json.tmp").exists()
    assert set(tmp_path.iterdir()) == {source, destination}


def test_copy_file_atomic_preserves_mtime(tmp_path: Path) -> None:
    """``shutil.copy2`` semantics: file metadata such as mtime should be preserved."""
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("payload")
    original_mtime = 1_700_000_000.0
    os.utime(source, (original_mtime, original_mtime))

    copy_file_atomic(source, destination)

    assert destination.stat().st_mtime == pytest.approx(original_mtime, abs=1.0)


def test_copy_file_atomic_handles_file_without_extension(tmp_path: Path) -> None:
    """A destination with no suffix should still receive an atomic replacement."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("payload")

    copy_file_atomic(source, destination)

    assert destination.read_text() == "payload"
    assert not (tmp_path / "destination.tmp").exists()
