"""Tests for jabs.core.utils.video_files."""

from pathlib import Path

from jabs.core.utils.video_files import VIDEO_FILE_EXTENSIONS, find_video_files


def test_find_video_files_returns_recognized_extensions(tmp_path: Path) -> None:
    """Only the recognized video extensions are returned, and without their directory."""
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.avi").touch()
    (tmp_path / "a_pose_est_v6.h5").touch()
    (tmp_path / "notes.txt").touch()

    assert find_video_files(tmp_path) == ["a.mp4", "b.avi"]


def test_recognized_video_extensions() -> None:
    """The recognized extensions are the two JABS has always accepted."""
    assert sorted(VIDEO_FILE_EXTENSIONS) == [".avi", ".mp4"]


def test_find_video_files_is_sorted(tmp_path: Path) -> None:
    """Results come back in name order, whatever order the directory lists them in."""
    for name in ("c.mp4", "a.mp4", "b.avi"):
        (tmp_path / name).touch()

    assert find_video_files(tmp_path) == ["a.mp4", "b.avi", "c.mp4"]


def test_find_video_files_excludes_dotfiles(tmp_path: Path) -> None:
    """Dotfiles are skipped, including macOS AppleDouble ('._*') sidecars."""
    (tmp_path / "real1.mp4").touch()
    (tmp_path / "real2.avi").touch()
    # macOS AppleDouble sidecars written when copying to exFAT/NTFS volumes
    (tmp_path / "._real1.mp4").touch()
    (tmp_path / "._real2.avi").touch()
    # other hidden files that should never be treated as videos
    (tmp_path / ".DS_Store").touch()
    (tmp_path / ".hidden.mp4").touch()

    assert find_video_files(tmp_path) == ["real1.mp4", "real2.avi"]


def test_find_video_files_is_not_recursive(tmp_path: Path) -> None:
    """Only the directory itself is scanned; videos in subdirectories are not found."""
    (tmp_path / "top.mp4").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "inner.mp4").touch()

    assert find_video_files(tmp_path) == ["top.mp4"]


def test_find_video_files_empty_directory(tmp_path: Path) -> None:
    """A directory with no videos yields an empty list rather than raising."""
    assert find_video_files(tmp_path) == []
