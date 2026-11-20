"""Pose file discovery: which file belongs to a video, and how it is opened."""

import shutil
from pathlib import Path

import pytest

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.pose_estimation import get_pose_path, open_pose_file
from jabs.pose_estimation.discovery import (
    POSE_FILE_GLOBS,
    find_pose_file,
    is_pose_file,
)
from jabs.pose_estimation.pose import Pose

from .conftest import build_pose_file
from .contract import sample_path


def _touch(directory: Path, *names: str) -> None:
    """Create empty files named ``names`` in ``directory``."""
    for name in names:
        (directory / name).touch()


def test_pose_file_globs() -> None:
    """The two accepted name patterns, unified first."""
    assert POSE_FILE_GLOBS == ("*_pose.h5", "*_pose_est_v*.h5")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("vid_pose.h5", True),
        ("vid_pose_est_v2.h5", True),
        ("vid_pose_est_v8.h5", True),
        ("vid_pose_est_v9.h5", False),
        ("vid_pose_est_vN.h5", False),
        ("vid_pose_est.h5", False),
        ("vid.h5", False),
        ("vid_pose.mp4", False),
        ("vid_pose.h5.bak", False),
        ("._vid_pose.h5", False),
        ("._vid_pose_est_v6.h5", False),
        (".vid_pose.h5", False),
    ],
)
def test_is_pose_file(name: str, expected: bool) -> None:
    """Unified and v2..v8 legacy names are pose files; dotfiles never are."""
    assert is_pose_file(Path("/data") / name) is expected


def test_unified_beats_legacy(tmp_path: Path) -> None:
    """When a video has both, the unified file wins."""
    _touch(tmp_path, "vid_pose_est_v8.h5", "vid_pose.h5", "vid_pose_est_v6.h5")
    assert find_pose_file(tmp_path / "vid.mp4") == tmp_path / "vid_pose.h5"


def test_highest_legacy_version_wins(tmp_path: Path) -> None:
    """Among legacy files the highest version wins."""
    _touch(tmp_path, "vid_pose_est_v2.h5", "vid_pose_est_v6.h5", "vid_pose_est_v4.h5")
    assert find_pose_file(tmp_path / "vid.mp4") == tmp_path / "vid_pose_est_v6.h5"


def test_unsupported_legacy_version_is_not_found(tmp_path: Path) -> None:
    """A legacy version outside 2..8 is not a pose file for the video."""
    _touch(tmp_path, "vid_pose_est_v9.h5")
    assert find_pose_file(tmp_path / "vid.mp4") is None


def test_no_pose_file(tmp_path: Path) -> None:
    """A video without a pose file gives None; get_pose_path raises ValueError."""
    _touch(tmp_path, "other_pose.h5", "vid2_pose.h5")
    assert find_pose_file(tmp_path / "vid.mp4") is None
    with pytest.raises(ValueError):
        get_pose_path(tmp_path / "vid.mp4")


def test_dotfiles_are_skipped(tmp_path: Path) -> None:
    """AppleDouble files (``._*``) are never pose files, even for a dot-named video."""
    _touch(tmp_path, "._vid_pose.h5", "._vid_pose_est_v6.h5")
    assert find_pose_file(tmp_path / "._vid.mp4") is None


def test_pose_dir(tmp_path: Path) -> None:
    """pose_dir is searched instead of the video's directory."""
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    _touch(tmp_path, "vid_pose_est_v6.h5")
    _touch(pose_dir, "vid_pose.h5")
    assert find_pose_file(tmp_path / "vid.mp4", pose_dir) == pose_dir / "vid_pose.h5"
    assert get_pose_path(tmp_path / "vid.mp4", pose_dir) == pose_dir / "vid_pose.h5"
    assert get_pose_path(tmp_path / "vid.mp4") == tmp_path / "vid_pose_est_v6.h5"


def test_open_pose_file_unified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``*_pose.h5`` file opens as Pose."""
    monkeypatch.delenv("JABS_FORCE_UNIFIED", raising=False)
    path = build_pose_file(tmp_path / "vid_pose.h5")
    with open_pose_file(path, tmp_path / "cache") as pose:
        assert isinstance(pose, Pose)
        assert pose.pose_file == path


def test_open_pose_file_legacy_uses_legacy_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy file still opens with its legacy reader by default."""
    monkeypatch.delenv("JABS_FORCE_UNIFIED", raising=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    pose = open_pose_file(sample_path(4), cache)
    assert isinstance(pose, PoseEstimation)
    assert not (cache / "converted").exists()


def test_force_unified_routes_legacy_through_pose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JABS_FORCE_UNIFIED=1 converts a legacy file and opens the conversion."""
    monkeypatch.setenv("JABS_FORCE_UNIFIED", "1")
    source = tmp_path / "vid_pose_est_v4.h5"
    shutil.copyfile(sample_path(4), source)
    cache = tmp_path / "cache"
    with open_pose_file(source, cache) as pose:
        assert isinstance(pose, Pose)
        assert pose.pose_file == source
        assert pose.source_format == "pose_est_v4"
    assert len(list((cache / "converted").glob("vid.*.h5"))) == 1
