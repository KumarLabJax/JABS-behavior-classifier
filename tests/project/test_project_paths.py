"""Tests for jabs.project.project_paths.ProjectPaths."""

import shutil
from pathlib import Path

import pytest

from jabs.project.project_paths import ProjectPaths


def test_create_directories_ignores_dotfile_videos_and_poses(tmp_path):
    """AppleDouble '._*' sidecars must not make a directory look like a JABS project.

    macOS creates a ``._<name>`` sidecar for every file on non-APFS/HFS+ volumes
    (e.g. exFAT external drives). Those match ``pathlib`` globs but are not real
    videos or valid pose HDF5 files, so a directory containing only sidecars is
    not a valid project.
    """
    (tmp_path / "._vid.mp4").touch()
    (tmp_path / "._vid_pose_est_v8.h5").touch()

    paths = ProjectPaths(base_path=tmp_path)
    with pytest.raises(ValueError, match="does not appear to be a valid JABS project"):
        paths.create_directories(validate=True)


def test_create_directories_validates_real_project(tmp_path):
    """A directory with a real video and pose file passes validation even alongside sidecars."""
    (tmp_path / "vid.mp4").touch()
    data_dir = Path(__file__).parent.parent / "data"
    shutil.copy(data_dir / "sample_pose_est_v6.h5", tmp_path / "vid_pose_est_v6.h5")
    # AppleDouble sidecars sitting next to the real files must not interfere.
    (tmp_path / "._vid.mp4").touch()
    (tmp_path / "._vid_pose_est_v6.h5").touch()

    paths = ProjectPaths(base_path=tmp_path)
    paths.create_directories(validate=True)  # should not raise

    assert paths.jabs_dir.is_dir()
