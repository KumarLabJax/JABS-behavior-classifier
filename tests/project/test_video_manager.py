import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from jabs.project.project_paths import ProjectPaths
from jabs.project.settings_manager import SettingsManager
from jabs.project.video_manager import VideoManager


@pytest.fixture
def project_paths(tmp_path):
    """Fixture to create a ProjectPaths instance."""
    paths = ProjectPaths(base_path=tmp_path)
    paths.create_directories(validate=False)
    return paths


@pytest.fixture
def settings_manager(project_paths):
    """Fixture to create a SettingsManager instance."""
    return SettingsManager(project_paths)


@pytest.fixture
def video_manager(project_paths, settings_manager):
    """Fixture to create a VideoManager instance."""
    # Create dummy video files
    video1 = project_paths.project_dir / "video1.avi"
    video2 = project_paths.project_dir / "video2.mp4"
    video1.touch()
    video2.touch()

    # Set data_dir relative to the current file
    data_dir = Path(__file__).parent.parent / "data"
    pose1_src = data_dir / "sample_pose_est_v3.h5"
    pose2_src = data_dir / "sample_pose_est_v6.h5"
    pose1_dst = project_paths.project_dir / "video1_pose_est_v3.h5"
    pose2_dst = project_paths.project_dir / "video2_pose_est_v6.h5"

    # Copy pose files
    shutil.copy(pose1_src, pose1_dst)
    shutil.copy(pose2_src, pose2_dst)

    scan_results = {
        "video1.avi": {
            "video": "video1.avi",
            "hdf5_frame_count": 100,
            "video_frame_count": None,
            "identity_count": 3,
            "static_objects": [],
            "lixit_keypoints": 0,
            "has_cm_per_pixel": False,
        },
        "video2.mp4": {
            "video": "video2.mp4",
            "hdf5_frame_count": 100,
            "video_frame_count": None,
            "identity_count": 5,
            "static_objects": [],
            "lixit_keypoints": 0,
            "has_cm_per_pixel": False,
        },
    }
    return VideoManager(
        project_paths, settings_manager, enable_video_check=False, scan_results=scan_results
    )


def test_get_videos(video_manager, project_paths):
    """Test retrieving video files from the project directory."""
    videos = video_manager.get_videos(project_paths.project_dir)
    assert "video1.avi" in videos
    assert "video2.mp4" in videos
    assert len(videos) == 2


def test_get_videos_excludes_dotfiles(tmp_path):
    """get_videos ignores dotfiles, including macOS AppleDouble ('._*') sidecars."""
    (tmp_path / "real1.mp4").touch()
    (tmp_path / "real2.avi").touch()
    # macOS AppleDouble sidecars written when copying to exFAT/NTFS volumes
    (tmp_path / "._real1.mp4").touch()
    (tmp_path / "._real2.avi").touch()
    # other hidden files that should never be treated as videos
    (tmp_path / ".DS_Store").touch()
    (tmp_path / ".hidden.mp4").touch()

    videos = VideoManager.get_videos(tmp_path)

    assert sorted(videos) == ["real1.mp4", "real2.avi"]


def test_check_video_name(video_manager):
    """Test checking if a video name is valid."""
    video_manager.check_video_name("video1.avi")  # Should not raise an exception
    with pytest.raises(ValueError, match="not in project"):
        video_manager.check_video_name("nonexistent_video.avi")


def test_load_video_labels(video_manager, project_paths):
    """Test loading video labels."""
    # Create a dummy annotation file
    annotation_file = project_paths.annotations_dir / "video1.json"
    annotation_file.write_text('{"labels": {}, "num_frames": 1000, "file": "video1.avi"}')

    # Create a mock pose_est object
    mock_pose_est = MagicMock()
    mock_pose_est.identity_mask.return_value = np.full(1000, True, dtype=bool)
    mock_pose_est.num_frames = 1000

    labels = video_manager.load_video_labels("video1.avi")
    assert labels is not None
    assert labels.filename == "video1.avi"


def test_video_manager_uses_custom_video_and_pose_dirs(tmp_path):
    """VideoManager should enumerate videos and resolve pose from the configured dirs."""
    project_root = tmp_path / "project"
    video_dir = tmp_path / "videos"
    pose_dir = tmp_path / "poses"
    project_root.mkdir()
    video_dir.mkdir()
    pose_dir.mkdir()

    paths = ProjectPaths(base_path=project_root, video_dir=video_dir, pose_dir=pose_dir)
    paths.create_directories(validate=False)

    (video_dir / "video1.avi").touch()

    data_dir = Path(__file__).parent.parent / "data"
    shutil.copy(data_dir / "sample_pose_est_v6.h5", pose_dir / "video1_pose_est_v6.h5")

    scan_results = {
        "video1.avi": {
            "video": "video1.avi",
            "hdf5_frame_count": 100,
            "video_frame_count": None,
            "identity_count": 2,
            "static_objects": [],
            "lixit_keypoints": 0,
            "has_cm_per_pixel": False,
        },
    }
    manager = VideoManager(
        paths, SettingsManager(paths), enable_video_check=False, scan_results=scan_results
    )

    assert manager.videos == ["video1.avi"]
    assert manager.video_path("video1.avi") == video_dir / "video1.avi"
    assert manager.get_cached_pose_path("video1.avi") == pose_dir / "video1_pose_est_v6.h5"
