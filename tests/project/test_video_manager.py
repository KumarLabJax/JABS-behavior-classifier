import gzip
import json
import shutil
from pathlib import Path

import pytest
from src.jabs.project.video_manager import VideoManager
from src.jabs.project.project_paths import ProjectPaths
from src.jabs.project.settings_manager import SettingsManager


@pytest.fixture
def project_paths(tmp_path):
    """Fixture to create a ProjectPaths instance."""
    paths = ProjectPaths(base_path=tmp_path)
    paths.create_directories()
    return paths


@pytest.fixture
def settings_manager(project_paths):
    """Fixture to create a SettingsManager instance."""
    # Create a project.json file with a "video_files" field
    project_json_path = project_paths.project_dir / "project.json"
    video_files = {
        "video1.avi": {"identities": 3},
        "video2.mp4": {"identities": 5},
    }
    with project_json_path.open("w") as f:
        json.dump({"video_files": video_files}, f)

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
    pose1_src = data_dir / "sample_pose_est_v3.h5.gz"
    pose2_src = data_dir / "sample_pose_est_v6.h5.gz"
    pose1_dst = project_paths.project_dir / "video1_pose_est_v3.h5"
    pose2_dst = project_paths.project_dir / "video2_pose_est_v6.h5"

    # Decompress and copy gzipped pose files
    with gzip.open(pose1_src, "rb") as f_in, open(pose1_dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    with gzip.open(pose2_src, "rb") as f_in, open(pose2_dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return VideoManager(project_paths, settings_manager, enable_video_check=False)


def test_get_videos(video_manager, project_paths):
    """Test retrieving video files from the project directory."""
    videos = video_manager.get_videos(project_paths.project_dir)
    assert "video1.avi" in videos
    assert "video2.mp4" in videos
    assert len(videos) == 2


def test_check_video_name(video_manager):
    """Test checking if a video name is valid."""
    video_manager.check_video_name("video1.avi")  # Should not raise an exception
    with pytest.raises(ValueError, match="not in project"):
        video_manager.check_video_name("nonexistent_video.avi")


def test_load_video_labels(video_manager, project_paths):
    """Test loading video labels."""

    # Create a dummy annotation file
    annotation_file = project_paths.annotations_dir / "video1.json"
    annotation_file.write_text(
        '{"labels": {}, "num_frames": 1000, "file": "video1.avi"}'
    )

    labels = video_manager.load_video_labels("video1.avi")
    assert labels is not None
    assert labels.filename == "video1.avi"
