import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from jabs.io.annotations import UNVERSIONED, AnnotationDocument, AnnotationStore
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


def test_load_video_labels_reads_through_the_annotation_store(video_manager):
    """Labels come from the store, not from a direct read of the annotations dir."""
    document = {"labels": {}, "num_frames": 1000, "file": "video1.avi"}
    store = MagicMock(spec=AnnotationStore)
    store.load_document.return_value = AnnotationDocument(document, UNVERSIONED)
    video_manager._annotation_store = store

    labels = video_manager.load_video_labels("video1.avi", pose=MagicMock())

    store.load_document.assert_called_once_with("video1.avi")
    assert labels is not None
    assert labels.filename == "video1.avi"


def test_load_video_labels_returns_none_when_the_store_has_no_document(video_manager):
    """An unlabeled video yields no VideoLabels, and no pose file is opened."""
    store = MagicMock(spec=AnnotationStore)
    store.load_document.return_value = None
    video_manager._annotation_store = store

    assert video_manager.load_video_labels("video1.avi") is None


def test_load_annotations_reads_through_the_annotation_store(video_manager):
    """The raw-document read path is served by the store as well."""
    document = {"labels": {}, "num_frames": 1000, "file": "video1.avi"}
    store = MagicMock(spec=AnnotationStore)
    store.load_document.return_value = AnnotationDocument(document, UNVERSIONED)
    video_manager._annotation_store = store

    assert video_manager.load_annotations("video1.avi") == document
    store.load_document.assert_called_once_with("video1.avi")


def test_load_annotations_rejects_a_video_outside_the_project(video_manager):
    """An unknown video is still rejected before the store is consulted."""
    store = MagicMock(spec=AnnotationStore)
    video_manager._annotation_store = store

    with pytest.raises(ValueError, match="not in project"):
        video_manager.load_annotations("not_in_project.avi")
    store.load_document.assert_not_called()


def test_annotations_path_comes_from_the_annotation_store(video_manager, project_paths):
    """The advertised annotation path is whatever the store reports."""
    assert video_manager.annotations_path("video1.avi") == (
        project_paths.annotations_dir / "video1.json"
    )


def test_remove_video_updates_derived_state(video_manager):
    """Removing a video drops all per-video state derived from the project scan."""
    assert video_manager.total_project_identities == 8
    # populate the pose path cache so we can assert it is invalidated
    assert video_manager.get_cached_pose_path("video1.avi").name == "video1_pose_est_v3.h5"

    video_manager.remove_video("video1.avi")

    assert video_manager.videos == ["video2.mp4"]
    assert video_manager.num_videos == 1
    assert video_manager.total_project_identities == 5
    assert video_manager.get_video_identity_count("video1.avi") == 0
    assert video_manager.video_has_cm_per_pixel("video1.avi") is False
    # the per-video caches should no longer carry an entry for the removed video
    assert "video1.avi" not in video_manager._video_has_cm_per_pixel
    assert "video1.avi" not in video_manager._pose_path_cache


def test_remove_video_removes_project_file_entry(video_manager, settings_manager, project_paths):
    """Removing a video drops its video_files entry from project.json."""
    settings_manager.save_project_file(
        {
            "video_files": {
                "video1.avi": {"identities": 3},
                "video2.mp4": {"identities": 5},
            }
        }
    )

    video_manager.remove_video("video1.avi")

    assert settings_manager.project_settings["video_files"] == {"video2.mp4": {"identities": 5}}
    # the removal was persisted, not just applied in memory
    on_disk = json.loads(project_paths.project_file.read_text())
    assert "video1.avi" not in on_disk["video_files"]


def test_remove_video_unknown_video_is_a_no_op(video_manager):
    """Removing a video that is not in the project leaves state untouched."""
    video_manager.remove_video("nonexistent_video.avi")

    assert video_manager.videos == ["video1.avi", "video2.mp4"]
    assert video_manager.total_project_identities == 8


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
