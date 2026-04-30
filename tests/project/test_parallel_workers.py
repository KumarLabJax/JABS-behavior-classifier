"""Tests for parallel_workers — scan_video_metadata and _get_identity_count."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jabs.project.parallel_workers import (
    VideoScanJobSpec,
    VideoScanResult,
    _get_identity_count,
    scan_video_metadata,
)


def _h5_like(pose_grp_dict: dict) -> dict:
    """Wrap a pose group dict in a minimal h5-like file dict."""
    return {"poseest": pose_grp_dict}


class TestGetIdentityCount:
    """Tests for _get_identity_count."""

    def test_v2_always_returns_one(self):
        """V2 is single-identity regardless of HDF5 contents."""
        assert _get_identity_count(_h5_like({}), major_version=2) == 1

    def test_v1_returns_one(self):
        """Any version below 3 returns 1."""
        assert _get_identity_count(_h5_like({}), major_version=1) == 1

    def test_v3_uses_points_shape(self):
        """V3 reads identity count from points.shape[1]."""
        points_mock = MagicMock()
        points_mock.shape = (500, 3, 12, 2)
        pose_grp = {"points": points_mock}
        assert _get_identity_count(_h5_like(pose_grp), major_version=3) == 3

    def test_v4_uses_instance_id_center(self):
        """V4+ with instance_id_center reads identity count from its shape[0]."""
        id_center_mock = MagicMock()
        id_center_mock.shape = (4,)
        pose_grp = {"instance_id_center": id_center_mock}
        assert _get_identity_count(_h5_like(pose_grp), major_version=4) == 4

    def test_v5_uses_instance_id_center(self):
        """V5 behaves the same as V4 when instance_id_center is present."""
        id_center_mock = MagicMock()
        id_center_mock.shape = (2,)
        pose_grp = {"instance_id_center": id_center_mock}
        assert _get_identity_count(_h5_like(pose_grp), major_version=5) == 2

    def test_v4_fallback_no_instance_id_center_returns_zero(self):
        """V4+ without instance_id_center or embed_id data returns 0."""
        assert _get_identity_count(_h5_like({}), major_version=4) == 0

    def test_v4_fallback_embed_id(self):
        """V4+ without instance_id_center falls back to instance_embed_id max."""
        id_mask = np.array([[0, 0], [0, 1]])  # 0 = valid
        embed_id = np.array([[1, 2], [1, 0]])  # 1-based; max valid = 2

        pose_grp = {"instance_embed_id": embed_id, "id_mask": id_mask}
        assert _get_identity_count(_h5_like(pose_grp), major_version=4) == 2


# ---------------------------------------------------------------------------
# scan_video_metadata — integration test with real pose files
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.mark.parametrize(
    ("pose_filename", "expected_major_version"),
    [
        ("sample_pose_est_v3.h5", 3),
        ("sample_pose_est_v6.h5", 6),
    ],
)
def test_scan_video_metadata_returns_correct_types(
    tmp_path, pose_filename, expected_major_version
):
    """scan_video_metadata returns a VideoScanResult with correct field types."""
    pose_src = DATA_DIR / pose_filename
    if not pose_src.exists():
        pytest.skip(f"Test data file not found: {pose_filename}")

    video_name = "test_video.mp4"
    video_path = tmp_path / video_name
    video_path.touch()
    pose_path = tmp_path / pose_filename
    shutil.copy(pose_src, pose_path)

    job = VideoScanJobSpec(
        video=video_name,
        video_path=video_path,
        pose_path=pose_path,
        pose_major_version=expected_major_version,
        scan_frame_counts=False,
    )

    result = scan_video_metadata(job)

    assert result["video"] == video_name
    assert isinstance(result["hdf5_frame_count"], int)
    assert result["hdf5_frame_count"] > 0
    assert result["video_frame_count"] is None  # scan_frame_counts=False
    assert isinstance(result["identity_count"], int)
    assert result["identity_count"] >= 0
    assert isinstance(result["static_objects"], list)
    assert isinstance(result["lixit_keypoints"], int)
    assert isinstance(result["has_cm_per_pixel"], bool)


def test_scan_video_metadata_static_objects_v3(tmp_path):
    """V3 pose files have no static objects (feature is V5+)."""
    pose_src = DATA_DIR / "sample_pose_est_v3.h5"
    if not pose_src.exists():
        pytest.skip("V3 test data file not found")

    video_path = tmp_path / "vid.mp4"
    video_path.touch()
    pose_path = tmp_path / "vid_pose_est_v3.h5"
    shutil.copy(pose_src, pose_path)

    job = VideoScanJobSpec(
        video="vid.mp4",
        video_path=video_path,
        pose_path=pose_path,
        pose_major_version=3,
        scan_frame_counts=False,
    )
    result = scan_video_metadata(job)

    assert result["static_objects"] == []
    assert result["lixit_keypoints"] == 0


# ---------------------------------------------------------------------------
# VideoManager with pre-loaded scan_results — no I/O for metadata/frame check
# ---------------------------------------------------------------------------


def test_video_manager_with_scan_results_no_pose_open(tmp_path):
    """VideoManager._load_video_metadata uses scan_results without opening pose files."""
    from jabs.project.project_paths import ProjectPaths
    from jabs.project.settings_manager import SettingsManager
    from jabs.project.video_manager import VideoManager

    paths = ProjectPaths(base_path=tmp_path)
    paths.create_directories(validate=False)

    data_dir = Path(__file__).parent.parent / "data"
    (tmp_path / "video1.avi").touch()
    shutil.copy(data_dir / "sample_pose_est_v3.h5", tmp_path / "video1_pose_est_v3.h5")

    sm = SettingsManager(paths)

    scan_results: dict[str, VideoScanResult] = {
        "video1.avi": VideoScanResult(
            video="video1.avi",
            hdf5_frame_count=100,
            video_frame_count=None,
            identity_count=2,
            static_objects=[],
            lixit_keypoints=0,
            has_cm_per_pixel=False,
        )
    }

    with patch("jabs.project.video_manager.open_pose_file") as mock_open:
        vm = VideoManager(paths, sm, enable_video_check=False, scan_results=scan_results)

    mock_open.assert_not_called()
    assert vm.get_video_identity_count("video1.avi") == 2
    assert vm.total_project_identities == 2


def test_video_manager_with_scan_results_frame_count_validation(tmp_path):
    """VideoManager._validate_video_frame_counts uses scan_results frame counts."""
    from jabs.project.project_paths import ProjectPaths
    from jabs.project.settings_manager import SettingsManager
    from jabs.project.video_manager import VideoManager

    paths = ProjectPaths(base_path=tmp_path)
    paths.create_directories(validate=False)

    data_dir = Path(__file__).parent.parent / "data"
    (tmp_path / "video1.avi").touch()
    shutil.copy(data_dir / "sample_pose_est_v3.h5", tmp_path / "video1_pose_est_v3.h5")

    sm = SettingsManager(paths)
    scan_results: dict[str, VideoScanResult] = {
        "video1.avi": VideoScanResult(
            video="video1.avi",
            hdf5_frame_count=100,
            video_frame_count=100,
            identity_count=1,
            static_objects=[],
            lixit_keypoints=0,
            has_cm_per_pixel=True,
        )
    }

    vm = VideoManager(paths, sm, enable_video_check=True, scan_results=scan_results)
    assert vm.videos == ["video1.avi"]


def test_video_manager_scan_results_frame_mismatch_raises(tmp_path):
    """Frame count mismatch in scan_results triggers ValueError."""
    from jabs.project.project_paths import ProjectPaths
    from jabs.project.settings_manager import SettingsManager
    from jabs.project.video_manager import VideoManager

    paths = ProjectPaths(base_path=tmp_path)
    paths.create_directories(validate=False)

    data_dir = Path(__file__).parent.parent / "data"
    (tmp_path / "video1.avi").touch()
    shutil.copy(data_dir / "sample_pose_est_v3.h5", tmp_path / "video1_pose_est_v3.h5")

    sm = SettingsManager(paths)
    scan_results: dict[str, VideoScanResult] = {
        "video1.avi": VideoScanResult(
            video="video1.avi",
            hdf5_frame_count=100,
            video_frame_count=99,  # mismatch
            identity_count=1,
            static_objects=[],
            lixit_keypoints=0,
            has_cm_per_pixel=True,
        )
    }

    with pytest.raises(ValueError, match="frame counts differ"):
        VideoManager(paths, sm, enable_video_check=True, scan_results=scan_results)


# ---------------------------------------------------------------------------
# FeatureManager with pre-loaded scan_results — no HDF5 opens
# ---------------------------------------------------------------------------


def test_feature_manager_with_scan_results_no_hdf5_open(tmp_path):
    """FeatureManager.__initialize_pose_data skips HDF5 opens when scan_results provided."""
    from jabs.project.feature_manager import FeatureManager
    from jabs.project.project_paths import ProjectPaths

    paths = ProjectPaths(base_path=tmp_path)
    paths.create_directories(validate=False)

    data_dir = Path(__file__).parent.parent / "data"
    (tmp_path / "video1.avi").touch()
    shutil.copy(data_dir / "sample_pose_est_v3.h5", tmp_path / "video1_pose_est_v3.h5")

    scan_results: dict[str, VideoScanResult] = {
        "video1.avi": VideoScanResult(
            video="video1.avi",
            hdf5_frame_count=100,
            video_frame_count=None,
            identity_count=3,
            static_objects=[],
            lixit_keypoints=0,
            has_cm_per_pixel=True,
        )
    }

    fm = FeatureManager(paths, ["video1.avi"], scan_results=scan_results)
    assert fm.is_cm_unit


def test_feature_manager_scan_results_no_cm_per_pixel(tmp_path):
    """FeatureManager detects pixel-only project from scan_results."""
    from jabs.project.feature_manager import FeatureManager
    from jabs.project.project_paths import ProjectPaths

    paths = ProjectPaths(base_path=tmp_path)
    paths.create_directories(validate=False)

    data_dir = Path(__file__).parent.parent / "data"
    (tmp_path / "video1.avi").touch()
    shutil.copy(data_dir / "sample_pose_est_v3.h5", tmp_path / "video1_pose_est_v3.h5")

    scan_results: dict[str, VideoScanResult] = {
        "video1.avi": VideoScanResult(
            video="video1.avi",
            hdf5_frame_count=100,
            video_frame_count=None,
            identity_count=1,
            static_objects=[],
            lixit_keypoints=0,
            has_cm_per_pixel=False,
        )
    }

    fm = FeatureManager(paths, ["video1.avi"], scan_results=scan_results)
    assert not fm.is_cm_unit
