"""Tests for the feature cache status a Project keeps in memory."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
from jabs.io.feature_cache.hdf5 import HDF5FeatureCacheWriter
from jabs.project import Project, VideoLabels

_DATA_DIR = Path(__file__).parent.parent / "data"
# sample_pose_est_v3.h5 tracks 5 identities over 1800 frames
_POSE_FILE = "sample_pose_est_v3.h5"
_NUM_FRAMES = 1800
_NUM_IDENTITIES = 5
_VIDEOS = ("video1.avi", "video2.avi")


@pytest.fixture(autouse=True)
def patch_session_tracker():
    """Patch the SessionTracker to avoid side effects during tests."""
    with patch("jabs.project.session_tracker.SessionTracker.__del__", return_value=None):
        yield


def _write_annotations(project_dir: Path, video: str, identities, behavior: str) -> None:
    """Write an annotation file labeling ``behavior`` for the given identities."""
    labels = VideoLabels(video, _NUM_FRAMES)
    for identity in identities:
        track = labels.get_track_labels(str(identity), behavior)
        track.label_behavior(100, 200)
        track.label_not_behavior(300, 400)

    pose_est = MagicMock()
    pose_est.identity_mask.return_value = np.full(_NUM_FRAMES, 1, dtype=bool)
    path = project_dir / "jabs" / "annotations" / Path(video).with_suffix(".json")
    with path.open("w", newline="\n") as f:
        json.dump(labels.as_dict(pose_est), f)


@pytest.fixture
def project(tmp_path) -> Project:
    """A two-video project; identity 0 of video1 is labeled for "Walking"."""
    for video in _VIDEOS:
        (tmp_path / video).touch()
        shutil.copy(_DATA_DIR / _POSE_FILE, tmp_path / video.replace(".avi", "_pose_est_v3.h5"))

    # first open creates the jabs directory so annotations can be placed in it
    Project(tmp_path, enable_video_check=False, enable_session_tracker=False)
    _write_annotations(tmp_path, _VIDEOS[0], identities=[0], behavior="Walking")
    return Project(tmp_path, enable_video_check=False, enable_session_tracker=False)


def _cache_window_features(project: Project, video: str, identity: int, window_size: int) -> None:
    """Write a per-frame plus window feature cache for one identity."""
    identity_dir = project.feature_dir / Path(video).stem / str(identity)
    writer = HDF5FeatureCacheWriter()
    metadata = FeatureCacheMetadata(
        feature_version=17,
        identity=identity,
        num_frames=_NUM_FRAMES,
        pose_hash="posehash",
    )
    writer.write_per_frame(
        identity_dir,
        metadata,
        PerFrameCacheData(
            frame_valid=np.ones(_NUM_FRAMES, dtype=np.uint8),
            features={"mod feat": np.zeros(_NUM_FRAMES)},
        ),
    )
    writer.write_window(
        identity_dir, metadata, window_size, {"mod mean feat": np.zeros(_NUM_FRAMES)}
    )


def test_status_starts_empty(project):
    """A freshly opened project has not scanned its feature cache."""
    assert project.feature_cache_status == {}


def test_refresh_stores_status_for_one_video(project):
    """Refreshing a video scans it and records the result on the project."""
    _cache_window_features(project, "video1.avi", identity=0, window_size=5)

    status = project.refresh_feature_cache_status("video1.avi")

    assert status.window_sizes == (5,)
    assert project.feature_cache_status["video1.avi"] is status
    assert "video2.avi" not in project.feature_cache_status


def test_set_and_invalidate_status(project):
    """Stored status can be replaced wholesale and dropped per video."""
    status = project.refresh_feature_cache_status("video1.avi")
    project.set_feature_cache_status({"video1.avi": status, "video2.avi": status})
    assert set(project.feature_cache_status) == {"video1.avi", "video2.avi"}

    project.invalidate_feature_cache_status(["video1.avi"])
    assert set(project.feature_cache_status) == {"video2.avi"}

    project.invalidate_feature_cache_status()
    assert project.feature_cache_status == {}


def test_clear_feature_cache_invalidates_status(project):
    """Clearing the cache discards the status describing it."""
    _cache_window_features(project, "video1.avi", identity=0, window_size=5)
    project.refresh_feature_cache_status("video1.avi")

    project.clear_feature_cache()

    assert project.feature_cache_status == {}
    assert project.refresh_feature_cache_status("video1.avi").has_cached_features is False


def test_all_videos_missing_when_nothing_cached(project):
    """With an empty cache every video needs feature computation."""
    assert project.videos_missing_window_features(5) == list(_VIDEOS)


def test_video_not_missing_once_every_identity_is_cached(project):
    """A video is only covered when all of its identities have the window size."""
    for identity in range(_NUM_IDENTITIES):
        _cache_window_features(project, "video1.avi", identity=identity, window_size=5)

    assert project.videos_missing_window_features(5) == ["video2.avi"]


def test_partially_cached_video_still_missing(project):
    """Caching some identities is not enough when every identity is required."""
    _cache_window_features(project, "video1.avi", identity=0, window_size=5)

    assert "video1.avi" in project.videos_missing_window_features(5)


def test_other_window_size_does_not_count(project):
    """A cache for a different window size does not cover the requested one."""
    for identity in range(_NUM_IDENTITIES):
        _cache_window_features(project, "video1.avi", identity=identity, window_size=10)

    assert "video1.avi" in project.videos_missing_window_features(5)


def test_missing_check_scans_unscanned_videos(project):
    """Videos with no stored status are scanned rather than assumed missing."""
    for identity in range(_NUM_IDENTITIES):
        _cache_window_features(project, "video1.avi", identity=identity, window_size=5)
    assert project.feature_cache_status == {}

    missing = project.videos_missing_window_features(5, videos=["video1.avi"])

    assert missing == []
    assert "video1.avi" in project.feature_cache_status


def test_missing_check_limited_to_requested_videos(project):
    """Only the requested videos are checked."""
    assert project.videos_missing_window_features(5, videos=["video2.avi"]) == ["video2.avi"]


def test_labeled_identities_only_needs_labeled_identity_cached(project):
    """Caching just the labeled identity covers a video for training."""
    _cache_window_features(project, "video1.avi", identity=0, window_size=5)
    identities = project.labeled_identities(["Walking"])

    assert identities == {"video1.avi": {0}}
    assert project.videos_missing_window_features(5, identities=identities) == []


def test_labeled_identities_ignores_other_behaviors(project):
    """A behavior with no labels anywhere selects no videos."""
    assert project.labeled_identities(["Grooming"]) == {}
    assert project.videos_missing_window_features(5, identities={}) == []


def test_labeled_identities_missing_when_its_cache_is_absent(project):
    """The labeled identity's own cache is what matters, not another identity's."""
    _cache_window_features(project, "video1.avi", identity=1, window_size=5)
    identities = project.labeled_identities(["Walking"])

    assert project.videos_missing_window_features(5, identities=identities) == ["video1.avi"]
