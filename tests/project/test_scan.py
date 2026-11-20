"""The project-load pose scan: unified files, and pose files it cannot scan."""

import json
import logging
import shutil
from pathlib import Path

import pytest

from jabs.core.enums import PoseCapability
from jabs.core.utils import hash_file
from jabs.pose_estimation.conversion_cache import ensure_converted
from jabs.project import Project

DATA_DIR = Path(__file__).parent.parent / "data"
LEGACY_CORPUS = (
    Path(__file__).parents[2] / "packages" / "jabs-io" / "tests" / "data" / "legacy-pose"
)
CACHE_RELPATH = Path("jabs") / "cache" / "pose_attribute_cache.json"


def _add_video(project_dir: Path, video: str, pose_source: Path, pose_name: str) -> None:
    """Add a stub video and a copy of a pose file under the given name."""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / video).touch()
    shutil.copyfile(pose_source, project_dir / pose_name)


def _open(project_dir: Path) -> Project:
    """Open a project on the deferred (cached) scan path."""
    return Project(project_dir, enable_video_check=False, enable_session_tracker=False)


def _cached_videos(project_dir: Path) -> dict:
    """Return the pose attribute cache's per-video entries."""
    return json.loads((project_dir / CACHE_RELPATH).read_text())["videos"]


@pytest.fixture
def project_with_bad_pose(tmp_path: Path) -> Path:
    """A project with one good v6 video and one whose pose the converter refuses."""
    project_dir = tmp_path / "proj"
    _add_video(project_dir, "good.avi", DATA_DIR / "sample_pose_est_v6.h5", "good_pose_est_v6.h5")
    _add_video(
        project_dir,
        "vid.avi",
        LEGACY_CORPUS / "v8-unknown-dataset.h5",
        "vid_pose_est_v8.h5",
    )
    return project_dir


def test_unscannable_pose_marks_the_video_unusable(
    project_with_bad_pose: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The project opens; the bad video is unusable with its reason, logged as a WARNING."""
    with caplog.at_level(logging.WARNING, logger="jabs.project.project"):
        project = _open(project_with_bad_pose)

    assert list(project.unusable_videos) == ["vid.avi"]
    assert project.unusable_videos["vid.avi"]
    assert "good.avi" not in project.unusable_videos
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("vid.avi" in r.getMessage() for r in warnings)


def test_unscannable_pose_is_not_cached(project_with_bad_pose: Path) -> None:
    """Only the good video is cached, so the bad one is rescanned next time."""
    _open(project_with_bad_pose)
    assert set(_cached_videos(project_with_bad_pose)) == {"good.avi"}

    reopened = _open(project_with_bad_pose)
    assert list(reopened.unusable_videos) == ["vid.avi"]


def test_unscannable_pose_does_not_restrict_project_features(
    project_with_bad_pose: Path, tmp_path: Path
) -> None:
    """The unusable video does not change the features the good video supports."""
    project = _open(project_with_bad_pose)
    alone = tmp_path / "alone"
    _add_video(alone, "good.avi", DATA_DIR / "sample_pose_est_v6.h5", "good_pose_est_v6.h5")
    reference = _open(alone)

    assert project.feature_manager.capabilities == reference.feature_manager.capabilities
    # the good v6 video has corners and a scale; the unusable one must not erase them
    assert project.feature_manager.static_objects == {"corners"}
    assert project.feature_manager.is_cm_unit
    assert reference.feature_manager.static_objects == {"corners"}


def test_unscannable_pose_with_video_check(project_with_bad_pose: Path) -> None:
    """The up-front frame-count check skips an unusable video instead of failing."""
    (project_with_bad_pose / "good.avi").unlink()
    project = Project(project_with_bad_pose, enable_video_check=True, enable_session_tracker=False)
    assert list(project.unusable_videos) == ["vid.avi"]


def test_project_opens_with_unified_pose_file(tmp_path: Path) -> None:
    """A unified ``*_pose.h5`` file is discovered, scanned and cached."""
    legacy = tmp_path / "src" / "vid_pose_est_v6.h5"
    legacy.parent.mkdir()
    shutil.copyfile(DATA_DIR / "sample_pose_est_v6.h5", legacy)
    converted = ensure_converted(legacy, hash_file(legacy), tmp_path / "cache")

    project_dir = tmp_path / "proj"
    _add_video(project_dir, "vid.avi", converted, "vid_pose.h5")
    project = _open(project_dir)

    assert project.unusable_videos == {}
    assert project.video_manager.videos == ["vid.avi"]
    assert project.video_manager.get_video_identity_count("vid.avi") == 4
    assert project.feature_manager.static_objects == {"corners"}
    assert project.feature_manager.is_cm_unit
    assert project.feature_manager.capabilities == frozenset(
        {PoseCapability.SOCIAL, PoseCapability.SEGMENTATION}
    )
    entry = _cached_videos(project_dir)["vid.avi"]
    assert entry["pose_file"] == "vid_pose.h5"
    assert "segmentation" in entry["capabilities"]


def test_directory_with_only_unified_pose_is_a_project(tmp_path: Path) -> None:
    """A directory whose only pose file is unified validates as a project."""
    legacy = tmp_path / "src" / "vid_pose_est_v4.h5"
    legacy.parent.mkdir()
    shutil.copyfile(DATA_DIR / "sample_pose_est_v4.h5", legacy)
    converted = ensure_converted(legacy, hash_file(legacy), tmp_path / "cache")

    project_dir = tmp_path / "proj"
    _add_video(project_dir, "vid.avi", converted, "vid_pose.h5")
    project = _open(project_dir)
    assert project.video_manager.videos == ["vid.avi"]
