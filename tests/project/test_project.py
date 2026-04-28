import contextlib
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jabs.core.utils import hide_stderr
from jabs.project import Project, VideoLabels


@pytest.fixture(autouse=True, scope="session")
def patch_session_tracker():
    """Patch the SessionTracker to avoid side effects during tests."""
    with patch("jabs.project.session_tracker.SessionTracker.__del__", return_value=None):
        yield


@pytest.fixture(scope="module")
def project_with_data():
    """Fixture to create a project with empty video file and annotations,and clean up afterwards."""
    _EXISTING_PROJ_PATH = Path("test_project_with_data")
    _FILENAMES: list[str] = ["test_file_1.avi", "test_file_2.avi"]

    # filenames of some sample pose files in the test/data directory.
    # must be at least as long as _FILENAMES
    _POSE_FILES: list[str] = [
        "identity_with_no_data_pose_est_v3.h5",
        "sample_pose_est_v3.h5",
    ]

    test_data_dir = Path(__file__).parent.parent / "data"

    # make sure the test project dir is gone in case we previously
    # threw an exception during setup
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(_EXISTING_PROJ_PATH)

    _EXISTING_PROJ_PATH.mkdir()

    for i, name in enumerate(_FILENAMES):
        # make a stub for the .avi file in the project directory
        (_EXISTING_PROJ_PATH / name).touch()

        # copy the sample pose_est files
        pose_filename = name.replace(".avi", "_pose_est_v3.h5")
        pose_path = _EXISTING_PROJ_PATH / pose_filename
        pose_source = test_data_dir / _POSE_FILES[i]

        shutil.copy(pose_source, pose_path)

    # set up a project directory with annotations
    Project(_EXISTING_PROJ_PATH, enable_video_check=False, enable_session_tracker=False)

    # Create a mock pose_est object with required methods
    mock_pose_est = MagicMock()
    mock_pose_est.identity_mask.return_value = np.full(10000, 1, dtype=bool)

    # create an annotation
    labels = VideoLabels(_FILENAMES[0], 10000)
    walking_labels = labels.get_track_labels("0", "Walking")
    walking_labels.label_behavior(100, 200)
    walking_labels.label_behavior(500, 1000)
    walking_labels.label_not_behavior(1001, 2000)

    # and manually place the .json file in the project directory
    with (
        _EXISTING_PROJ_PATH / "jabs" / "annotations" / Path(_FILENAMES[0]).with_suffix(".json")
    ).open("w", newline="\n") as f:
        json.dump(labels.as_dict(mock_pose_est), f)

    # open project
    project = Project(_EXISTING_PROJ_PATH, enable_video_check=False, enable_session_tracker=False)

    yield project

    # teardown
    shutil.rmtree(_EXISTING_PROJ_PATH)


def test_create():
    """test creating a new empty Project"""
    project_dir = Path("test_project_dir")
    project = Project(project_dir, enable_session_tracker=False, validate_project_dir=False)

    # make sure that the empty project directory was created
    assert project_dir.exists()

    # make sure the jabs directory was created
    assert project.project_paths.jabs_dir.exists()

    # make sure the jabs/annotations directory was created
    assert project.project_paths.annotations_dir.exists()

    # make sure the jabs/predictions directory was created
    assert project.project_paths.prediction_dir.exists()

    # remove project dir
    shutil.rmtree(project_dir)


def test_get_video_list(project_with_data):
    """get list of video files in an existing project"""
    assert project_with_data.video_manager.videos == ["test_file_1.avi", "test_file_2.avi"]


def test_load_annotations(project_with_data):
    """test loading annotations from a saved project"""
    mock_pose_est = MagicMock()
    mock_pose_est.identity_mask.return_value = np.full(10000, 1, dtype=bool)
    mock_pose_est.num_frames = 10000

    labels = project_with_data.video_manager.load_video_labels("test_file_1.avi")

    with (
        Path("test_project_with_data")
        / "jabs"
        / "annotations"
        / Path("test_file_1.avi").with_suffix(".json")
    ).open("r") as f:
        dict_from_file = json.load(f)

    assert len(project_with_data.video_manager.videos) == 2

    # check to see that calling as_dict() on the VideoLabels object
    # matches what was used to load the annotation track from disk
    assert labels.as_dict(mock_pose_est) == dict_from_file


def test_save_annotations(project_with_data):
    """test saving annotations"""
    mock_pose_est = MagicMock()
    mock_pose_est.identity_mask.return_value = np.full(10000, 1, dtype=bool)
    mock_pose_est.num_frames = 10000

    labels = project_with_data.video_manager.load_video_labels("test_file_1.avi")
    walking_labels = labels.get_track_labels("0", "Walking")

    # make some changes
    walking_labels.label_behavior(5000, 5500)

    # save changes
    project_with_data.save_annotations(labels, mock_pose_est)

    # make sure the .json file in the project directory matches the new
    # state
    with (
        Path("test_project_with_data")
        / "jabs"
        / "annotations"
        / Path("test_file_1.avi").with_suffix(".json")
    ).open("r") as f:
        dict_from_file = json.load(f)

    # need to add the project labeler to the labels dict
    labels_as_dict = labels.as_dict(mock_pose_est)
    labels_as_dict["labeler"] = project_with_data.labeler

    assert labels_as_dict == dict_from_file


def test_load_annotations_bad_filename(project_with_data):
    """test load annotations for a file that doesn't exist raises ValueError"""
    with pytest.raises(ValueError):
        project_with_data.video_manager.load_video_labels("bad_filename.avi")


def test_no_saved_video_labels(project_with_data):
    """test loading labels for a video with no saved labels returns None"""
    assert project_with_data.video_manager.load_video_labels("test_file_2.avi") is None


def test_bad_video_file(project_with_data):
    """test loading a video file that doesn't exist raises ValueError"""
    with pytest.raises(IOError), hide_stderr():
        _ = Project(Path("test_project_with_data"), enable_session_tracker=False)


def test_min_pose_version(project_with_data):
    """dummy project contains version 3 and 4 pose files min should be 3"""
    assert project_with_data.feature_manager.min_pose_version == 3


def test_can_use_social_true(project_with_data):
    """test that can_use_social_features is True when social features are enabled"""
    assert project_with_data.feature_manager.can_use_social_features


def test_project_load_pose_from_custom_pose_dir(tmp_path):
    """Project.load_pose_est should resolve pose from pose_dir, not project_dir."""
    project_root = tmp_path / "project"
    video_dir = tmp_path / "videos"
    pose_dir = tmp_path / "poses"
    project_root.mkdir()
    video_dir.mkdir()
    pose_dir.mkdir()

    (video_dir / "video1.avi").touch()

    test_data_dir = Path(__file__).parent.parent / "data"
    shutil.copy(test_data_dir / "sample_pose_est_v4.h5", pose_dir / "video1_pose_est_v4.h5")

    project = Project(
        project_root,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
        video_dir=video_dir,
        pose_dir=pose_dir,
    )

    pose = project.load_pose_est(project.video_manager.video_path("video1.avi"))

    assert Path(pose.pose_file) == pose_dir / "video1_pose_est_v4.h5"


def test_rename_behavior_raises_if_new_name_exists(tmp_path) -> None:
    """Test that renaming a behavior to an existing name raises a ValueError."""
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )

    # Seed settings with two behaviors: one to rename, and one that already exists
    existing_settings = {
        "window_size": 5,
        "balance_labels": True,
        "symmetric_behavior": False,
    }
    project.settings_manager.save_behavior("Existing", existing_settings)
    project.settings_manager.save_behavior("Old", existing_settings)

    # Attempt to rename "Old" -> "Existing" should raise
    with pytest.raises(ValueError):
        project.rename_behavior("Old", "Existing")

    # Ensure behaviors are unchanged after the failed rename
    names = set(project.settings_manager.behavior_names)
    assert "Existing" in names
    assert "Old" in names


# ---------------------------------------------------------------------------
# get_overlapping_behavior_label_videos
# ---------------------------------------------------------------------------


def _make_project_with_mock_vm(tmp_path: Path, videos_and_labels: dict) -> Project:
    """Return a Project whose VideoManager is replaced by a mock.

    Args:
        tmp_path: Temporary directory for the project.
        videos_and_labels: Mapping of video filename → VideoLabels | None.
    """
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )
    mock_vm = MagicMock()
    mock_vm.videos = list(videos_and_labels.keys())
    mock_vm.video_path.side_effect = lambda v: tmp_path / v
    mock_vm.load_video_labels.side_effect = lambda v, pose: videos_and_labels[v]
    project._video_manager = mock_vm
    project.load_pose_est = MagicMock(return_value=MagicMock())
    project.save_annotations = MagicMock()
    return project


def test_overlapping_labels_no_videos(tmp_path: Path) -> None:
    """Returns empty list when the project has no videos."""
    project = _make_project_with_mock_vm(tmp_path, {})
    assert project.get_overlapping_behavior_label_videos() == []


def test_overlapping_labels_none_labels(tmp_path: Path) -> None:
    """Returns empty list when load_video_labels returns None."""
    project = _make_project_with_mock_vm(tmp_path, {"video1.avi": None})
    assert project.get_overlapping_behavior_label_videos() == []


def test_overlapping_labels_single_behavior(tmp_path: Path) -> None:
    """Single behavior per identity - no conflict possible."""
    labels = VideoLabels("video1.avi", 100)
    track = labels.get_track_labels("0", "Walk")
    track.label_behavior(10, 20)
    project = _make_project_with_mock_vm(tmp_path, {"video1.avi": labels})
    assert project.get_overlapping_behavior_label_videos() == []


def test_overlapping_labels_no_conflict(tmp_path: Path) -> None:
    """Two behaviors on the same identity but on different frames - no conflict."""
    labels = VideoLabels("video1.avi", 100)
    track_a = labels.get_track_labels("0", "Walk")
    track_a.label_behavior(10, 20)
    track_b = labels.get_track_labels("0", "Run")
    track_b.label_behavior(30, 40)
    project = _make_project_with_mock_vm(tmp_path, {"video1.avi": labels})
    assert project.get_overlapping_behavior_label_videos() == []


def test_overlapping_labels_conflict_detected(tmp_path: Path) -> None:
    """Two behaviors share a labeled frame on the same identity - conflict detected."""
    labels = VideoLabels("video1.avi", 100)
    track_a = labels.get_track_labels("0", "Walk")
    track_a.label_behavior(10, 30)
    track_b = labels.get_track_labels("0", "Run")
    track_b.label_behavior(20, 40)  # overlaps frames 20-30
    project = _make_project_with_mock_vm(tmp_path, {"video1.avi": labels})
    assert project.get_overlapping_behavior_label_videos() == ["video1.avi"]


def test_overlapping_labels_none_behavior_is_a_conflict(tmp_path: Path) -> None:
    """A behavior and the None behavior sharing a BEHAVIOR-labeled frame is a conflict.

    This keeps the validator consistent with MultiClassClassifier.merge_labels(),
    which raises ValueError for the same condition at training time.
    """
    labels = VideoLabels("video1.avi", 100)
    track_none = labels.get_track_labels("0", "None")
    track_none.label_behavior(10, 20)
    track_walk = labels.get_track_labels("0", "Walk")
    track_walk.label_behavior(15, 25)  # overlaps frames 15-20 with None
    project = _make_project_with_mock_vm(tmp_path, {"video1.avi": labels})
    assert project.get_overlapping_behavior_label_videos() == ["video1.avi"]


def test_overlapping_labels_multiple_videos_sorted(tmp_path: Path) -> None:
    """Conflicting filenames are returned sorted."""

    def _conflicting_labels(name: str) -> VideoLabels:
        lbl = VideoLabels(name, 100)
        lbl.get_track_labels("0", "Walk").label_behavior(10, 30)
        lbl.get_track_labels("0", "Run").label_behavior(20, 40)
        return lbl

    clean = VideoLabels("aaa.avi", 100)
    clean.get_track_labels("0", "Walk").label_behavior(10, 20)

    project = _make_project_with_mock_vm(
        tmp_path,
        {
            "zzz.avi": _conflicting_labels("zzz.avi"),
            "aaa.avi": clean,
            "mmm.avi": _conflicting_labels("mmm.avi"),
        },
    )
    assert project.get_overlapping_behavior_label_videos() == ["mmm.avi", "zzz.avi"]
