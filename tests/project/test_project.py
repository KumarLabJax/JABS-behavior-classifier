import contextlib
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from jabs.classifier import MultiClassClassifier
from jabs.classifier.protocols import ClassifierProtocol
from jabs.core.constants import CLASSIFIER_MODE_KEY, MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierMode
from jabs.core.utils import hash_file, hide_stderr
from jabs.project import Project, VideoLabels
from jabs.project.prediction_manager import MULTICLASS_PREDICTION_KEY
from jabs.project.project_utils import to_safe_name
from jabs.project.track_labels import TrackLabels


class _PathRecordingClassifier(ClassifierProtocol):
    """Minimal classifier test double that records save/load paths."""

    def __init__(self) -> None:
        self.saved_path: Path | None = None
        self.loaded_path: Path | None = None

    @property
    def feature_names(self) -> list[str] | None:
        """Return no feature names for this path-only test double."""
        return None

    def train(self, data: dict, random_seed: int | None = None) -> None:
        """Ignore training calls; tests only exercise save/load paths."""

    def predict(
        self,
        features: pd.DataFrame,
        frame_indexes: npt.NDArray[np.intp] | None = None,
    ) -> npt.NDArray[np.int8]:
        """Return an empty prediction array for protocol compatibility."""
        return np.array([], dtype=np.int8)

    def predict_proba(
        self,
        features: pd.DataFrame,
        frame_indexes: npt.NDArray[np.intp] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Return an empty probability array for protocol compatibility."""
        return np.array([], dtype=np.float32)

    def save(self, path: Path) -> None:
        """Record the save path and create a placeholder classifier file."""
        self.saved_path = path
        path.write_text("classifier", encoding="utf-8")

    def load(self, path: Path) -> None:
        """Record the load path and raise OSError when the file is missing."""
        self.loaded_path = path
        if not path.exists():
            raise OSError(f"Classifier does not exist: {path}")


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
    assert labels is not None
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
# save_classifier / load_classifier
# ---------------------------------------------------------------------------


def test_save_classifier_binary_uses_behavior_path(tmp_path: Path) -> None:
    """Binary classifier save path remains one pickle file per behavior."""
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )
    classifier = _PathRecordingClassifier()
    behavior = "Groom & Rear"

    project.save_classifier(classifier, behavior)

    expected_path = project.classifier_dir / f"{to_safe_name(behavior)}.pickle"
    assert classifier.saved_path == expected_path
    assert expected_path.exists()


def test_save_classifier_binary_requires_behavior(tmp_path: Path) -> None:
    """Binary classifier storage needs a behavior name for path selection."""
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )
    classifier = _PathRecordingClassifier()

    with pytest.raises(ValueError, match="behavior is required"):
        project.save_classifier(classifier)


def test_load_classifier_binary_uses_behavior_path(tmp_path: Path) -> None:
    """Binary classifier load path remains one pickle file per behavior."""
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )
    classifier = _PathRecordingClassifier()
    behavior = "Groom & Rear"
    expected_path = project.classifier_dir / f"{to_safe_name(behavior)}.pickle"
    expected_path.write_text("classifier", encoding="utf-8")

    assert project.load_classifier(classifier, behavior) is True
    assert classifier.loaded_path == expected_path


def test_load_classifier_missing_binary_returns_false(tmp_path: Path) -> None:
    """Missing binary classifier files return False instead of raising."""
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )
    classifier = _PathRecordingClassifier()

    assert project.load_classifier(classifier, "Missing Behavior") is False


def test_save_classifier_multiclass_uses_reserved_path(tmp_path: Path) -> None:
    """Multi-class classifier save path ignores behavior and uses one reserved file."""
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )
    project.settings_manager.save_project_file(
        {"settings": {CLASSIFIER_MODE_KEY: ClassifierMode.MULTICLASS.value}}
    )
    classifier = _PathRecordingClassifier()

    project.save_classifier(classifier)

    expected_path = project.classifier_dir / "_multiclass.pickle"
    assert classifier.saved_path == expected_path
    assert expected_path.exists()


def test_load_classifier_multiclass_uses_reserved_path(tmp_path: Path) -> None:
    """Multi-class classifier load path ignores behavior and uses one reserved file."""
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )
    project.settings_manager.save_project_file(
        {"settings": {CLASSIFIER_MODE_KEY: ClassifierMode.MULTICLASS.value}}
    )
    classifier = _PathRecordingClassifier()
    expected_path = project.classifier_dir / "_multiclass.pickle"
    expected_path.write_text("classifier", encoding="utf-8")

    assert project.load_classifier(classifier, "Ignored Behavior") is True
    assert classifier.loaded_path == expected_path


def test_load_classifier_missing_multiclass_returns_false(tmp_path: Path) -> None:
    """Missing multi-class classifier files return False instead of raising."""
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )
    project.settings_manager.save_project_file(
        {"settings": {CLASSIFIER_MODE_KEY: ClassifierMode.MULTICLASS.value}}
    )
    classifier = _PathRecordingClassifier()

    assert project.load_classifier(classifier) is False
    assert classifier.loaded_path == project.classifier_dir / "_multiclass.pickle"


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
    mock_vm.load_video_labels.side_effect = lambda v, pose=None: videos_and_labels[v]
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

    This keeps the validator consistent with classifier_utils.merge_labels(),
    which raises ValueError for the same condition at training time.
    """
    labels = VideoLabels("video1.avi", 100)
    track_none = labels.get_track_labels("0", MULTICLASS_NONE_BEHAVIOR)
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


# ---------------------------------------------------------------------------
# save_predictions probability-array shape
# ---------------------------------------------------------------------------


def _prediction_test_pose(num_identities: int, num_frames: int):
    """Minimal pose stand-in exposing the attributes save_predictions reads."""
    return type(
        "PoseEstimation",
        (object,),
        {
            "num_identities": num_identities,
            "num_frames": num_frames,
            "pose_file": "video1_pose_est_v6.h5",
            "hash": "posehash",
            "identity_to_track": None,
            "external_identities": None,
        },
    )()


def _prediction_test_classifier():
    """Minimal classifier stand-in exposing the metadata write_predictions reads."""
    return type(
        "Classifier",
        (object,),
        {"classifier_file": "_multiclass.pickle", "classifier_hash": "clshash"},
    )()


def _bare_project(tmp_path: Path) -> Project:
    return Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )


def test_save_predictions_multiclass_empty_allocates_per_class_shape(tmp_path: Path) -> None:
    """An empty multi-class video still allocates a (n_id, n_frames, n_classes) prob array.

    Shape is derived from class_names, not from sniffing a sample probability
    array, so an empty ``probabilities`` dict no longer falls through to a binary
    2D shape (which would then fail BehaviorPrediction's shape validation).
    """
    project = _bare_project(tmp_path)
    class_names = [MULTICLASS_NONE_BEHAVIOR, "Walk", "Run"]

    project.save_predictions(
        _prediction_test_pose(2, 5),
        "video1.avi",
        {},
        {},
        MULTICLASS_PREDICTION_KEY,
        _prediction_test_classifier(),
        class_names=class_names,
    )

    safe = to_safe_name(MULTICLASS_PREDICTION_KEY)
    with h5py.File(project.project_paths.prediction_dir / "video1.h5", "r") as hf:
        assert hf[f"predictions/{safe}/probabilities"].shape == (2, 5, 3)


def test_save_predictions_multiclass_writes_per_class_probabilities(tmp_path: Path) -> None:
    """Non-empty multi-class predictions write the per-identity per-class matrices."""
    project = _bare_project(tmp_path)
    class_names = [MULTICLASS_NONE_BEHAVIOR, "Walk", "Run"]
    predictions = {0: np.array([1, 0, 2], dtype=np.int8)}
    probabilities = {
        0: np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32)
    }

    project.save_predictions(
        _prediction_test_pose(1, 3),
        "video1.avi",
        predictions,
        probabilities,
        MULTICLASS_PREDICTION_KEY,
        _prediction_test_classifier(),
        class_names=class_names,
    )

    safe = to_safe_name(MULTICLASS_PREDICTION_KEY)
    with h5py.File(project.project_paths.prediction_dir / "video1.h5", "r") as hf:
        probs = hf[f"predictions/{safe}/probabilities"][()]
    assert probs.shape == (1, 3, 3)
    np.testing.assert_allclose(probs[0], probabilities[0])


def test_save_predictions_multiclass_rejects_misshaped_probabilities(tmp_path: Path) -> None:
    """A per-identity probability array that doesn't match (n_frames, n_classes) raises.

    Guards against a mis-shaped array (e.g. 1D or (n_frames, 1)) broadcasting
    silently into the class-sized allocation instead of failing.
    """
    project = _bare_project(tmp_path)
    class_names = [MULTICLASS_NONE_BEHAVIOR, "Walk", "Run"]
    predictions = {0: np.array([1, 0, 2], dtype=np.int8)}
    # 1D probabilities would broadcast across the 3 class columns without a check
    probabilities = {0: np.array([0.9, 0.4, 0.8], dtype=np.float32)}

    with pytest.raises(ValueError, match="probability array for identity 0"):
        project.save_predictions(
            _prediction_test_pose(1, 3),
            "video1.avi",
            predictions,
            probabilities,
            MULTICLASS_PREDICTION_KEY,
            _prediction_test_classifier(),
            class_names=class_names,
        )


def test_save_predictions_binary_allocates_scalar_shape(tmp_path: Path) -> None:
    """Binary predictions (no class_names) allocate a 2D (n_id, n_frames) prob array."""
    project = _bare_project(tmp_path)
    predictions = {0: np.array([1, 0, 1], dtype=np.int8)}
    probabilities = {0: np.array([0.9, 0.4, 0.8], dtype=np.float32)}

    project.save_predictions(
        _prediction_test_pose(1, 3),
        "video1.avi",
        predictions,
        probabilities,
        "Walk",
        _prediction_test_classifier(),
    )

    with h5py.File(project.project_paths.prediction_dir / "video1.h5", "r") as hf:
        assert hf["predictions/Walk/probabilities"].shape == (1, 3)


# ---------------------------------------------------------------------------
# excluded-from-training group ids
# ---------------------------------------------------------------------------


def test_excluded_group_ids_maps_excluded_videos(tmp_path: Path) -> None:
    """_excluded_group_ids returns the group ids whose source video is excluded."""
    project = _bare_project(tmp_path)
    project.settings_manager.set_video_excluded("v2.avi", True)

    group_mapping = {
        0: {"video": "v1.avi", "identity": None},
        1: {"video": "v2.avi", "identity": None},
        2: {"video": "v3.avi", "identity": None},
    }
    assert project._excluded_group_ids(group_mapping) == {1}


def test_excluded_group_ids_empty_when_none_excluded(tmp_path: Path) -> None:
    """No excluded videos -> empty set."""
    project = _bare_project(tmp_path)
    group_mapping = {0: {"video": "v1.avi", "identity": None}}
    assert project._excluded_group_ids(group_mapping) == set()


def test_rename_behavior_multiclass_updates_classifier_and_predictions(tmp_path: Path) -> None:
    """In multi-class mode, rename updates the shared classifier and prediction class_names.

    The single ``_multiclass.pickle`` and the per-video ``class_names`` dataset
    are not behavior-keyed, so they must be updated in place rather than moved.
    """
    project = _make_project_with_mock_vm(tmp_path, {"video1.avi": None})
    project.settings_manager.save_project_file(
        {"settings": {CLASSIFIER_MODE_KEY: ClassifierMode.MULTICLASS.value}}
    )
    project.settings_manager.save_behavior("Walk", {})
    project.settings_manager.save_behavior("Run", {})

    # save a real multi-class classifier under the reserved filename
    classifier_path = project.classifier_dir / "_multiclass.pickle"
    MultiClassClassifier(["Walk", "Run"]).save(classifier_path)

    # write a prediction file holding the shared multi-class class_names dataset
    # plus stale classifier metadata referencing the pre-rename pickle
    safe_multiclass = to_safe_name(MULTICLASS_PREDICTION_KEY)
    pred_file = project.project_paths.prediction_dir / "video1.h5"
    with h5py.File(pred_file, "w") as hf:
        group = hf.create_group(f"predictions/{safe_multiclass}")
        group.attrs["classifier_file"] = "_multiclass.pickle"
        group.attrs["classifier_hash"] = "stale-pre-rename-hash"
        group.create_dataset(
            "class_names",
            data=np.array([MULTICLASS_NONE_BEHAVIOR, "Walk", "Run"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    project.rename_behavior("Walk", "Standing")

    # classifier behavior names updated in place (class index order preserved)
    reloaded = MultiClassClassifier.from_pickle(classifier_path)
    assert reloaded.behavior_names == ["Standing", "Run"]

    # the rewritten pickle's recorded hash matches its new contents (not stale)
    assert reloaded.classifier_hash == hash_file(classifier_path)

    # prediction class_names dataset updated, None class untouched, and the
    # group's classifier metadata repointed at the rewritten pickle
    with h5py.File(pred_file, "r") as hf:
        group = hf[f"predictions/{safe_multiclass}"]
        names = [v.decode("utf-8") for v in group["class_names"][()]]
        assert group.attrs["classifier_hash"] == reloaded.classifier_hash
        assert group.attrs["classifier_file"] == reloaded.classifier_file
    assert names == [MULTICLASS_NONE_BEHAVIOR, "Standing", "Run"]

    # project settings reflect the rename
    assert "Standing" in project.settings_manager.behavior_names
    assert "Walk" not in project.settings_manager.behavior_names


def test_rename_behavior_multiclass_rejects_reserved_none_name(tmp_path: Path) -> None:
    """In multi-class mode, renaming a behavior to the reserved "None" name is rejected.

    The guard lives at the project level so it fires even when no multi-class
    classifier has been trained/saved yet (the in-classifier validation would
    otherwise be skipped).
    """
    project = _make_project_with_mock_vm(tmp_path, {"video1.avi": None})
    project.settings_manager.save_project_file(
        {"settings": {CLASSIFIER_MODE_KEY: ClassifierMode.MULTICLASS.value}}
    )
    project.settings_manager.save_behavior("Walk", {})

    # No _multiclass.pickle exists, so the only protection is the project guard.
    with pytest.raises(ValueError, match="reserved"):
        project.rename_behavior("Walk", MULTICLASS_NONE_BEHAVIOR)

    # behavior list is unchanged and never gained the reserved name
    assert "Walk" in project.settings_manager.behavior_names
    assert MULTICLASS_NONE_BEHAVIOR not in project.settings_manager.behavior_names


def test_get_multiclass_labeled_features_aligns_labels_and_features(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Project multiclass feature collection concatenates aligned labels_by_behavior arrays."""
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )
    project.settings_manager.save_behavior("Walk", {"window_size": 3})
    project.settings_manager.save_behavior("Run", {"window_size": 3})

    mock_vm = MagicMock()
    mock_vm.videos = ["video_a.avi", "video_b.avi"]
    mock_vm.video_path.side_effect = lambda v: tmp_path / v
    mock_vm.get_cached_pose_path.side_effect = lambda v: tmp_path / v.replace(
        ".avi", "_pose_est_v6.h5"
    )
    project._video_manager = mock_vm

    jobs_seen: list[dict] = []

    def _fake_collect(job: dict) -> dict:
        jobs_seen.append(job)
        if job["video"] == "video_a.avi":
            return {
                "per_frame": [pd.DataFrame({"f": [1.0, 2.0]})],
                "window": [pd.DataFrame({"w": [10.0, 20.0]})],
                "labels": [np.array([1, 1], dtype=np.int8)],
                "labels_by_behavior": [
                    {
                        MULTICLASS_NONE_BEHAVIOR: np.array([TrackLabels.Label.BEHAVIOR, 0]),
                        "Walk": np.array([0, TrackLabels.Label.BEHAVIOR]),
                        "Run": np.array([0, 0]),
                    }
                ],
                "group_keys": [("video_a.avi", 0)],
            }
        return {
            "per_frame": [pd.DataFrame({"f": [3.0]})],
            "window": [pd.DataFrame({"w": [30.0]})],
            "labels": [np.array([1], dtype=np.int8)],
            "labels_by_behavior": [
                {
                    MULTICLASS_NONE_BEHAVIOR: np.array([0]),
                    "Walk": np.array([0]),
                    "Run": np.array([TrackLabels.Label.BEHAVIOR]),
                }
            ],
            "group_keys": [("video_b.avi", 1)],
        }

    monkeypatch.setattr("jabs.project.project.collect_multiclass_labeled_features", _fake_collect)

    progress_calls = {"count": 0}
    features, group_mapping = project.get_multiclass_labeled_features(
        progress_callable=lambda: progress_calls.__setitem__("count", progress_calls["count"] + 1)
    )

    assert progress_calls["count"] == 2
    assert all(job["behavior_names"] == ["Walk", "Run"] for job in jobs_seen)

    assert features["per_frame"].shape[0] == 3
    assert features["window"].shape[0] == 3
    assert features["groups"].shape[0] == 3

    assert np.array_equal(
        features["labels_by_behavior"][MULTICLASS_NONE_BEHAVIOR],
        np.array([TrackLabels.Label.BEHAVIOR, 0, 0]),
    )
    assert np.array_equal(
        features["labels_by_behavior"]["Walk"],
        np.array([0, TrackLabels.Label.BEHAVIOR, 0]),
    )
    assert np.array_equal(
        features["labels_by_behavior"]["Run"],
        np.array([0, 0, TrackLabels.Label.BEHAVIOR]),
    )
    assert {(entry["video"], entry["identity"]) for entry in group_mapping.values()} == {
        ("video_a.avi", 0),
        ("video_b.avi", 1),
    }


def test_get_multiclass_labeled_features_fills_missing_behavior_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Missing behavior label keys are filled with NONE arrays aligned to feature rows."""
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )
    project.settings_manager.save_behavior("Walk", {"window_size": 3})
    project.settings_manager.save_behavior("Run", {"window_size": 3})

    mock_vm = MagicMock()
    mock_vm.videos = ["video_a.avi"]
    mock_vm.video_path.side_effect = lambda v: tmp_path / v
    mock_vm.get_cached_pose_path.side_effect = lambda v: tmp_path / v.replace(
        ".avi", "_pose_est_v6.h5"
    )
    project._video_manager = mock_vm

    def _fake_collect(_job: dict) -> dict:
        return {
            "per_frame": [pd.DataFrame({"f": [1.0, 2.0]})],
            "window": [pd.DataFrame({"w": [10.0, 20.0]})],
            "labels": [np.array([1, 1], dtype=np.int8)],
            "labels_by_behavior": [
                {
                    MULTICLASS_NONE_BEHAVIOR: np.array([TrackLabels.Label.BEHAVIOR, 0]),
                    "Walk": np.array([0, TrackLabels.Label.BEHAVIOR]),
                    # "Run" intentionally omitted
                }
            ],
            "group_keys": [("video_a.avi", 0)],
        }

    monkeypatch.setattr("jabs.project.project.collect_multiclass_labeled_features", _fake_collect)

    features, _ = project.get_multiclass_labeled_features()

    assert features["per_frame"].shape[0] == 2
    assert "Run" in features["labels_by_behavior"]
    np.testing.assert_array_equal(
        features["labels_by_behavior"]["Run"],
        np.full(2, TrackLabels.Label.NONE, dtype=np.int8),
    )
