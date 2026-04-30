import h5py
import numpy as np
import pytest

from jabs.project.prediction_manager import MULTICLASS_PREDICTION_KEY, PredictionManager


class MockProjectPaths:
    """Class to simulate project paths."""

    def __init__(self, base_path):
        self.prediction_dir = base_path / "predictions"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)


class MockProject:
    """Class to simulate a project."""

    def __init__(self, base_path):
        self.project_paths = MockProjectPaths(base_path)
        self.settings_manager = MockSettingsManager()


class MockSettingsManager:
    """Class to simulate project settings."""

    def __init__(self):
        self.project_settings = {"video_files": {"test_video.avi": {"identities": 2}}}


@pytest.fixture
def mock_project(tmp_path):
    """Fixture to create a mock project with necessary paths."""
    return MockProject(tmp_path)


@pytest.fixture
def prediction_manager(mock_project):
    """Fixture to create a PredictionManager instance."""
    return PredictionManager(mock_project)


@pytest.fixture
def mock_pose():
    """Fixture to create a minimal pose object for prediction writes."""
    return type(
        "PoseEstimation",
        (object,),
        {
            "pose_file": "pose_file.h5",
            "hash": "12345",
            "identity_to_track": None,
            "external_identities": None,
        },
    )()


@pytest.fixture
def mock_classifier():
    """Fixture to create a minimal classifier object for prediction writes."""
    return type(
        "Classifier",
        (object,),
        {"classifier_file": "classifier.pkl", "classifier_hash": "67890"},
    )()


def test_write_predictions(prediction_manager, tmp_path):
    """Test writing predictions to an HDF5 file."""
    output_path = tmp_path / "test_predictions.h5"
    behavior = "Walking"
    predictions = np.array([[1, 0, -1], [0, 1, -1]])
    probabilities = np.array([[0.9, 0.8, -1], [0.7, 0.6, -1]])
    poses = type(
        "PoseEstimation",
        (object,),
        {
            "pose_file": "pose_file.h5",
            "hash": "12345",
            "identity_to_track": None,
            "external_identities": None,
        },
    )()
    classifier = type(
        "Classifier",
        (object,),
        {"classifier_file": "classifier.pkl", "classifier_hash": "67890"},
    )()

    PredictionManager.write_predictions(
        behavior, output_path, predictions, probabilities, poses, classifier
    )

    with h5py.File(output_path, "r") as h5:
        assert h5.attrs["pose_file"] == "pose_file.h5"
        assert h5.attrs["pose_hash"] == "12345"
        assert h5.attrs["version"] == 2
        assert "predictions" in h5
        assert behavior in h5["predictions"]
        behavior_group = h5["predictions"][behavior]
        assert np.array_equal(behavior_group["predicted_class"], predictions)
        assert np.array_equal(behavior_group["probabilities"], probabilities)


def test_load_predictions(prediction_manager, mock_project):
    """Test loading predictions for a video and behavior."""
    video = "test_video.avi"
    behavior = "Walking"
    prediction_file = mock_project.project_paths.prediction_dir / "test_video.h5"

    # Create a valid HDF5 prediction file
    with h5py.File(prediction_file, "w") as h5:
        h5.attrs["version"] = 2
        h5.attrs["pose_file"] = "test_pose.h5"
        h5.attrs["pose_hash"] = "testhash"
        prediction_group = h5.create_group("predictions")
        behavior_group = prediction_group.create_group(behavior)
        behavior_group.attrs["classifier_file"] = "test_classifier.pkl"
        behavior_group.attrs["classifier_hash"] = "clshash"
        behavior_group.attrs["app_version"] = "1.0.0"
        behavior_group.attrs["prediction_date"] = "2025-01-01"
        behavior_group.create_dataset("predicted_class", data=[[1, 0, -1], [0, 1, -1]])
        behavior_group.create_dataset("probabilities", data=[[0.9, 0.8, -1], [0.7, 0.6, -1]])

    # prediction file doesn't have postprocessed predictions so only get raw predictions and probabilities
    predictions, probabilities, _ = prediction_manager.load_predictions(video, behavior)

    assert 0 in predictions
    assert 1 in predictions
    assert np.array_equal(predictions[0], [1, 0, -1])
    assert np.array_equal(probabilities[0], [0.9, 0.8, -1])


def test_load_multiclass_predictions_roundtrip(
    prediction_manager,
    mock_project,
    mock_pose,
    mock_classifier,
) -> None:
    """Multi-class predictions load with class names and per-identity probability matrices."""
    prediction_file = mock_project.project_paths.prediction_dir / "test_video.h5"
    predictions = np.array([[1, 0, -1], [2, 1, -1]], dtype=np.int8)
    probabilities = np.array(
        [
            [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.0, 0.0, 0.0]],
            [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    class_names = ["None", "Walking", "Rearing"]

    PredictionManager.write_predictions(
        MULTICLASS_PREDICTION_KEY,
        prediction_file,
        predictions,
        probabilities,
        mock_pose,
        mock_classifier,
        class_names=class_names,
    )

    loaded_predictions, loaded_probabilities, loaded_postprocessed, loaded_class_names = (
        prediction_manager.load_multiclass_predictions("test_video.avi")
    )

    assert loaded_class_names == class_names
    assert loaded_postprocessed == {}
    assert np.array_equal(loaded_predictions[0], predictions[0])
    assert np.array_equal(loaded_predictions[1], predictions[1])
    assert np.array_equal(loaded_probabilities[0], probabilities[0])
    assert np.array_equal(loaded_probabilities[1], probabilities[1])


def test_load_multiclass_predictions_missing(prediction_manager) -> None:
    """Missing multi-class predictions return empty dicts and no class names."""
    predictions, probabilities, predictions_postprocessed, class_names = (
        prediction_manager.load_multiclass_predictions("test_video.avi")
    )

    assert predictions == {}
    assert probabilities == {}
    assert predictions_postprocessed == {}
    assert class_names is None


def test_load_predictions_missing_behavior(prediction_manager, mock_project):
    """Test loading predictions when behavior is missing."""
    video = "test_video.avi"
    behavior = "Running"
    prediction_file = mock_project.project_paths.prediction_dir / "test_video.h5"

    # Create a mock prediction file without the behavior
    with h5py.File(prediction_file, "w") as h5:
        h5.attrs["version"] = 2
        h5.create_group("predictions")

    predictions, probabilities, predictions_postprocessed = prediction_manager.load_predictions(
        video, behavior
    )

    assert predictions == {}
    assert probabilities == {}
    assert predictions_postprocessed == {}


def test_load_predictions_invalid_file(prediction_manager, mock_project):
    """Test loading predictions from an invalid file."""
    video = "test_video.avi"
    prediction_file = mock_project.project_paths.prediction_dir / "test_video.h5"

    # Create an invalid prediction file
    with open(prediction_file, "w") as f:
        f.write("invalid content")

    # Assert that an exception is raised when trying to load predictions
    with pytest.raises(OSError):
        prediction_manager.load_predictions(video, "Walking")
