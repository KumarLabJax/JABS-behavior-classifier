import pytest
import h5py
import numpy as np
from unittest.mock import MagicMock
from src.jabs.project.prediction_manager import PredictionManager


@pytest.fixture
def mock_project(tmp_path):
    """Fixture to create a mock project with necessary paths."""
    project = MagicMock()
    project.project_paths.prediction_dir = tmp_path / "predictions"
    project.project_paths.prediction_dir.mkdir(parents=True)
    project._metadata = {
        "video_files": {
            "test_video.avi": {"identities": 2}
        }
    }
    return project


@pytest.fixture
def prediction_manager(mock_project):
    """Fixture to create a PredictionManager instance."""
    return PredictionManager(mock_project)


def test_write_predictions(prediction_manager, tmp_path):
    """Test writing predictions to an HDF5 file."""
    output_path = tmp_path / "test_predictions.h5"
    behavior = "Walking"
    predictions = np.array([[1, 0, -1], [0, 1, -1]])
    probabilities = np.array([[0.9, 0.8, -1], [0.7, 0.6, -1]])
    poses = MagicMock(pose_file="pose_file.h5", hash="12345", identity_to_track=None)
    classifier = MagicMock(classifier_file="classifier.pkl", classifier_hash="67890")

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

    # Create a mock prediction file
    with h5py.File(prediction_file, "w") as h5:
        h5.attrs["version"] = 2
        prediction_group = h5.create_group("predictions")
        behavior_group = prediction_group.create_group(behavior)
        behavior_group.create_dataset("predicted_class", data=[[1, 0, -1], [0, 1, -1]])
        behavior_group.create_dataset("probabilities", data=[[0.9, 0.8, -1], [0.7, 0.6, -1]])

    predictions, probabilities, frame_indexes = prediction_manager.load_predictions(video, behavior)

    assert "0" in predictions
    assert "1" in predictions
    assert np.array_equal(predictions["0"], [1, 0, -1])
    assert np.array_equal(probabilities["0"], [0.9, 0.8, -1])
    assert np.array_equal(frame_indexes["0"], [0, 1])


def test_load_predictions_missing_behavior(prediction_manager, mock_project):
    """Test loading predictions when behavior is missing."""
    video = "test_video.avi"
    behavior = "Running"
    prediction_file = mock_project.project_paths.prediction_dir / "test_video.h5"

    # Create a mock prediction file without the behavior
    with h5py.File(prediction_file, "w") as h5:
        h5.attrs["version"] = 2
        h5.create_group("predictions")

    predictions, probabilities, frame_indexes = prediction_manager.load_predictions(video, behavior)

    assert predictions == {}
    assert probabilities == {}
    assert frame_indexes == {}


def test_load_predictions_invalid_file(prediction_manager, mock_project):
    """Test loading predictions from an invalid file."""
    video = "test_video.avi"
    prediction_file = mock_project.project_paths.prediction_dir / "test_video.h5"

    # Create an invalid prediction file
    with open(prediction_file, "w") as f:
        f.write("invalid content")

    predictions, probabilities, frame_indexes = prediction_manager.load_predictions(video, "Walking")

    assert predictions == {}
    assert probabilities == {}
    assert frame_indexes == {}