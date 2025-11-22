from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.io.prediction import PREDICTION_FILE_VERSION, save_predictions
from jabs.project.project_utils import to_safe_name
from jabs.version import version_str


class DummyClassifier:
    """Minimal classifier stub providing metadata attributes used by save_predictions."""

    def __init__(
        self, classifier_file: str = "classifier.pkl", classifier_hash: str = "abc123"
    ) -> None:
        self.classifier_file = classifier_file
        self.classifier_hash = classifier_hash


def _create_dummy_data(num_ids: int = 2, num_frames: int = 5):
    predictions = np.arange(num_ids * num_frames, dtype=np.int8).reshape(num_ids, num_frames)
    probabilities = np.linspace(0.0, 1.0, num_ids * num_frames, dtype=float).reshape(
        num_ids, num_frames
    )
    pose_identity_to_track = np.arange(num_ids, dtype=np.int32)
    return predictions, probabilities, pose_identity_to_track


def test_save_predictions_creates_file_and_datasets(tmp_path: Path) -> None:
    """Test that save_predictions creates the file and expected datasets."""
    output_path = tmp_path / "predictions.h5"
    behavior = "Some Behavior"
    predictions, probabilities, pose_identity_to_track = _create_dummy_data()
    external_identities = ["id_0", "id_1"]
    classifier = DummyClassifier()
    pose_file = "pose_file.h5"
    pose_hash = "pose-hash-123"

    save_predictions(
        output_path=output_path,
        predictions=predictions,
        probabilities=probabilities,
        behavior=behavior,
        classifier=classifier,
        pose_file=pose_file,
        pose_hash=pose_hash,
        pose_identity_to_track=pose_identity_to_track,
        external_identities=external_identities,
    )

    assert output_path.exists()

    with h5py.File(output_path, "r") as h5:
        # root-level metadata
        assert h5.attrs["pose_file"] == pose_file
        assert h5.attrs["pose_hash"] == pose_hash
        assert h5.attrs["version"] == PREDICTION_FILE_VERSION

        prediction_group = h5["predictions"]
        behavior_group = prediction_group[to_safe_name(behavior)]

        # classifier + app metadata
        assert behavior_group.attrs["classifier_file"] == classifier.classifier_file
        assert behavior_group.attrs["classifier_hash"] == classifier.classifier_hash
        assert behavior_group.attrs["app_version"] == version_str()
        assert "prediction_date" in behavior_group.attrs

        # datasets
        np.testing.assert_array_equal(behavior_group["predicted_class"][...], predictions)
        np.testing.assert_array_equal(behavior_group["probabilities"][...], probabilities)
        np.testing.assert_array_equal(
            behavior_group["identity_to_track"][...], pose_identity_to_track
        )

        # external identity mapping written once
        mapping = prediction_group["external_identity_mapping"][...].astype(str)
        assert mapping.tolist() == external_identities


def test_save_predictions_validates_external_identities_length(tmp_path: Path) -> None:
    """Test that save_predictions raises ValueError on identity length mismatch."""
    output_path = tmp_path / "predictions.h5"
    behavior = "Behavior"
    predictions, probabilities, pose_identity_to_track = _create_dummy_data(num_ids=2)
    external_identities = ["id_0", "id_1", "id_2"]  # length mismatch

    classifier = DummyClassifier()

    with pytest.raises(ValueError):
        save_predictions(
            output_path=output_path,
            predictions=predictions,
            probabilities=probabilities,
            behavior=behavior,
            classifier=classifier,
            pose_file="pose_file.h5",
            pose_hash="pose-hash-123",
            pose_identity_to_track=pose_identity_to_track,
            external_identities=external_identities,
        )


def test_external_identity_mapping_not_overwritten_on_subsequent_calls(tmp_path: Path) -> None:
    """Test that external identity mapping is not overwritten on subsequent saves."""
    output_path = tmp_path / "predictions.h5"
    behavior = "Behavior"
    classifier = DummyClassifier()

    # First call
    predictions1, probabilities1, pose_identity_to_track1 = _create_dummy_data(
        num_ids=2, num_frames=4
    )
    external_identities1 = ["id_a", "id_b"]

    save_predictions(
        output_path=output_path,
        predictions=predictions1,
        probabilities=probabilities1,
        behavior=behavior,
        classifier=classifier,
        pose_file="pose_file_1.h5",
        pose_hash="pose-hash-1",
        pose_identity_to_track=pose_identity_to_track1,
        external_identities=external_identities1,
    )

    # Second call with different mapping but same shape
    predictions2, probabilities2, pose_identity_to_track2 = _create_dummy_data(
        num_ids=2, num_frames=4
    )
    external_identities2 = ["id_c", "id_d"]

    save_predictions(
        output_path=output_path,
        predictions=predictions2,
        probabilities=probabilities2,
        behavior=behavior,
        classifier=classifier,
        pose_file="pose_file_2.h5",
        pose_hash="pose-hash-2",
        pose_identity_to_track=pose_identity_to_track2,
        external_identities=external_identities2,
    )

    with h5py.File(output_path, "r") as h5:
        prediction_group = h5["predictions"]
        behavior_group = prediction_group[to_safe_name(behavior)]

        # mapping should remain from the first call
        mapping = prediction_group["external_identity_mapping"][...].astype(str)
        assert mapping.tolist() == external_identities1

        # predictions should reflect the most recent call
        np.testing.assert_array_equal(behavior_group["predicted_class"][...], predictions2)
        np.testing.assert_array_equal(behavior_group["probabilities"][...], probabilities2)


def test_identity_to_track_removed_when_none(tmp_path: Path) -> None:
    """Test that identity_to_track is removed when None is provided."""
    output_path = tmp_path / "predictions.h5"
    behavior = "Behavior"
    classifier = DummyClassifier()

    predictions, probabilities, pose_identity_to_track = _create_dummy_data(
        num_ids=2, num_frames=3
    )
    external_identities = ["id_0", "id_1"]

    # First call with identity_to_track present
    save_predictions(
        output_path=output_path,
        predictions=predictions,
        probabilities=probabilities,
        behavior=behavior,
        classifier=classifier,
        pose_file="pose_file_1.h5",
        pose_hash="pose-hash-1",
        pose_identity_to_track=pose_identity_to_track,
        external_identities=external_identities,
    )

    # Second call removes identity_to_track
    save_predictions(
        output_path=output_path,
        predictions=predictions,
        probabilities=probabilities,
        behavior=behavior,
        classifier=classifier,
        pose_file="pose_file_2.h5",
        pose_hash="pose-hash-2",
        pose_identity_to_track=None,
        external_identities=external_identities,
    )

    with h5py.File(output_path, "r") as h5:
        prediction_group = h5["predictions"]
        behavior_group = prediction_group[to_safe_name(behavior)]

        assert "identity_to_track" not in behavior_group
