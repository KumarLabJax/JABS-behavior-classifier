import numpy as np
import pytest

from jabs.core.types.prediction import BehaviorPrediction, ClassifierMetadata


@pytest.fixture()
def classifier_metadata() -> ClassifierMetadata:
    """Return minimal classifier metadata for prediction type tests."""
    return ClassifierMetadata(
        classifier_file="classifier.pickle",
        classifier_hash="abc123",
        app_version="1.0.0",
        prediction_date="2026-04-29T00:00:00",
    )


def _make_prediction(
    classifier_metadata: ClassifierMetadata,
    probabilities: np.ndarray,
    class_names: list[str] | None = None,
) -> BehaviorPrediction:
    return BehaviorPrediction(
        behavior="Walking",
        predicted_class=np.array([[0, 1, -1], [1, 0, -1]], dtype=np.int8),
        probabilities=probabilities,
        classifier=classifier_metadata,
        pose_file="video_pose_est_v6.h5",
        pose_hash="posehash",
        class_names=class_names,
    )


def test_binary_probabilities_match_predicted_class_shape(
    classifier_metadata: ClassifierMetadata,
) -> None:
    """Binary predictions keep the legacy 2-D probability shape."""
    probabilities = np.array([[0.9, 0.8, 0.0], [0.7, 0.6, 0.0]], dtype=np.float32)

    prediction = _make_prediction(classifier_metadata, probabilities)

    np.testing.assert_array_equal(prediction.probabilities, probabilities)
    assert prediction.class_names is None


def test_multiclass_probabilities_match_class_names(
    classifier_metadata: ClassifierMetadata,
) -> None:
    """Multi-class predictions allow one probability column per class."""
    class_names = ["background", "Walking", "Rearing"]
    probabilities = np.zeros((2, 3, len(class_names)), dtype=np.float32)
    probabilities[:, :, 0] = 0.2
    probabilities[:, :, 1] = 0.6
    probabilities[:, :, 2] = 0.2

    prediction = _make_prediction(classifier_metadata, probabilities, class_names=class_names)

    np.testing.assert_array_equal(prediction.probabilities, probabilities)
    assert prediction.class_names == class_names


def test_multiclass_probability_shape_must_match_class_names(
    classifier_metadata: ClassifierMetadata,
) -> None:
    """Multi-class probability shape must include exactly len(class_names) classes."""
    probabilities = np.zeros((2, 3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="probabilities shape"):
        _make_prediction(
            classifier_metadata,
            probabilities,
            class_names=["background", "Walking", "Rearing"],
        )


def test_binary_probability_shape_rejects_multiclass_array_without_class_names(
    classifier_metadata: ClassifierMetadata,
) -> None:
    """3-D probabilities require class_names so legacy binary shape checks remain strict."""
    probabilities = np.zeros((2, 3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="probabilities shape"):
        _make_prediction(classifier_metadata, probabilities)
