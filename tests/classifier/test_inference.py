"""Tests for full-sequence per-identity inference."""

import numpy as np
import pandas as pd

from jabs.classifier.inference import predict_identity


class _FakeIdentityFeatures:
    """Stand-in for ``IdentityFeatures`` returning fixed full-video features."""

    def __init__(self, per_frame: dict, window: dict, frame_indexes: np.ndarray) -> None:
        self._features = {
            "per_frame": per_frame,
            "window": window,
            "frame_indexes": frame_indexes,
        }
        self.requested_window_sizes: list[int] = []

    def get_features(self, window_size: int) -> dict:
        """Record the requested window size and return the fixed features."""
        self.requested_window_sizes.append(window_size)
        return self._features


class _FakeClassifier:
    """Classifier stand-in whose ``predict_proba`` zeroes out excluded frames."""

    def __init__(self, probabilities: np.ndarray) -> None:
        self._probabilities = probabilities
        self.predict_proba_calls: list[np.ndarray] = []

    @staticmethod
    def combine_data(per_frame: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([per_frame, window], axis=1)

    def predict_proba(self, features: pd.DataFrame, frame_indexes=None) -> np.ndarray:
        self.predict_proba_calls.append(frame_indexes)
        result = np.zeros(self._probabilities.shape, dtype=np.float32)
        result[frame_indexes] = self._probabilities[frame_indexes]
        return result

    @staticmethod
    def derive_predictions(probabilities: np.ndarray):
        predictions = np.argmax(probabilities, axis=1).astype(np.int8)
        confidence = probabilities[np.arange(len(probabilities)), predictions]
        predictions[confidence == 0] = -1
        return predictions, confidence


def test_predict_identity_returns_full_length_arrays() -> None:
    """Predictions span every frame, with -1 where the identity has no pose."""
    # frame 2 is excluded from frame_indexes, standing in for a missing identity
    frame_indexes = np.array([0, 1, 3], dtype=np.intp)
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
    features = _FakeIdentityFeatures(
        per_frame={"a": np.arange(4, dtype=np.float64)},
        window={"b": np.arange(4, dtype=np.float64)},
        frame_indexes=frame_indexes,
    )
    classifier = _FakeClassifier(probabilities)

    result = predict_identity(classifier, features, window_size=5)

    assert result is not None
    assert result.predictions.tolist() == [0, 1, -1, 1]
    assert result.confidence[2] == 0.0
    assert result.probabilities.shape == (4, 2)
    assert features.requested_window_sizes == [5]
    np.testing.assert_array_equal(classifier.predict_proba_calls[0], frame_indexes)


def test_predict_identity_returns_none_without_feature_rows() -> None:
    """An identity with no feature rows yields None so callers can zero-fill."""
    features = _FakeIdentityFeatures(
        per_frame={},
        window={},
        frame_indexes=np.array([], dtype=np.intp),
    )
    classifier = _FakeClassifier(np.zeros((0, 2), dtype=np.float32))

    assert predict_identity(classifier, features, window_size=5) is None
