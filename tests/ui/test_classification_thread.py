"""Tests for ClassifyThread binary and multiclass branches."""

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    from jabs.core.enums import ClassifierMode, ProjectDistanceUnit
    from jabs.project.prediction_manager import MULTICLASS_PREDICTION_KEY
    from jabs.ui.classification_thread import ClassifyThread

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


class _FakeClassifier:
    """Simple classifier test double for ClassifyThread."""

    def __init__(self, multiclass: bool = False) -> None:
        self._multiclass = multiclass
        self.project_settings = {"window_size": 7}
        self.behavior_names = ["Walk", "Run"]

    @staticmethod
    def combine_data(per_frame, window):
        import pandas as pd

        return pd.concat([per_frame, window], axis=1)

    def predict_proba(self, data, frame_indexes):
        n = len(data)
        if self._multiclass:
            probs = np.zeros((n, 3), dtype=np.float32)
            probs[:, 0] = 0.1
            probs[:, 1] = 0.7
            probs[:, 2] = 0.2
            probs[frame_indexes == -1] = 0.0
            return probs

        probs = np.zeros((n, 2), dtype=np.float32)
        probs[:, 0] = 0.2
        probs[:, 1] = 0.8
        return probs

    def derive_predictions(self, probabilities):
        predictions = np.argmax(probabilities, axis=1).astype(np.int8)
        confidence = probabilities[np.arange(len(probabilities)), predictions].astype(np.float32)
        predictions[confidence == 0] = -1
        return predictions, confidence

    @staticmethod
    def get_class_names() -> list[str]:
        return ["background", "Walk", "Run"]


class _FakePose:
    """Pose-estimation test double."""

    def __init__(self) -> None:
        self.identities = [0]
        self.num_identities = 1
        self.num_frames = 5
        self.fps = 30


class _FakeProject:
    """Project test double for classification thread tests."""

    def __init__(self, mode: ClassifierMode) -> None:
        self.settings_manager = SimpleNamespace(
            classifier_mode=mode,
            get_behavior=lambda _behavior: {"window_size": 5, "postprocessing": []},
        )
        self.feature_manager = SimpleNamespace(distance_unit=ProjectDistanceUnit.PIXEL)
        self.video_manager = SimpleNamespace(
            videos=["video.avi"],
            video_path=lambda _video: "video.avi",
            num_videos=1,
        )
        self.feature_dir = "feature_dir"
        self.cache_format = "hdf5"
        self.save_predictions = MagicMock()
        self._pose = _FakePose()

    def load_pose_est(self, _video_path):
        return self._pose

    @staticmethod
    def get_project_defaults() -> dict:
        return {"window_size": 9}


def test_classify_thread_binary_path(monkeypatch) -> None:
    """Binary mode applies postprocessing and writes behavior-scoped predictions."""

    class _FakeIdentityFeatures:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def get_features(_window_size):
            return {
                "per_frame": {"a": np.arange(5, dtype=np.float32)},
                "window": {"b": np.arange(5, dtype=np.float32)},
                "frame_indexes": np.arange(5, dtype=np.intp),
            }

    class _FakePostprocessingPipeline:
        def __init__(self, _config):
            pass

        @staticmethod
        def run(predictions, _probabilities):
            return predictions.copy()

    monkeypatch.setattr("jabs.ui.classification_thread.IdentityFeatures", _FakeIdentityFeatures)
    monkeypatch.setattr(
        "jabs.ui.classify_strategy.PostprocessingPipeline",
        _FakePostprocessingPipeline,
    )

    project = _FakeProject(ClassifierMode.BINARY)
    classifier = _FakeClassifier(multiclass=False)
    thread = ClassifyThread(classifier, project, "Walk", "video.avi")
    completions: list[dict] = []
    errors: list[Exception] = []
    thread.classification_complete.connect(lambda output, _elapsed: completions.append(output))
    thread.error_callback.connect(errors.append)

    thread.run()

    assert errors == []
    assert len(completions) == 1
    output = completions[0]
    assert 0 in output["predictions"]
    assert 0 in output["probabilities"]
    assert output["probabilities"][0].ndim == 1
    assert 0 in output["predictions_postprocessed"]

    project.save_predictions.assert_called_once()
    args, kwargs = project.save_predictions.call_args
    assert args[4] == "Walk"
    assert kwargs["class_names"] is None
    assert kwargs["postprocessed_predictions"] is not None


def test_classify_thread_multiclass_path(monkeypatch) -> None:
    """Multiclass mode skips postprocessing and writes reserved-key predictions with class names."""

    class _FakeIdentityFeatures:
        op_settings_seen: ClassVar[list[dict]] = []

        def __init__(self, *_args, **kwargs):
            self.__class__.op_settings_seen.append(kwargs["op_settings"])

        @staticmethod
        def get_features(_window_size):
            return {
                "per_frame": {"a": np.arange(5, dtype=np.float32)},
                "window": {"b": np.arange(5, dtype=np.float32)},
                "frame_indexes": np.arange(5, dtype=np.intp),
            }

    class _PostprocessingMustNotRun:
        def __init__(self, _config):
            raise AssertionError(
                "PostprocessingPipeline should not be instantiated in multiclass mode"
            )

    monkeypatch.setattr("jabs.ui.classification_thread.IdentityFeatures", _FakeIdentityFeatures)
    monkeypatch.setattr(
        "jabs.ui.classify_strategy.PostprocessingPipeline",
        _PostprocessingMustNotRun,
    )

    project = _FakeProject(ClassifierMode.MULTICLASS)
    classifier = _FakeClassifier(multiclass=True)
    thread = ClassifyThread(classifier, project, "Walk", "video.avi")
    completions: list[dict] = []
    errors: list[Exception] = []
    thread.classification_complete.connect(lambda output, _elapsed: completions.append(output))
    thread.error_callback.connect(errors.append)

    thread.run()

    assert errors == []
    assert len(completions) == 1
    output = completions[0]
    assert 0 in output["predictions"]
    assert 0 in output["probabilities"]
    assert output["probabilities"][0].ndim == 2
    assert output["predictions_postprocessed"] == {}

    project.save_predictions.assert_called_once()
    args, kwargs = project.save_predictions.call_args
    assert args[4] == MULTICLASS_PREDICTION_KEY
    assert kwargs["class_names"] == ["None", "Walk", "Run"]
    assert kwargs["postprocessed_predictions"] == {}
    assert _FakeIdentityFeatures.op_settings_seen[0]["window_size"] == 7
