"""Tests for ClassifyThread binary and multiclass branches."""

from typing import ClassVar

import numpy as np
import pytest

from jabs.core.enums import ClassifierMode
from jabs.project.prediction_manager import MULTICLASS_PREDICTION_KEY

from ._fakes import FakeClassifyingClassifier, FakeClassifyingProject

try:
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

    project = FakeClassifyingProject(ClassifierMode.BINARY)
    classifier = FakeClassifyingClassifier(multiclass=False)
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

    project = FakeClassifyingProject(ClassifierMode.MULTICLASS)
    classifier = FakeClassifyingClassifier(multiclass=True)
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
