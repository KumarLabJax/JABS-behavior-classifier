"""Tests for ClassifyThread binary and multiclass branches."""

import pytest

from jabs.core.enums import ClassifierMode
from jabs.project.prediction_manager import MULTICLASS_PREDICTION_KEY

from ._fakes import (
    FakeClassifyingClassifier,
    FakeClassifyingProject,
    make_fake_identity_features,
)

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

    class _FakePostprocessingPipeline:
        def __init__(self, _config):
            pass

        @staticmethod
        def run(predictions, _probabilities):
            return predictions.copy()

    monkeypatch.setattr(
        "jabs.ui.classification_thread.IdentityFeatures",
        make_fake_identity_features(),
    )
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

    class _PostprocessingMustNotRun:
        def __init__(self, _config):
            raise AssertionError(
                "PostprocessingPipeline should not be instantiated in multiclass mode"
            )

    fake_features_cls = make_fake_identity_features()
    monkeypatch.setattr("jabs.ui.classification_thread.IdentityFeatures", fake_features_cls)
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
    assert fake_features_cls.op_settings_seen[0]["window_size"] == 7


def _binary_thread_env(monkeypatch) -> None:
    """Patch IdentityFeatures / PostprocessingPipeline for a binary ClassifyThread run."""

    class _FakePostprocessingPipeline:
        def __init__(self, _config):
            pass

        @staticmethod
        def run(predictions, _probabilities):
            return predictions.copy()

    monkeypatch.setattr(
        "jabs.ui.classification_thread.IdentityFeatures",
        make_fake_identity_features(),
    )
    monkeypatch.setattr(
        "jabs.ui.classify_strategy.PostprocessingPipeline",
        _FakePostprocessingPipeline,
    )


def test_classify_thread_classifies_only_the_video_subset(monkeypatch) -> None:
    """A ``videos`` subset drives iteration instead of the project's video list."""
    _binary_thread_env(monkeypatch)

    project = FakeClassifyingProject(ClassifierMode.BINARY)
    classifier = FakeClassifyingClassifier(multiclass=False)
    # the project only lists "video.avi", but we ask for a different two-video set
    thread = ClassifyThread(classifier, project, "Walk", "a.avi", videos=["a.avi", "b.avi"])
    errors: list[Exception] = []
    thread.error_callback.connect(errors.append)

    thread.run()

    assert errors == []
    # one save per requested video, keyed by the subset names (not the project list)
    saved_videos = [call.args[1] for call in project.save_predictions.call_args_list]
    assert saved_videos == ["a.avi", "b.avi"]


def test_classify_thread_empty_subset_saves_nothing(monkeypatch) -> None:
    """An empty ``videos`` subset classifies no videos but still completes."""
    _binary_thread_env(monkeypatch)

    project = FakeClassifyingProject(ClassifierMode.BINARY)
    classifier = FakeClassifyingClassifier(multiclass=False)
    thread = ClassifyThread(classifier, project, "Walk", "video.avi", videos=[])
    completions: list[dict] = []
    errors: list[Exception] = []
    thread.classification_complete.connect(lambda output, _elapsed: completions.append(output))
    thread.error_callback.connect(errors.append)

    thread.run()

    assert errors == []
    project.save_predictions.assert_not_called()
    # completion still fires (with no current-video predictions)
    assert len(completions) == 1
    assert completions[0]["predictions"] == {}
