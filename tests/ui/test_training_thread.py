"""Tests for TrainingThread binary and multiclass training branches."""

import numpy as np
import pandas as pd
import pytest

from jabs.core.enums import ClassifierMode

from ._fakes import FakeTrainingClassifier, FakeTrainingProject

try:
    from jabs.ui.training_thread import TrainingThread

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


def test_training_thread_binary_path(monkeypatch, tmp_path) -> None:
    """Binary mode uses get_labeled_features + CV/report path and saves behavior-scoped classifier."""
    features = {
        "per_frame": pd.DataFrame({"feat_a": [1.0, 2.0]}),
        "window": pd.DataFrame({"feat_b": [3.0, 4.0]}),
        "labels": np.array([1, 0], dtype=np.int8),
        "groups": np.array([0, 1], dtype=np.int32),
    }
    project = FakeTrainingProject(tmp_path, ClassifierMode.BINARY, binary_features=features)
    classifier = FakeTrainingClassifier()

    cv_called = {"count": 0}

    def _fake_cv(**kwargs):
        cv_called["count"] += 1
        return []

    monkeypatch.setattr("jabs.ui.training_thread.run_leave_one_group_out_cv", _fake_cv)
    monkeypatch.setattr(
        "jabs.ui.training_thread.save_training_report", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "jabs.ui.training_thread.generate_markdown_report", lambda *_args, **_kwargs: "report"
    )

    thread = TrainingThread(classifier, project, "Walk", (3, 4), k=1)
    reports: list[str] = []
    errors: list[Exception] = []
    completions: list[int] = []
    thread.training_report.connect(reports.append)
    thread.error_callback.connect(errors.append)
    thread.training_complete.connect(completions.append)

    thread.run()

    assert errors == []
    assert project.binary_calls == 1
    assert project.multiclass_calls == 0
    assert cv_called["count"] == 1
    assert len(classifier.train_calls) == 1
    assert "training_data" in classifier.train_calls[0]
    assert "training_labels" in classifier.train_calls[0]
    project.save_classifier.assert_called_once_with(classifier, "Walk")
    assert reports == ["report"]
    assert len(completions) == 1
    project.session_tracker.classifier_trained.assert_called_once_with("Walk", "random_forest", 0)


def test_training_thread_multiclass_path(monkeypatch, tmp_path) -> None:
    """Multiclass mode runs multiclass CV/report and saves shared classifier."""
    features = {
        "per_frame": pd.DataFrame({"feat_a": [1.0, 2.0, 3.0]}),
        "window": pd.DataFrame({"feat_b": [0.1, 0.2, 0.3]}),
        "labels_by_behavior": {
            "None": np.array([1, 0, 0], dtype=np.int8),
            "Walk": np.array([0, 1, 0], dtype=np.int8),
            "Run": np.array([0, 0, 1], dtype=np.int8),
        },
        "groups": np.array([0, 0, 1], dtype=np.int32),
    }
    project = FakeTrainingProject(
        tmp_path, ClassifierMode.MULTICLASS, multiclass_features=features
    )
    classifier = FakeTrainingClassifier(name="catboost", project_settings={"window_size": 7})

    cv_called = {"count": 0}

    def _fake_cv(**kwargs):
        cv_called["count"] += 1
        return []

    monkeypatch.setattr("jabs.ui.training_thread.run_leave_one_group_out_cv", _fake_cv)
    monkeypatch.setattr(
        "jabs.ui.training_thread.save_training_report", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "jabs.ui.training_thread.generate_markdown_report", lambda *_args, **_kwargs: "report"
    )

    thread = TrainingThread(classifier, project, "Walk", (0, 0), k=5)
    reports: list[str] = []
    errors: list[Exception] = []
    completions: list[int] = []
    thread.training_report.connect(reports.append)
    thread.error_callback.connect(errors.append)
    thread.training_complete.connect(completions.append)

    thread.run()

    assert errors == []
    assert project.binary_calls == 0
    assert project.multiclass_calls == 1
    assert cv_called["count"] == 1
    assert len(classifier.train_calls) == 1
    assert "labels_by_behavior" in classifier.train_calls[0]
    assert "training_labels" not in classifier.train_calls[0]
    project.save_classifier.assert_called_once_with(classifier)
    assert reports == ["report"]
    assert len(completions) == 1
    project.session_tracker.classifier_trained.assert_called_once_with("Walk", "catboost", 0)


def _capture_cv_kwargs(monkeypatch) -> dict:
    """Patch the training thread's CV call and report writers, returning captured kwargs."""
    captured: dict = {}

    def _fake_cv(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("jabs.ui.training_thread.run_leave_one_group_out_cv", _fake_cv)
    monkeypatch.setattr(
        "jabs.ui.training_thread.save_training_report", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "jabs.ui.training_thread.generate_markdown_report", lambda *_args, **_kwargs: "report"
    )
    return captured


def _binary_features() -> dict:
    """Return a minimal binary feature payload."""
    return {
        "per_frame": pd.DataFrame({"feat_a": [1.0, 2.0]}),
        "window": pd.DataFrame({"feat_b": [3.0, 4.0]}),
        "labels": np.array([1, 0], dtype=np.int8),
        "groups": np.array([0, 1], dtype=np.int32),
    }


_STITCH_STAGE = {
    "stage_name": "BoutStitchingStage",
    "enabled": True,
    "parameters": {"max_stitch_gap": 3},
}


def test_training_thread_forwards_postprocessing_setting(monkeypatch, tmp_path) -> None:
    """The behavior's saved flag reaches CV, and the stages reach the report."""
    project = FakeTrainingProject(
        tmp_path, ClassifierMode.BINARY, binary_features=_binary_features()
    )
    project.settings_manager.evaluate_postprocessing_in_cv = lambda _behavior: True
    project.settings_manager.postprocessing_config = lambda _behavior: [_STITCH_STAGE]

    captured = _capture_cv_kwargs(monkeypatch)
    report_data: list = []
    monkeypatch.setattr(
        "jabs.ui.training_thread.generate_markdown_report",
        lambda data: report_data.append(data) or "report",
    )

    errors: list[Exception] = []
    thread = TrainingThread(FakeTrainingClassifier(), project, "Walk", (3, 4), k=1)
    thread.error_callback.connect(errors.append)

    thread.run()

    assert errors == []
    assert captured["evaluate_postprocessing"] is True
    assert report_data[0].postprocessing_stages == [_STITCH_STAGE]


def test_training_thread_omits_postprocessing_when_disabled(monkeypatch, tmp_path) -> None:
    """With the flag off, CV is not asked to evaluate and the report stays silent."""
    project = FakeTrainingProject(
        tmp_path, ClassifierMode.BINARY, binary_features=_binary_features()
    )
    captured = _capture_cv_kwargs(monkeypatch)
    report_data: list = []
    monkeypatch.setattr(
        "jabs.ui.training_thread.generate_markdown_report",
        lambda data: report_data.append(data) or "report",
    )

    thread = TrainingThread(FakeTrainingClassifier(), project, "Walk", (3, 4), k=1)
    thread.run()

    assert captured["evaluate_postprocessing"] is False
    assert report_data[0].postprocessing_stages is None


def test_training_thread_skips_postprocessing_in_multiclass_mode(monkeypatch, tmp_path) -> None:
    """Postprocessing is binary-only, so multi-class training never requests it."""
    features = {
        "per_frame": pd.DataFrame({"feat_a": [1.0, 2.0]}),
        "window": pd.DataFrame({"feat_b": [3.0, 4.0]}),
        "labels_by_behavior": {"Walk": np.array([1, 0], dtype=np.int8)},
        "groups": np.array([0, 1], dtype=np.int32),
    }
    project = FakeTrainingProject(
        tmp_path, ClassifierMode.MULTICLASS, multiclass_features=features
    )
    # even with the flag set, multi-class must not ask for the evaluation
    project.settings_manager.evaluate_postprocessing_in_cv = lambda _behavior: True
    project.settings_manager.postprocessing_config = lambda _behavior: [_STITCH_STAGE]

    captured = _capture_cv_kwargs(monkeypatch)

    errors: list[Exception] = []
    thread = TrainingThread(FakeTrainingClassifier(), project, "Walk", (3, 4), k=1)
    thread.error_callback.connect(errors.append)

    thread.run()

    assert errors == []
    assert captured["evaluate_postprocessing"] is False
