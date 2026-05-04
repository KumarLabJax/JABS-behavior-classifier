"""Tests for TrainingThread binary and multiclass training branches."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

try:
    from jabs.core.enums import (
        ClassifierMode,
        CrossValidationGroupingStrategy,
        ProjectDistanceUnit,
    )
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


class _FakeClassifier:
    """Small classifier test double for TrainingThread."""

    def __init__(self, name: str = "random_forest", project_settings: dict | None = None) -> None:
        self.classifier_name = name
        self.project_settings = {} if project_settings is None else dict(project_settings)
        self.behavior_names = ["Walk", "Run"]
        self.train_calls: list[dict] = []

    @staticmethod
    def combine_data(per_frame: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([per_frame, window], axis=1)

    def train(self, data: dict, random_seed: int | None = None) -> None:
        call = dict(data)
        call["random_seed"] = random_seed
        self.train_calls.append(call)

    @staticmethod
    def get_feature_importance(limit: int = 20) -> list[tuple[str, float]]:
        return [("feat_a", 1.0)][:limit]


class _FakeProject:
    """Project test double providing only the APIs TrainingThread uses."""

    def __init__(
        self,
        tmp_path,
        mode: ClassifierMode,
        binary_features: dict | None = None,
        multiclass_features: dict | None = None,
    ) -> None:
        self.project_paths = SimpleNamespace(training_log_dir=tmp_path)
        self.feature_manager = SimpleNamespace(distance_unit=ProjectDistanceUnit.PIXEL)
        self.session_tracker = SimpleNamespace(classifier_trained=MagicMock())
        self.settings_manager = SimpleNamespace(
            classifier_mode=mode,
            cv_grouping_strategy=CrossValidationGroupingStrategy.INDIVIDUAL,
            get_behavior=lambda _behavior: {
                "window_size": 5,
                "balance_labels": False,
                "symmetric_behavior": False,
            },
        )
        self._binary_features = binary_features
        self._multiclass_features = multiclass_features
        self.binary_calls = 0
        self.multiclass_calls = 0
        self.save_classifier = MagicMock()

    def get_project_defaults(self) -> dict:
        return {
            "window_size": 5,
            "balance_labels": False,
            "symmetric_behavior": False,
        }

    def get_labeled_features(
        self,
        behavior: str,
        progress_callable=None,
        should_terminate_callable=None,
    ) -> tuple[dict, dict]:
        self.binary_calls += 1
        if should_terminate_callable is not None:
            should_terminate_callable()
        if progress_callable is not None:
            progress_callable()
        return self._binary_features, {0: {"video": "video.avi", "identity": 0}}

    def get_multiclass_labeled_features(
        self,
        progress_callable=None,
        should_terminate_callable=None,
    ) -> tuple[dict, dict]:
        self.multiclass_calls += 1
        if should_terminate_callable is not None:
            should_terminate_callable()
        if progress_callable is not None:
            progress_callable()
        return self._multiclass_features, {}

    @staticmethod
    def counts(_behavior: str) -> dict:
        return {
            "video.avi": {
                0: {
                    "unfragmented_bout_counts": (1, 0),
                }
            }
        }


def test_training_thread_binary_path(monkeypatch, tmp_path) -> None:
    """Binary mode uses get_labeled_features + CV/report path and saves behavior-scoped classifier."""
    features = {
        "per_frame": pd.DataFrame({"feat_a": [1.0, 2.0]}),
        "window": pd.DataFrame({"feat_b": [3.0, 4.0]}),
        "labels": np.array([1, 0], dtype=np.int8),
        "groups": np.array([0, 1], dtype=np.int32),
    }
    project = _FakeProject(tmp_path, ClassifierMode.BINARY, binary_features=features)
    classifier = _FakeClassifier()

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
    project = _FakeProject(tmp_path, ClassifierMode.MULTICLASS, multiclass_features=features)
    classifier = _FakeClassifier(name="catboost", project_settings={"window_size": 7})

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
