"""Tests for cross-validation helpers."""

import logging

import numpy as np
import pandas as pd
import pytest

from jabs.classifier import cross_validation
from jabs.classifier.cross_validation import (
    NO_VALID_SPLITS_WARNING,
    run_leave_one_group_out_cv,
)


class _NoSplitClassifier:
    """Classifier test double reporting no valid LOGO splits."""

    @staticmethod
    def get_leave_one_group_out_max(_labels, _groups, _excluded_groups=None) -> int:
        return 0

    @staticmethod
    def leave_one_group_out(*_args, **_kwargs):
        raise AssertionError("leave_one_group_out should not be called when max splits is zero")


class _MultiClassCVClassifier:
    """Minimal multiclass test double for CV settings behavior."""

    def __init__(self):
        self.behavior_names = ["Walk"]
        self._project_settings = {"window_size": 123, "balance_labels": True}
        self.set_project_settings_calls = 0
        self.train_settings: list[dict] = []

    @property
    def project_settings(self) -> dict:
        return dict(self._project_settings)

    @staticmethod
    def merge_labels(_labels_by_behavior, _behavior_names):
        return np.array([0, 1, 0, 1], dtype=np.int8), np.array([True, True, True, True])

    @staticmethod
    def get_leave_one_group_out_max(_labels, _groups, _excluded_groups=None) -> int:
        return 1

    @staticmethod
    def leave_one_group_out(*_args, **_kwargs):
        test_labels = np.array([0, 1], dtype=np.int8)
        yield {
            "test_group": 1,
            "training_idx": np.array([0, 1], dtype=np.intp),
            "test_data": pd.DataFrame({"f": [3.0, 4.0]}),
            "test_labels": test_labels,
            "feature_names": ["f"],
        }

    def set_project_settings(self, _project, _behavior=None) -> None:
        self.set_project_settings_calls += 1

    def train(self, data: dict) -> None:
        self.train_settings.append(dict(data["settings"]))

    @staticmethod
    def predict(_test_data):
        return np.array([0, 1], dtype=np.int8)

    @staticmethod
    def get_feature_importance(limit=10):
        return []


class _EmptyMultiClassClassifier:
    """Multiclass test double reporting no valid splits (no labeled frames)."""

    def __init__(self):
        self.behavior_names = ["Walk"]

    @property
    def project_settings(self) -> dict:
        return {"window_size": 5}

    @staticmethod
    def get_leave_one_group_out_max(_labels, _groups, _excluded_groups=None) -> int:
        return 0

    @staticmethod
    def leave_one_group_out(*_args, **_kwargs):
        raise AssertionError("leave_one_group_out should not be called when max splits is zero")


def test_multiclass_cv_skips_when_no_labels() -> None:
    """Empty labels_by_behavior should skip CV gracefully rather than raise.

    merge_labels() raises on an empty dict; _prepare_cv_labels must short-circuit
    so the multiclass path mirrors the binary "no valid splits" behavior.
    """
    features = {
        "per_frame": pd.DataFrame({"a": []}),
        "window": pd.DataFrame({"b": []}),
        "groups": np.array([], dtype=np.int32),
        "labels_by_behavior": {},
    }
    status_messages: list[str] = []
    results = run_leave_one_group_out_cv(
        classifier=_EmptyMultiClassClassifier(),
        project=type("P", (), {"get_project_defaults": lambda self: {"window_size": 5}})(),
        features=features,
        group_mapping={},
        behavior="Walk",
        k=1,
        status_callback=status_messages.append,
    )

    assert results == []
    assert any("skipping CV" in msg for msg in status_messages)


def test_run_leave_one_group_out_cv_returns_empty_when_no_valid_splits() -> None:
    """No valid CV splits should not raise; CV is skipped with empty results."""
    features = {
        "per_frame": pd.DataFrame({"a": [1.0, 2.0]}),
        "window": pd.DataFrame({"b": [3.0, 4.0]}),
        "labels": np.array([0, 1], dtype=np.int8),
        "groups": np.array([0, 1], dtype=np.int32),
    }
    status_messages: list[str] = []
    results = run_leave_one_group_out_cv(
        classifier=_NoSplitClassifier(),
        project=object(),
        features=features,
        group_mapping={},
        behavior="Walk",
        k=1,
        status_callback=status_messages.append,
    )

    assert results == []
    assert any("skipping CV" in msg for msg in status_messages)


def test_no_valid_splits_reports_a_warning() -> None:
    """Requested-but-impossible CV is reported through warning_callback.

    The status message scrolls past in the status bar; the warning is what the GUI
    and CLI surface so the user knows why the report has no CV metrics.
    """
    features = {
        "per_frame": pd.DataFrame({"a": [1.0, 2.0]}),
        "window": pd.DataFrame({"b": [3.0, 4.0]}),
        "labels": np.array([0, 1], dtype=np.int8),
        "groups": np.array([0, 1], dtype=np.int32),
    }
    warnings: list[str] = []
    run_leave_one_group_out_cv(
        classifier=_NoSplitClassifier(),
        project=object(),
        features=features,
        group_mapping={},
        behavior="Walk",
        k=1,
        warning_callback=warnings.append,
    )

    assert warnings == [NO_VALID_SPLITS_WARNING]


def test_no_warning_when_cross_validation_was_not_requested() -> None:
    """k=0 means the user turned cross-validation off, which is not a problem to report."""
    features = {
        "per_frame": pd.DataFrame({"a": [1.0, 2.0]}),
        "window": pd.DataFrame({"b": [3.0, 4.0]}),
        "labels": np.array([0, 1], dtype=np.int8),
        "groups": np.array([0, 1], dtype=np.int32),
    }
    warnings: list[str] = []
    results = run_leave_one_group_out_cv(
        classifier=_NoSplitClassifier(),
        project=object(),
        features=features,
        group_mapping={},
        behavior="Walk",
        k=0,
        warning_callback=warnings.append,
    )

    assert results == []
    assert warnings == []


def test_multiclass_cv_reuses_classifier_settings_without_resetting() -> None:
    """Multiclass CV should not overwrite persisted classifier settings per fold."""
    features = {
        "per_frame": pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}),
        "window": pd.DataFrame({"b": [5.0, 6.0, 7.0, 8.0]}),
        "groups": np.array([0, 0, 1, 1], dtype=np.int32),
        "labels_by_behavior": {
            "None": np.array([1, 0, 1, 0], dtype=np.int8),
            "Walk": np.array([0, 1, 0, 1], dtype=np.int8),
        },
    }
    classifier = _MultiClassCVClassifier()

    run_leave_one_group_out_cv(
        classifier=classifier,
        project=type("P", (), {"get_project_defaults": lambda self: {"window_size": 5}})(),
        features=features,
        group_mapping={1: {"video": "v1.avi", "identity": "0"}},
        behavior="Walk",
        k=1,
    )

    assert classifier.set_project_settings_calls == 0
    assert classifier.train_settings == [{"window_size": 123, "balance_labels": True}]


class _BinaryCVClassifier:
    """Minimal binary test double yielding a single valid LOGO split."""

    def __init__(self) -> None:
        self.train_calls = 0

    @staticmethod
    def get_leave_one_group_out_max(_labels, _groups, _excluded_groups=None) -> int:
        return 1

    @staticmethod
    def leave_one_group_out(*_args, **_kwargs):
        yield {
            "test_group": 1,
            "training_idx": np.array([0, 1], dtype=np.intp),
            "test_data": pd.DataFrame({"f": [3.0, 4.0]}),
            "test_labels": np.array([0, 1], dtype=np.int8),
            "test_idx": np.array([2, 3], dtype=np.intp),
            "feature_names": ["f"],
        }

    def set_project_settings(self, _project, _behavior=None) -> None:
        pass

    def train(self, _data) -> None:
        self.train_calls += 1

    @staticmethod
    def predict(_test_data):
        return np.array([0, 1], dtype=np.int8)

    @staticmethod
    def get_feature_importance(limit=10):
        return []


def _binary_features() -> dict:
    """Return a small binary feature payload with two CV groups."""
    return {
        "per_frame": pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}),
        "window": pd.DataFrame({"b": [5.0, 6.0, 7.0, 8.0]}),
        "labels": np.array([0, 1, 0, 1], dtype=np.int8),
        "groups": np.array([0, 0, 1, 1], dtype=np.int32),
    }


class _FakeSettingsManager:
    """Settings manager stand-in exposing only what CV reads."""

    def __init__(self, postprocessing: list[dict], window_size: int = 5) -> None:
        self._postprocessing = postprocessing
        self._window_size = window_size

    def postprocessing_config(self, _behavior: str) -> list[dict]:
        return list(self._postprocessing)

    def get_behavior(self, _behavior: str) -> dict:
        return {"window_size": self._window_size, "postprocessing": self._postprocessing}


class _FakeProject:
    """Project stand-in carrying a settings manager."""

    def __init__(self, settings_manager: _FakeSettingsManager) -> None:
        self.settings_manager = settings_manager

    @staticmethod
    def get_project_defaults() -> dict:
        return {"window_size": 5}


_STITCH_CONFIG: list[dict] = [
    {
        "stage_name": "BoutStitchingStage",
        "enabled": True,
        "parameters": {"max_stitch_gap": 2},
    }
]


def test_postprocessing_evaluation_attaches_metrics(monkeypatch) -> None:
    """When enabled, each binary fold carries postprocessed metrics alongside raw."""
    # postprocessing recovers the frame raw got wrong
    evaluation = cross_validation.FoldPostprocessingEvaluation(
        truth=np.array([0, 1, 1, 1], dtype=np.int8),
        raw=np.array([0, 1, 1, 0], dtype=np.int8),
        postprocessed=np.array([0, 1, 1, 1], dtype=np.int8),
    )
    captured: dict = {}

    def _fake_evaluate(**kwargs):
        captured.update(kwargs)
        return evaluation

    monkeypatch.setattr(cross_validation, "evaluate_group_with_postprocessing", _fake_evaluate)

    results = run_leave_one_group_out_cv(
        classifier=_BinaryCVClassifier(),
        project=_FakeProject(_FakeSettingsManager(_STITCH_CONFIG)),
        features=_binary_features(),
        group_mapping={1: {"video": "v1.avi", "identity": 0, "members": [("v1.avi", 0)]}},
        behavior="Walk",
        k=1,
        evaluate_postprocessing=True,
    )

    assert len(results) == 1
    postprocessed = results[0].postprocessed
    assert postprocessed is not None
    assert postprocessed.accuracy == pytest.approx(1.0)
    assert postprocessed.recall_behavior == pytest.approx(1.0)
    assert postprocessed.confusion_matrix.shape == (2, 2)
    # raw metrics are untouched, so the two are directly comparable
    assert results[0].accuracy == pytest.approx(1.0)
    assert captured["members"] == [("v1.avi", 0)]
    assert captured["window_size"] == 5


def test_postprocessing_evaluation_skipped_without_enabled_stages(monkeypatch) -> None:
    """No enabled stages means the pipeline is a no-op, so skip the expensive pass."""

    def _fail(**_kwargs):
        raise AssertionError("postprocessing evaluation should not run")

    monkeypatch.setattr(cross_validation, "evaluate_group_with_postprocessing", _fail)
    status_messages: list[str] = []

    results = run_leave_one_group_out_cv(
        classifier=_BinaryCVClassifier(),
        project=_FakeProject(
            _FakeSettingsManager(
                [{"stage_name": "BoutStitchingStage", "enabled": False, "parameters": {}}]
            )
        ),
        features=_binary_features(),
        group_mapping={1: {"video": "v1.avi", "identity": 0, "members": [("v1.avi", 0)]}},
        behavior="Walk",
        k=1,
        status_callback=status_messages.append,
        evaluate_postprocessing=True,
    )

    assert results[0].postprocessed is None
    assert any("No postprocessing stages are enabled" in msg for msg in status_messages)


def test_postprocessing_evaluation_skipped_in_multiclass_mode(monkeypatch) -> None:
    """Postprocessing has no multi-class semantics, so the request is refused."""

    def _fail(**_kwargs):
        raise AssertionError("postprocessing evaluation should not run")

    monkeypatch.setattr(cross_validation, "evaluate_group_with_postprocessing", _fail)
    features = {
        "per_frame": pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}),
        "window": pd.DataFrame({"b": [5.0, 6.0, 7.0, 8.0]}),
        "groups": np.array([0, 0, 1, 1], dtype=np.int32),
        "labels_by_behavior": {
            "None": np.array([1, 0, 1, 0], dtype=np.int8),
            "Walk": np.array([0, 1, 0, 1], dtype=np.int8),
        },
    }
    status_messages: list[str] = []

    run_leave_one_group_out_cv(
        classifier=_MultiClassCVClassifier(),
        project=_FakeProject(_FakeSettingsManager(_STITCH_CONFIG)),
        features=features,
        group_mapping={1: {"video": "v1.avi", "identity": 0, "members": [("v1.avi", 0)]}},
        behavior="Walk",
        k=1,
        status_callback=status_messages.append,
        evaluate_postprocessing=True,
    )

    assert any("not supported in multi-class mode" in msg for msg in status_messages)


def test_postprocessing_evaluation_skipped_when_group_has_no_members(monkeypatch) -> None:
    """A group with no recorded members cannot be re-predicted, so it is skipped."""

    def _fail(**_kwargs):
        raise AssertionError("postprocessing evaluation should not run")

    monkeypatch.setattr(cross_validation, "evaluate_group_with_postprocessing", _fail)

    results = run_leave_one_group_out_cv(
        classifier=_BinaryCVClassifier(),
        project=_FakeProject(_FakeSettingsManager(_STITCH_CONFIG)),
        features=_binary_features(),
        group_mapping={1: {"video": "v1.avi", "identity": 0, "members": []}},
        behavior="Walk",
        k=1,
        evaluate_postprocessing=True,
    )

    assert results[0].postprocessed is None


def test_postprocessed_metrics_use_explicit_binary_labels() -> None:
    """A leftover -1 prediction must not shift which array element is which class.

    Interpolation can leave a frame with no prediction. Letting sklearn infer
    the label set would make -1 the first class and silently relabel the
    precision/recall entries.
    """
    evaluation = cross_validation.FoldPostprocessingEvaluation(
        truth=np.array([0, 0, 1, 1], dtype=np.int8),
        raw=np.array([0, 0, 1, 1], dtype=np.int8),
        postprocessed=np.array([0, 0, 1, -1], dtype=np.int8),
    )

    metrics = cross_validation._build_postprocessed_metrics(
        evaluation, fold_labels=evaluation.truth, fold_predictions=evaluation.raw
    )

    assert metrics.precision_not_behavior == pytest.approx(1.0)
    assert metrics.precision_behavior == pytest.approx(1.0)
    # the -1 frame counts as a miss for the behavior class, not as its own class
    assert metrics.recall_behavior == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.confusion_matrix.shape == (2, 2)
    # the -1 frame is excluded from the matrix (restricted to [0, 1]), so it
    # sums to fewer than the 4 scored frames - no_prediction_count says why
    assert metrics.confusion_matrix.sum() == 3
    assert metrics.no_prediction_count == 1


def test_postprocessed_metrics_no_prediction_count_is_zero_when_fully_predicted() -> None:
    """A fold where postprocessing never leaves a frame unpredicted reports zero."""
    evaluation = cross_validation.FoldPostprocessingEvaluation(
        truth=np.array([0, 0, 1, 1], dtype=np.int8),
        raw=np.array([0, 0, 1, 1], dtype=np.int8),
        postprocessed=np.array([0, 0, 1, 1], dtype=np.int8),
    )

    metrics = cross_validation._build_postprocessed_metrics(
        evaluation, fold_labels=evaluation.truth, fold_predictions=evaluation.raw
    )

    assert metrics.no_prediction_count == 0
    assert metrics.confusion_matrix.sum() == 4


def test_postprocessed_metrics_warn_when_raw_predictions_differ(caplog) -> None:
    """A raw-prediction disagreement between the two paths is surfaced, not hidden.

    The warning has to reach the saved report as well as the log: a GUI user
    never sees the log, and the report puts raw and postprocessed metrics side
    by side as though they were comparable.
    """
    evaluation = cross_validation.FoldPostprocessingEvaluation(
        truth=np.array([0, 1], dtype=np.int8),
        raw=np.array([1, 0], dtype=np.int8),
        postprocessed=np.array([0, 1], dtype=np.int8),
    )

    with caplog.at_level(logging.WARNING, logger="jabs.classifier.cross_validation"):
        metrics = cross_validation._build_postprocessed_metrics(
            evaluation,
            fold_labels=np.array([0, 1], dtype=np.int8),
            fold_predictions=np.array([0, 1], dtype=np.int8),
        )

    assert "does not reproduce" in caplog.text
    assert metrics.consistency_warning is not None
    assert "may not be comparable" in metrics.consistency_warning


def test_postprocessed_metrics_warn_when_predictions_differ_but_accuracy_matches() -> None:
    """Different prediction sequences with the same accuracy must still be flagged.

    With truth [0, 1], fold predictions [0, 0] and full-track predictions [1, 1]
    both score 0.5, so an accuracy comparison would pass them as consistent.
    """
    evaluation = cross_validation.FoldPostprocessingEvaluation(
        truth=np.array([0, 1], dtype=np.int8),
        raw=np.array([1, 1], dtype=np.int8),
        postprocessed=np.array([1, 1], dtype=np.int8),
    )

    metrics = cross_validation._build_postprocessed_metrics(
        evaluation,
        fold_labels=np.array([0, 1], dtype=np.int8),
        fold_predictions=np.array([0, 0], dtype=np.int8),
    )

    assert metrics.consistency_warning is not None


def test_postprocessed_metrics_warn_when_labels_differ() -> None:
    """A different ground truth is as disqualifying as different predictions."""
    evaluation = cross_validation.FoldPostprocessingEvaluation(
        truth=np.array([0, 1], dtype=np.int8),
        raw=np.array([0, 1], dtype=np.int8),
        postprocessed=np.array([0, 1], dtype=np.int8),
    )

    metrics = cross_validation._build_postprocessed_metrics(
        evaluation,
        fold_labels=np.array([1, 1], dtype=np.int8),
        fold_predictions=np.array([0, 1], dtype=np.int8),
    )

    assert metrics.consistency_warning is not None


def test_postprocessed_metrics_warn_when_frame_counts_differ() -> None:
    """Vectors of different lengths cannot be the same frames."""
    evaluation = cross_validation.FoldPostprocessingEvaluation(
        truth=np.array([0, 1, 1], dtype=np.int8),
        raw=np.array([0, 1, 1], dtype=np.int8),
        postprocessed=np.array([0, 1, 1], dtype=np.int8),
    )

    metrics = cross_validation._build_postprocessed_metrics(
        evaluation,
        fold_labels=np.array([0, 1], dtype=np.int8),
        fold_predictions=np.array([0, 1], dtype=np.int8),
    )

    assert metrics.consistency_warning is not None


def test_postprocessed_metrics_have_no_warning_when_the_passes_agree() -> None:
    """Labels and raw predictions that match frame for frame leave the metrics unannotated."""
    evaluation = cross_validation.FoldPostprocessingEvaluation(
        truth=np.array([0, 1], dtype=np.int8),
        raw=np.array([0, 1], dtype=np.int8),
        postprocessed=np.array([0, 1], dtype=np.int8),
    )

    metrics = cross_validation._build_postprocessed_metrics(
        evaluation, fold_labels=evaluation.truth, fold_predictions=evaluation.raw
    )

    assert metrics.consistency_warning is None
