"""Tests for cross-validation helpers."""

import numpy as np
import pandas as pd

from jabs.classifier.cross_validation import run_leave_one_group_out_cv


class _NoSplitClassifier:
    """Classifier test double reporting no valid LOGO splits."""

    @staticmethod
    def get_leave_one_group_out_max(_labels, _groups) -> int:
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
    def get_leave_one_group_out_max(_labels, _groups) -> int:
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
    def get_leave_one_group_out_max(_labels, _groups) -> int:
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
