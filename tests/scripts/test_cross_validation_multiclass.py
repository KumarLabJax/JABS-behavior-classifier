"""Tests for ``jabs-cli cross-validation`` on multi-class projects.

``run_cross_validation`` runs against a stubbed ``Project`` but a real
``MultiClassClassifier`` and the real cross-validation loop, on small synthetic
features, so the multi-class path is exercised end to end without a project on disk.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import jabs.scripts.cli.cross_validation as cv_module
from jabs.classifier import NO_VALID_SPLITS_WARNING, MultiClassClassifier
from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import (
    ClassifierMode,
    ClassifierType,
    CrossValidationGroupingStrategy,
    ProjectDistanceUnit,
)
from jabs.project.track_labels import TrackLabels
from jabs.scripts.cli.cross_validation import (
    _max_multiclass_splits,
    _multiclass_class_counts,
    _train_final_multiclass,
    run_cross_validation,
)

BEHAVIORS = ["Walk", "Run"]
ROWS_PER_CLASS_PER_GROUP = 30
SETTINGS = {"window_size": 5, "balance_labels": False, "symmetric_behavior": False}


def _make_features(n_groups: int, excluded_groups: set[int] | None = None) -> dict:
    """Build separable synthetic multi-class features with equal class counts per group.

    Args:
        n_groups: Number of cross-validation groups (videos).
        excluded_groups: Group ids to mark as excluded from training.

    Returns:
        A feature payload shaped like ``Project.get_multiclass_labeled_features``.
    """
    rng = np.random.default_rng(seed=0)
    classes = [MULTICLASS_NONE_BEHAVIOR, *BEHAVIORS]
    n_rows = n_groups * len(classes) * ROWS_PER_CLASS_PER_GROUP

    class_idx = np.tile(np.repeat(np.arange(len(classes)), ROWS_PER_CLASS_PER_GROUP), n_groups)
    groups = np.repeat(np.arange(n_groups), len(classes) * ROWS_PER_CLASS_PER_GROUP)

    signal = class_idx[:, None] + rng.normal(0, 0.1, size=(n_rows, 2))
    per_frame = pd.DataFrame(signal, columns=["pf_a", "pf_b"])
    window = pd.DataFrame(signal * 2, columns=["w_a", "w_b"])

    labels_by_behavior = {}
    for idx, name in enumerate(classes):
        arr = np.full(n_rows, TrackLabels.Label.NONE, dtype=np.int8)
        arr[class_idx == idx] = TrackLabels.Label.BEHAVIOR
        labels_by_behavior[name] = arr

    return {
        "per_frame": per_frame,
        "window": window,
        "labels_by_behavior": labels_by_behavior,
        "groups": groups,
        "excluded_groups": excluded_groups or set(),
    }


def _make_project(features: dict, behavior_names: list[str] | None = None) -> mock.MagicMock:
    """Build a Project-like mock for a multi-class project holding ``features``."""
    project = mock.MagicMock()
    names = BEHAVIORS if behavior_names is None else behavior_names
    project.settings_manager = SimpleNamespace(
        classifier_mode=ClassifierMode.MULTICLASS,
        behavior_names=names,
        cv_grouping_strategy=CrossValidationGroupingStrategy.VIDEO,
        cv_grouping_regex=None,
        is_video_excluded=lambda _video: False,
    )
    project.get_project_defaults.return_value = dict(SETTINGS)
    n_groups = len(np.unique(features["groups"]))
    group_mapping = {g: {"video": f"vid_{g}.avi", "identity": None} for g in range(n_groups)}
    project.get_multiclass_labeled_features.return_value = (features, group_mapping)
    project.counts.return_value = {
        "vid_0.avi": {0: {"unfragmented_bout_counts": (2, 0)}},
        "vid_1.avi": {0: {"unfragmented_bout_counts": (3, 0)}},
    }
    project.feature_manager.distance_unit = ProjectDistanceUnit.CM
    return project


@pytest.fixture
def patch_project(monkeypatch: pytest.MonkeyPatch):
    """Return a function that makes ``run_cross_validation`` open the given project."""

    def _patch(project: mock.MagicMock) -> None:
        project_cls = mock.MagicMock()
        project_cls.is_valid_project_directory.return_value = True
        project_cls.return_value = project
        monkeypatch.setattr(cv_module, "Project", project_cls)

    return _patch


def _run(tmp_path: Path, **kwargs) -> Path:
    """Run multi-class cross-validation into ``tmp_path`` and return the report path."""
    report = tmp_path / "report.md"
    params = {
        "project_dir": tmp_path,
        "behavior": None,
        "classifier_type": ClassifierType.RANDOM_FOREST,
        "grouping_strategy": None,
        "k": 0,
        "report_file": report,
    }
    params.update(kwargs)
    run_cross_validation(**params)
    return report


def test_multiclass_cross_validation_uses_multiclass_features(
    tmp_path: Path, patch_project
) -> None:
    """A multi-class project is cross-validated over all behaviors, not as a binary task."""
    project = _make_project(_make_features(n_groups=3))
    patch_project(project)

    report = _run(tmp_path)

    project.get_multiclass_labeled_features.assert_called_once()
    project.get_labeled_features.assert_not_called()
    text = report.read_text()
    assert "# Training Report: multiclass" in text
    # one iteration per group: every group is a valid test split (markdown escapes "_")
    assert "Mean F1 Score (Macro)" in text
    assert all(f"vid\\_{group}.avi" in text for group in range(3))
    for class_name in (MULTICLASS_NONE_BEHAVIOR, *BEHAVIORS):
        assert class_name in text


def test_multiclass_cross_validation_does_not_require_behavior(
    tmp_path: Path, patch_project, capsys: pytest.CaptureFixture[str]
) -> None:
    """Omitting --behavior is fine for multi-class projects and emits no warning."""
    patch_project(_make_project(_make_features(n_groups=3)))

    _run(tmp_path, behavior=None)

    assert "ignored" not in capsys.readouterr().out


def test_multiclass_cross_validation_warns_when_behavior_given(
    tmp_path: Path, patch_project, capsys: pytest.CaptureFixture[str]
) -> None:
    """A --behavior value has no meaning in multi-class mode, so the user is told it is ignored."""
    patch_project(_make_project(_make_features(n_groups=3)))

    _run(tmp_path, behavior="Walk")

    assert "--behavior is ignored" in capsys.readouterr().out


def test_multiclass_cross_validation_skips_postprocessing_with_warning(
    tmp_path: Path, patch_project, capsys: pytest.CaptureFixture[str]
) -> None:
    """Postprocessing evaluation is binary-only, so an explicit request is skipped and reported."""
    project = _make_project(_make_features(n_groups=3))
    patch_project(project)

    report = _run(tmp_path, evaluate_postprocessing=True)

    assert "postprocessing evaluation is not supported" in capsys.readouterr().out
    assert "ostprocess" not in report.read_text()


def test_multiclass_cross_validation_with_no_valid_splits_still_reports(
    tmp_path: Path, patch_project
) -> None:
    """With one group there is no held-out split, but the final model and report still happen."""
    patch_project(_make_project(_make_features(n_groups=1)))

    report = _run(tmp_path)

    assert NO_VALID_SPLITS_WARNING in report.read_text()


def test_multiclass_cross_validation_rejects_project_without_behaviors(
    tmp_path: Path, patch_project
) -> None:
    """A multi-class project with no behaviors has nothing to classify."""
    patch_project(_make_project(_make_features(n_groups=3), behavior_names=[]))

    with pytest.raises(ValueError, match="no behaviors defined"):
        _run(tmp_path)


def test_binary_cross_validation_requires_behavior(tmp_path: Path, patch_project) -> None:
    """--behavior became optional on the command, but binary projects still need it."""
    project = _make_project(_make_features(n_groups=3))
    project.settings_manager.classifier_mode = ClassifierMode.BINARY
    patch_project(project)

    with pytest.raises(ValueError, match="--behavior is required"):
        _run(tmp_path, behavior=None)


def test_max_multiclass_splits_counts_valid_groups() -> None:
    """Every group with enough of each class is a valid test split."""
    classifier = MultiClassClassifier(BEHAVIORS, classifier_type=ClassifierType.RANDOM_FOREST)

    assert _max_multiclass_splits(classifier, _make_features(n_groups=3)) == 3


def test_max_multiclass_splits_without_labels_is_zero() -> None:
    """No labeled frames means no valid splits, not an error."""
    classifier = MultiClassClassifier(BEHAVIORS, classifier_type=ClassifierType.RANDOM_FOREST)

    assert (
        _max_multiclass_splits(classifier, {"labels_by_behavior": {}, "groups": np.empty(0)}) == 0
    )


def test_train_final_multiclass_drops_excluded_groups() -> None:
    """The final model never trains on videos excluded from training."""
    features = _make_features(n_groups=3, excluded_groups={2})
    classifier = mock.MagicMock()
    classifier.combine_data.side_effect = lambda per_frame, window: pd.concat(
        [per_frame, window], axis=1
    )
    classifier.get_feature_importance.return_value = [("pf_a", 0.5)]

    top = _train_final_multiclass(classifier, features, SETTINGS)

    assert top == [("pf_a", 0.5)]
    payload = classifier.train.call_args.args[0]
    n_included = int(np.sum(features["groups"] != 2))
    assert len(payload["per_frame"]) == n_included
    assert all(len(arr) == n_included for arr in payload["labels_by_behavior"].values())
    assert payload["settings"] == SETTINGS
    classifier.set_dict_settings.assert_called_once_with(SETTINGS)


def test_multiclass_class_counts_cover_every_class() -> None:
    """Frame and bout counts are reported for the None class and every behavior."""
    classifier = MultiClassClassifier(BEHAVIORS, classifier_type=ClassifierType.RANDOM_FOREST)
    project = _make_project(_make_features(n_groups=3))

    frames, bouts = _multiclass_class_counts(project, classifier, _make_features(n_groups=3))

    expected = dict.fromkeys(classifier.get_class_names(), 3 * ROWS_PER_CLASS_PER_GROUP)
    assert frames == expected
    assert bouts == dict.fromkeys(classifier.get_class_names(), 5)
