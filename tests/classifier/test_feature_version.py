"""Classifiers record the feature version they were trained on and refuse any other."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest

from jabs.classifier import (
    Classifier,
    ClassifierFeatureVersionError,
    MultiClassClassifier,
)
from jabs.core.constants import CLASSIFIER_MODE_KEY, MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierMode, ClassifierType
from jabs.feature_extraction import FEATURE_VERSION
from jabs.project import Project, TrackLabels
from jabs.project.export_training import export_training_data
from jabs.project.prediction_manager import MULTICLASS_PREDICTION_KEY
from jabs.project.project_utils import to_safe_name

_SETTINGS = {"balance_labels": False, "symmetric_behavior": False}
_ROWS = 40


def _binary_data() -> dict:
    """Return a small binary training data dict, the shape project training produces."""
    rng = np.random.default_rng(0)
    labels = np.array([TrackLabels.Label.BEHAVIOR, TrackLabels.Label.NOT_BEHAVIOR] * (_ROWS // 2))
    return {
        "training_data": pd.DataFrame({"a": rng.standard_normal(_ROWS), "b": rng.random(_ROWS)}),
        "training_labels": labels,
    }


def _trained_binary() -> Classifier:
    """Return a binary classifier trained on project-style data."""
    classifier = Classifier(ClassifierType.RANDOM_FOREST)
    classifier.behavior_name = "Walk"
    classifier.set_dict_settings(dict(_SETTINGS))
    classifier.train(_binary_data(), random_seed=0)
    return classifier


def _multiclass_data() -> dict:
    """Return a small multi-class training data dict."""
    rng = np.random.default_rng(0)
    behavior = TrackLabels.Label.BEHAVIOR
    not_behavior = TrackLabels.Label.NOT_BEHAVIOR
    walk = np.array([behavior, not_behavior] * (_ROWS // 2))
    run = np.array([not_behavior, behavior] * (_ROWS // 2))
    return {
        "per_frame": pd.DataFrame({"a": rng.standard_normal(_ROWS)}),
        "window": pd.DataFrame({"b": rng.random(_ROWS)}),
        "labels_by_behavior": {"Walk": walk, "Run": run},
        "settings": dict(_SETTINGS),
    }


def _trained_multiclass() -> MultiClassClassifier:
    """Return a multi-class classifier trained on project-style data."""
    classifier = MultiClassClassifier(["Walk", "Run"])
    classifier.train(_multiclass_data(), random_seed=0)
    return classifier


def test_training_on_project_features_records_the_feature_version() -> None:
    """Project training data carries no version; the current FEATURE_VERSION is recorded."""
    assert _trained_binary()._feature_version == FEATURE_VERSION
    assert _trained_multiclass()._feature_version == FEATURE_VERSION


@pytest.mark.parametrize("make", [_trained_binary, _trained_multiclass], ids=["binary", "multi"])
def test_round_trip_keeps_the_feature_version(tmp_path: Path, make) -> None:
    """Saving and loading (both load paths) keeps the recorded feature version."""
    path = tmp_path / "classifier.pickle"
    make().save(path)
    cls = type(make())

    assert cls.from_pickle(path)._feature_version == FEATURE_VERSION
    target = Classifier() if cls is Classifier else MultiClassClassifier(["x"])
    target.load(path)
    assert target._feature_version == FEATURE_VERSION


def _new_instance(cls: type) -> Classifier | MultiClassClassifier:
    """Return an untrained instance of ``cls`` to load into."""
    return Classifier() if cls is Classifier else MultiClassClassifier(["x"])


@pytest.mark.parametrize("make", [_trained_binary, _trained_multiclass], ids=["binary", "multi"])
def test_classifier_without_a_feature_version_is_refused(tmp_path: Path, make) -> None:
    """A classifier saved before feature versions were recorded must be retrained."""
    classifier = make()
    del classifier._feature_version
    path = tmp_path / "old.pickle"
    classifier.save(path)

    with pytest.raises(ClassifierFeatureVersionError, match="retrain"):
        type(classifier).from_pickle(path)
    target = _new_instance(type(classifier))
    with pytest.raises(ClassifierFeatureVersionError, match="retrain"):
        target.load(path)
    # a refused load leaves the target untouched
    assert target._classifier is None


@pytest.mark.parametrize("make", [_trained_binary, _trained_multiclass], ids=["binary", "multi"])
def test_classifier_with_another_feature_version_is_refused(tmp_path: Path, make) -> None:
    """A classifier trained on other features must be retrained."""
    classifier = make()
    classifier._feature_version = FEATURE_VERSION - 1
    path = tmp_path / "stale.pickle"
    classifier.save(path)

    with pytest.raises(ClassifierFeatureVersionError, match="retrain") as raised:
        type(classifier).from_pickle(path)
    assert str(FEATURE_VERSION - 1) in str(raised.value)
    with pytest.raises(ClassifierFeatureVersionError, match="retrain"):
        _new_instance(type(classifier)).load(path)


def test_feature_version_error_is_a_value_error() -> None:
    """Callers catching ValueError keep catching the new error."""
    assert issubclass(ClassifierFeatureVersionError, ValueError)


@pytest.mark.parametrize("version", [FEATURE_VERSION - 1, None], ids=["v19", "missing"])
def test_training_refuses_data_exported_with_another_feature_version(version) -> None:
    """Exported data from another feature version is refused before fitting."""
    data = {**_binary_data(), "file_version": version}
    classifier = Classifier()
    classifier.set_dict_settings(dict(_SETTINGS))
    with pytest.raises(ClassifierFeatureVersionError, match="re-export the training data"):
        classifier.train(data)
    assert classifier._classifier is None

    multi = MultiClassClassifier(["Walk", "Run"])
    with pytest.raises(ClassifierFeatureVersionError, match="retrain"):
        multi.train({**_multiclass_data(), "file_version": version})


def test_training_on_current_exported_data_records_its_version() -> None:
    """Exported data from this feature version trains and is recorded."""
    classifier = Classifier()
    classifier.set_dict_settings(dict(_SETTINGS))
    classifier.train({**_binary_data(), "file_version": FEATURE_VERSION})
    assert classifier._feature_version == FEATURE_VERSION


def _export(tmp_path: Path, file_version: int) -> Path:
    """Export binary training data and stamp ``file_version`` on it."""
    rows = 2 * Classifier.LABEL_THRESHOLD
    features = {
        "per_frame": pd.DataFrame({"feat_a": np.arange(rows, dtype=np.float32)}),
        "window": pd.DataFrame({"feat_b": np.arange(rows, dtype=np.float32) * 0.1}),
        "labels": np.tile(np.array([1, 0], dtype=np.int8), rows // 2),
        "groups": np.zeros(rows, dtype=np.int32),
    }
    project = MagicMock()
    project.dir = tmp_path
    project.get_labeled_features.return_value = (
        features,
        {0: {"video": "vid_a.mp4", "identity": 0}},
    )
    settings = {"window_size": 5, **_SETTINGS}
    project.settings_manager = SimpleNamespace(get_behavior=lambda _: settings)
    out = export_training_data(project, "Walk", frozenset(), ClassifierType.RANDOM_FOREST)
    with h5py.File(out, "r+") as h5:
        h5.attrs["file_version"] = file_version
    return out


def test_from_training_file_refuses_an_old_export(tmp_path: Path) -> None:
    """A v19 export is refused; the message says to re-export."""
    with pytest.raises(ClassifierFeatureVersionError, match="re-export the training data"):
        Classifier.from_training_file(_export(tmp_path, FEATURE_VERSION - 1))


def test_from_training_file_records_the_export_version(tmp_path: Path) -> None:
    """A current export trains, and the classifier records its version."""
    classifier = Classifier.from_training_file(_export(tmp_path, FEATURE_VERSION))
    assert classifier._feature_version == FEATURE_VERSION


def _multiclass_project(tmp_path: Path) -> Project:
    """Return a bare multi-class project with behaviors Walk and Run."""
    project = Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )
    project.settings_manager.save_project_file(
        {"settings": {CLASSIFIER_MODE_KEY: ClassifierMode.MULTICLASS.value}}
    )
    project.settings_manager.save_behavior("Walk", {})
    project.settings_manager.save_behavior("Run", {})
    return project


def test_rename_with_a_stale_classifier_succeeds_and_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A stale multi-class classifier is skipped, logged, and left alone by a rename."""
    project = _multiclass_project(tmp_path)
    stale = _trained_multiclass()
    stale._feature_version = FEATURE_VERSION - 1
    classifier_path = project.classifier_dir / "_multiclass.pickle"
    stale.save(classifier_path)
    before = classifier_path.read_bytes()

    safe_multiclass = to_safe_name(MULTICLASS_PREDICTION_KEY)
    pred_file = project.project_paths.prediction_dir / "video1.h5"
    project._video_manager._videos = ["video1.avi"]
    with h5py.File(pred_file, "w") as hf:
        group = hf.create_group(f"predictions/{safe_multiclass}")
        group.create_dataset(
            "class_names",
            data=np.array([MULTICLASS_NONE_BEHAVIOR, "Walk", "Run"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    with caplog.at_level(logging.WARNING, logger="jabs.project.project"):
        project.rename_behavior("Walk", "Standing")

    assert "retrain" in caplog.text
    assert classifier_path.read_bytes() == before
    assert "Standing" in project.settings_manager.behavior_names
    with h5py.File(pred_file, "r") as hf:
        names = [v.decode() for v in hf[f"predictions/{safe_multiclass}/class_names"][()]]
    assert names == [MULTICLASS_NONE_BEHAVIOR, "Standing", "Run"]


def test_project_load_classifier_raises_for_a_stale_classifier(tmp_path: Path) -> None:
    """The GUI needs the error to tell the user; load_classifier does not swallow it."""
    project = _multiclass_project(tmp_path)
    stale = _trained_multiclass()
    stale._feature_version = FEATURE_VERSION - 1
    stale.save(project.classifier_dir / "_multiclass.pickle")

    with pytest.raises(ClassifierFeatureVersionError, match="retrain"):
        project.load_classifier(MultiClassClassifier(["Walk", "Run"]))


def test_feature_version_is_20() -> None:
    """Features follow the unified pose file's semantics from version 20 on (spec section 2)."""
    assert FEATURE_VERSION == 20
