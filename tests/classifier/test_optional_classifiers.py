"""Tests for the optional classifiers, CatBoost and LightGBM.

Both are installed with the ``extra-classifiers`` extra. Tests that train a model
skip without the package; the rest cover how JABS behaves when one is missing.
"""

import sys
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from jabs.classifier import base, classifier_utils, factories
from jabs.classifier.classifier import Classifier
from jabs.classifier.multi_class_classifier import MultiClassClassifier
from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierType
from jabs.project import TrackLabels

requires_lightgbm = pytest.mark.skipif(
    not factories.LIGHTGBM_AVAILABLE, reason="lightgbm not installed"
)

# Feature names with spaces, the way JABS names its features. LightGBM replaces the
# spaces with underscores, so these exercise the column matching at prediction time.
_FEATURE_NAMES = [
    "base_features angle NOSE-BASE_NECK-CENTER_SPINE",
    "base_features window 5 mean speed",
    "social_features closest distance",
]


@pytest.fixture
def features() -> pd.DataFrame:
    """200 frames of random features, with a few NaNs, under JABS-style names."""
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.standard_normal((200, len(_FEATURE_NAMES))), columns=_FEATURE_NAMES)
    data.iloc[5:10, 0] = np.nan
    return data


@pytest.fixture
def mock_project() -> MagicMock:
    """A project stand-in with the settings binary training reads."""
    project = MagicMock()
    settings = {"balance_labels": False, "symmetric_behavior": False}
    project.get_project_defaults.return_value = settings
    project.settings_manager.get_behavior.return_value = settings
    return project


def test_clean_features_keeps_nan_for_lightgbm() -> None:
    """LightGBM handles NaN natively, so only infinities are replaced."""
    features = pd.DataFrame({"a": [1.0, np.nan, np.inf, -np.inf]})

    cleaned = classifier_utils.clean_features(features, ClassifierType.LIGHTGBM)

    assert cleaned["a"].isna().tolist() == [False, True, True, True]


def _block_import(monkeypatch, package: str) -> None:
    """Make ``import <package>`` fail for the rest of the test."""
    monkeypatch.setitem(sys.modules, package, None)


@pytest.mark.parametrize(
    ("package", "factory"),
    [
        ("lightgbm", factories.make_lightgbm),
        ("catboost", factories.make_catboost),
        ("catboost", factories.make_catboost_multiclass),
    ],
    ids=["lightgbm", "catboost", "catboost-multiclass"],
)
def test_factory_raises_when_package_is_missing(monkeypatch, package, factory) -> None:
    """Asking for an optional classifier without its package gives a clear error."""
    _block_import(monkeypatch, package)

    with pytest.raises(RuntimeError, match="extra-classifiers"):
        factory(n_jobs=1, random_seed=0)


@pytest.mark.parametrize(
    ("classifier_type", "available"),
    [
        (ClassifierType.LIGHTGBM, factories.LIGHTGBM_AVAILABLE),
        (ClassifierType.CATBOOST, factories.CATBOOST_AVAILABLE),
    ],
    ids=["lightgbm", "catboost"],
)
def test_optional_classifier_is_offered_only_when_available(classifier_type, available) -> None:
    """An optional classifier is a choice in both modes exactly when it can be imported."""
    for multiclass in (False, True):
        supported = factories.supported_classifier_types(multiclass=multiclass)
        assert (classifier_type in supported) == available


def test_loading_a_classifier_whose_package_is_missing_explains_the_fix(
    monkeypatch, tmp_path
) -> None:
    """Unpickling a classifier that needs a missing optional package names the extra."""
    missing = ModuleNotFoundError("No module named 'catboost'", name="catboost")
    monkeypatch.setattr(joblib, "load", MagicMock(side_effect=missing))

    with pytest.raises(ValueError, match="'catboost'.*extra-classifiers"):
        Classifier.from_pickle(tmp_path / "classifier.pickle")


def test_loading_a_classifier_missing_another_module_is_not_masked(monkeypatch, tmp_path) -> None:
    """A missing module that is not an optional classifier is reported as it is."""
    missing = ModuleNotFoundError("No module named 'nonexistent'", name="nonexistent")
    monkeypatch.setattr(joblib, "load", MagicMock(side_effect=missing))

    with pytest.raises(ModuleNotFoundError, match="nonexistent"):
        base.unpickle_classifier(tmp_path / "classifier.pickle")


def test_loading_a_classifier_whose_package_cannot_load_explains_the_fix(
    monkeypatch, tmp_path
) -> None:
    """A native-library failure while unpickling names the package that could not load."""
    broken = OSError("dlopen(.../lightgbm/lib/lib_lightgbm.dylib): Library not loaded")
    monkeypatch.setattr(joblib, "load", MagicMock(side_effect=broken))

    with pytest.raises(ValueError, match="'lightgbm'.*could not be loaded"):
        Classifier.from_pickle(tmp_path / "classifier.pickle")


def test_loading_a_missing_classifier_file_is_not_masked(tmp_path) -> None:
    """A file error is reported as it is, even if the path names an optional package."""
    with pytest.raises(FileNotFoundError):
        base.unpickle_classifier(tmp_path / "lightgbm" / "classifier.pickle")


@pytest.mark.parametrize(
    "classifier_type",
    [ClassifierType.LIGHTGBM, ClassifierType.CATBOOST],
    ids=["lightgbm", "catboost"],
)
def test_creating_a_classifier_whose_package_is_missing_explains_the_fix(
    monkeypatch, classifier_type
) -> None:
    """A training file naming an unavailable optional classifier names the extra."""
    monkeypatch.setattr(
        Classifier, "_supported_classifier_choices", classmethod(lambda cls: set())
    )

    with pytest.raises(ValueError, match="extra-classifiers"):
        Classifier(classifier=classifier_type)


@requires_lightgbm
def test_binary_train_predict_and_reload(features, mock_project, tmp_path) -> None:
    """A binary LightGBM classifier trains, predicts, and survives a save and load."""
    labels = np.array([TrackLabels.Label.BEHAVIOR, TrackLabels.Label.NOT_BEHAVIOR] * 100)
    clf = Classifier(classifier=ClassifierType.LIGHTGBM)
    clf.behavior_name = "Test"
    clf.set_project_settings(mock_project)
    clf.train({"training_data": features, "training_labels": labels}, random_seed=42)

    # predicting with the original, space-separated names must still find the columns
    probabilities = clf.predict_proba(features)
    predictions = clf.predict(features)

    assert probabilities.shape == (len(features), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)
    assert set(np.unique(predictions)) <= {0, 1}
    # importances are reported under the original feature names, as fractions of the total
    importances = clf.get_feature_importance()
    assert {name for name, _ in importances} <= set(_FEATURE_NAMES)
    assert sum(value for _, value in importances) == pytest.approx(1.0, abs=0.02)

    path = tmp_path / "lightgbm.pickle"
    clf.save(path)
    reloaded = Classifier.from_pickle(path)
    np.testing.assert_allclose(reloaded.predict_proba(features), probabilities)


@requires_lightgbm
def test_multiclass_train_and_predict(features) -> None:
    """A multi-class LightGBM classifier predicts one probability per class."""
    behaviors = ["running", "grooming"]
    n = len(features)
    unlabeled = TrackLabels.Label.NONE
    labels = {
        name: np.full(n, unlabeled, dtype=np.int8)
        for name in [*behaviors, MULTICLASS_NONE_BEHAVIOR]
    }
    # every frame gets exactly one of the three classes, in rotation
    for offset, name in enumerate([MULTICLASS_NONE_BEHAVIOR, *behaviors]):
        labels[name][offset::3] = TrackLabels.Label.BEHAVIOR

    clf = MultiClassClassifier(behaviors, classifier_type=ClassifierType.LIGHTGBM)
    clf.train(
        {
            "per_frame": features.iloc[:, :2],
            "window": features.iloc[:, 2:],
            "labels_by_behavior": labels,
        },
        random_seed=42,
    )

    probabilities = clf.predict_proba(features)
    predictions = clf.predict(features)

    assert probabilities.shape == (n, len(behaviors) + 1)
    assert set(np.unique(predictions)) <= {0, 1, 2}
