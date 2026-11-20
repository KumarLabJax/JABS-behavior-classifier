"""Tests for loading saved classifiers whose algorithm is unavailable."""

import logging
import warnings
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from jabs.classifier import Classifier, MultiClassClassifier, UnsupportedClassifierError
from jabs.classifier import base as base_module
from jabs.classifier.base import unpickle_classifier
from jabs.core.enums import ClassifierType
from jabs.scripts.classify import load_classifier_from_pickle


@pytest.fixture
def unknown_type_pickle(tmp_path: Path) -> Path:
    """A saved classifier whose ClassifierType value doesn't exist in this JABS version.

    Mimics a classifier trained with a JABS install that defines an extra
    classifier type (e.g. LightGBM). The replacement value has the same length
    so the pickle framing stays valid.
    """
    path = tmp_path / "classifier.pickle"
    Classifier(ClassifierType.RANDOM_FOREST).save(path)
    data = path.read_bytes()
    assert data.count(b"Random Forest") == 1
    path.write_bytes(data.replace(b"Random Forest", b"Unknown Algo1"))
    return path


def test_load_unknown_classifier_type_raises_unsupported(unknown_type_pickle: Path) -> None:
    """An unknown enum value in the pickle raises UnsupportedClassifierError."""
    with pytest.raises(UnsupportedClassifierError) as exc_info:
        Classifier().load(unknown_type_pickle)
    assert exc_info.value.classifier_name == "Unknown Algo1"


def test_from_pickle_unknown_classifier_type_raises_unsupported(
    unknown_type_pickle: Path,
) -> None:
    """from_pickle also translates an unknown enum value."""
    with pytest.raises(UnsupportedClassifierError):
        Classifier.from_pickle(unknown_type_pickle)


def test_load_missing_package_raises_unsupported(tmp_path: Path) -> None:
    """A pickle that needs an uninstalled package raises UnsupportedClassifierError."""
    path = tmp_path / "classifier.pickle"
    Classifier().save(path)
    missing = ModuleNotFoundError("No module named 'lightgbm'", name="lightgbm")
    with (
        patch("jabs.classifier.base.joblib.load", side_effect=missing),
        pytest.raises(UnsupportedClassifierError) as exc_info,
    ):
        Classifier().load(path)
    assert exc_info.value.classifier_name == "lightgbm"


@pytest.mark.parametrize("cls", [Classifier, MultiClassClassifier], ids=["binary", "multi"])
def test_load_unavailable_known_type_raises_unsupported(
    cls: type[Classifier] | type[MultiClassClassifier], tmp_path: Path
) -> None:
    """A known type that's unavailable in this environment raises UnsupportedClassifierError."""
    path = tmp_path / "classifier.pickle"
    clf = Classifier() if cls is Classifier else MultiClassClassifier(["a", "b"])
    clf.save(path)
    with (
        patch.object(cls, "_supported_classifier_choices", return_value=set()),
        pytest.raises(UnsupportedClassifierError) as exc_info,
    ):
        cls.from_pickle(path)
    assert exc_info.value.classifier_name == ClassifierType.RANDOM_FOREST.value


def test_unsupported_classifier_error_is_value_error() -> None:
    """Existing callers that catch ValueError keep working."""
    assert issubclass(UnsupportedClassifierError, ValueError)


def test_load_unrelated_value_error_propagates(tmp_path: Path) -> None:
    """A ValueError unrelated to the classifier type is not reclassified."""
    path = tmp_path / "classifier.pickle"
    Classifier().save(path)
    with (
        patch("jabs.classifier.base.joblib.load", side_effect=ValueError("bad data")),
        pytest.raises(ValueError, match="bad data") as exc_info,
    ):
        Classifier().load(path)
    assert not isinstance(exc_info.value, UnsupportedClassifierError)


def test_load_missing_unrelated_module_propagates(tmp_path: Path) -> None:
    """A missing module that isn't an optional classifier package is not reclassified."""
    path = tmp_path / "classifier.pickle"
    Classifier().save(path)
    missing = ModuleNotFoundError(
        "No module named 'sklearn.ensemble._gb_losses'", name="sklearn.ensemble._gb_losses"
    )
    with (
        patch("jabs.classifier.base.joblib.load", side_effect=missing),
        pytest.raises(ModuleNotFoundError) as exc_info,
    ):
        Classifier().load(path)
    assert not isinstance(exc_info.value, UnsupportedClassifierError)


def test_cli_load_unknown_classifier_type_raises_unsupported(unknown_type_pickle: Path) -> None:
    """The jabs-classify loader also raises UnsupportedClassifierError."""
    with pytest.raises(UnsupportedClassifierError):
        load_classifier_from_pickle(unknown_type_pickle)


def test_unpickle_warnings_reach_the_caller(tmp_path: Path) -> None:
    """Warnings raised while unpickling are passed on to the caller."""

    def load_with_warning(_path: Path) -> object:
        warnings.warn("from unpickling", UserWarning, stacklevel=1)
        return object()

    with (
        patch("jabs.classifier.base.joblib.load", load_with_warning),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("default")
        unpickle_classifier(tmp_path / "classifier.pickle")

    assert [str(w.message) for w in caught] == ["from unpickling"]


class XGBoostError(ValueError):
    """Stand-in for xgboost.core.XGBoostError, which subclasses ValueError."""

    __module__ = "xgboost.core"


def test_unloadable_xgboost_library_raises_unsupported(tmp_path: Path) -> None:
    """xgboost failing to load its native library is reported as unsupported.

    On macOS without libomp, importing xgboost while unpickling raises XGBoostError.
    """
    error = XGBoostError("XGBoost Library (libxgboost.dylib) could not be loaded.")
    with (
        patch("jabs.classifier.base.joblib.load", side_effect=error),
        pytest.raises(UnsupportedClassifierError) as exc_info,
    ):
        unpickle_classifier(tmp_path / "classifier.pickle")
    assert exc_info.value.classifier_name == "xgboost"
    assert "installed but could not be loaded" in str(exc_info.value)


@pytest.fixture
def two_type_classifier() -> Iterator[Classifier]:
    """A classifier whose environment supports exactly Random Forest and XGBoost."""
    supported = {ClassifierType.RANDOM_FOREST, ClassifierType.XGBOOST}
    with patch.object(Classifier, "_supported_classifier_choices", return_value=supported):
        yield Classifier(ClassifierType.RANDOM_FOREST)


def test_training_file_type_used_when_no_override(two_type_classifier: Classifier) -> None:
    """With no override, the type recorded in the training file is applied."""
    two_type_classifier._apply_training_file_classifier_type(ClassifierType.XGBOOST, None)
    assert two_type_classifier.classifier_type is ClassifierType.XGBOOST


def test_training_file_type_override_wins(two_type_classifier: Classifier) -> None:
    """An explicit override takes precedence over the recorded type."""
    two_type_classifier._apply_training_file_classifier_type(
        ClassifierType.XGBOOST, ClassifierType.RANDOM_FOREST
    )
    assert two_type_classifier.classifier_type is ClassifierType.RANDOM_FOREST


@pytest.mark.parametrize(
    ("recorded", "override"),
    [
        (ClassifierType.LIGHTGBM, None),
        (ClassifierType.RANDOM_FOREST, ClassifierType.LIGHTGBM),
    ],
    ids=["recorded", "override"],
)
def test_unavailable_training_file_type_keeps_default(
    two_type_classifier: Classifier,
    recorded: ClassifierType,
    override: ClassifierType | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unavailable request is warned about and leaves the classifier unchanged."""
    with caplog.at_level(logging.WARNING, logger=base_module.__name__):
        two_type_classifier._apply_training_file_classifier_type(recorded, override)

    assert two_type_classifier.classifier_type is ClassifierType.RANDOM_FOREST
    assert "LIGHTGBM is unavailable" in caplog.text
