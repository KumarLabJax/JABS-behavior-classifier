"""Shared infrastructure for behavior classifiers.

``BaseClassifier`` consolidates persistence, identity properties, factory
dispatch, feature cleaning, and feature-importance reporting that are common
to both binary and multi-class classifiers. Subclasses provide the train and
predict implementations that determine the actual learning behavior.

The class is concrete (not abstract): subclasses are not required to
override anything to instantiate. Public surface for classifier *consumers*
is governed by :class:`jabs.classifier.ClassifierProtocol`, which both
subclasses satisfy structurally.
"""

from __future__ import annotations

import logging
import typing
import warnings
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from jabs.core.enums import ClassifierType
from jabs.core.utils import hash_file

from . import classifier_utils, factories

logger = logging.getLogger(__name__)


class UnsupportedClassifierError(ValueError):
    """Raised when a saved classifier uses an algorithm unavailable in this environment.

    This happens when a project is opened with a JABS install that doesn't
    support the algorithm the classifier was trained with: the classifier type
    isn't defined in this JABS version, or the package providing it isn't
    installed or importable.

    Attributes:
        classifier_name: Name of the unsupported algorithm, if it could be
            determined from the saved file.
    """

    def __init__(self, classifier_name: str | None = None, detail: str | None = None) -> None:
        self.classifier_name = classifier_name
        name = classifier_name if classifier_name else "unknown"
        message = f"Classifier type '{name}' is not supported by this JABS installation"
        super().__init__(f"{message}: {detail}" if detail else message)


class ClassifierFeatureVersionError(ValueError):
    """Raised when a classifier or its training data belongs to another feature version.

    Classifiers record the ``FEATURE_VERSION`` of the features they were trained
    on. Predicting with features computed by a different version is meaningless,
    so a classifier with another (or no recorded) version must be retrained, and
    exported training data from another version must be re-exported.

    Attributes:
        found: The feature version recorded, or None when none was recorded.
        expected: The feature version of this JABS installation.
    """

    def __init__(self, message: str, found: int | None, expected: int) -> None:
        self.found = found
        self.expected = expected
        super().__init__(message)


def _current_feature_version() -> int:
    """Return this installation's ``FEATURE_VERSION``.

    Imported lazily: ``jabs.feature_extraction`` is heavy and pulls in modules
    that import ``jabs.classifier``.
    """
    from jabs.feature_extraction import FEATURE_VERSION

    return FEATURE_VERSION


def _describe_version(version: int | None) -> str:
    """Return a phrase naming a recorded feature version for error messages."""
    return "no recorded feature version" if version is None else f"feature version {version}"


def check_classifier_feature_version(classifier: typing.Any, path: Path) -> None:
    """Refuse a loaded classifier trained on features from another feature version.

    Args:
        classifier: The unpickled classifier.
        path: The file it was loaded from, named in the error.

    Raises:
        ClassifierFeatureVersionError: If the classifier records no feature
            version or one other than this installation's ``FEATURE_VERSION``.
    """
    expected = _current_feature_version()
    found = getattr(classifier, "_feature_version", None)
    if found != expected:
        raise ClassifierFeatureVersionError(
            f"The classifier in {Path(path).name} was trained with {_describe_version(found)}, "
            f"but this JABS computes feature version {expected}. "
            f"Its features no longer match; retrain the classifier.",
            found,
            expected,
        )


def check_training_data_feature_version(file_version: int | None) -> int:
    """Refuse exported training data written with another feature version.

    Args:
        file_version: The export's ``file_version`` attribute (None when absent).

    Returns:
        The data's feature version, which equals this installation's.

    Raises:
        ClassifierFeatureVersionError: If ``file_version`` differs from
            ``FEATURE_VERSION`` or is missing.
    """
    expected = _current_feature_version()
    if file_version != expected:
        raise ClassifierFeatureVersionError(
            f"The training data was exported with {_describe_version(file_version)}, but this "
            f"JABS computes feature version {expected}. Open the project with this JABS, "
            f"re-export the training data, and retrain the classifier from the new export.",
            file_version,
            expected,
        )
    return file_version


# packages providing classifier algorithms that may be missing from an environment: the
# extra-classifiers packages, plus xgboost, which can be unimportable on some platforms
_OPTIONAL_CLASSIFIER_PACKAGES = frozenset(
    {"xgboost", *factories.OPTIONAL_CLASSIFIER_PACKAGES.values()}
)


def unpickle_classifier(path: Path) -> typing.Any:
    """Load a pickled classifier, translating unsupported-algorithm failures.

    Args:
        path: Path to the saved classifier pickle file.

    Returns:
        The unpickled object.

    Raises:
        UnsupportedClassifierError: If the pickle references a classifier type
            or algorithm package that is not available in this environment.
        ValueError: If the classifier was trained with a different sklearn version.
    """
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", InconsistentVersionWarning)
        try:
            c = joblib.load(path)
        except ModuleNotFoundError as e:
            # only a missing optional classifier package means an unsupported algorithm;
            # other missing modules (e.g. sklearn/numpy internals) are version mismatches
            package = (e.name or "").partition(".")[0]
            if package not in _OPTIONAL_CLASSIFIER_PACKAGES:
                raise
            detail = f"the '{package}' package is not installed"
            if package in factories.OPTIONAL_CLASSIFIER_PACKAGES.values():
                detail += f". {factories.EXTRA_CLASSIFIERS_ADVICE}"
            raise UnsupportedClassifierError(package, detail) from e
        except OSError as e:
            # an installed package whose native library cannot load (e.g. macOS without
            # libomp) raises OSError while importing it; any other OSError, such as a
            # failure reading the file, is reported as it is
            package = _optional_package_raising(e)
            if package is None or e.filename is not None:
                raise
            raise _unloadable_package_error(package, e) from e
        except ValueError as e:
            # an optional classifier package that is installed but can't be imported,
            # e.g. xgboost raises XGBoostError when its native library fails to load
            package = type(e).__module__.partition(".")[0]
            if package in _OPTIONAL_CLASSIFIER_PACKAGES:
                raise _unloadable_package_error(package, e) from e
            # the pickled ClassifierType value is not defined in this JABS version
            if ClassifierType.__name__ not in str(e):
                raise
            raise UnsupportedClassifierError(_unknown_enum_value(e)) from e
    # re-emit outside the recording block, otherwise each re-emitted warning is
    # recorded into the list being iterated instead of reaching the caller
    for warning in caught_warnings:
        if issubclass(warning.category, InconsistentVersionWarning):
            raise ValueError("Classifier trained with different version of sklearn.")
        warnings.warn(warning.message, warning.category, stacklevel=3)
    return c


def _optional_package_raising(error: BaseException) -> str | None:
    """Return the optional classifier package whose code raised ``error``, if any."""
    package = None
    tb = error.__traceback__
    while tb is not None:
        top = tb.tb_frame.f_globals.get("__name__", "").partition(".")[0]
        if top in _OPTIONAL_CLASSIFIER_PACKAGES:
            package = top
        tb = tb.tb_next
    return package


def _unloadable_package_error(package: str, error: Exception) -> UnsupportedClassifierError:
    """Build the error for an optional package that is installed but fails to import."""
    return UnsupportedClassifierError(
        package, f"the '{package}' package is installed but could not be loaded: {error}"
    )


def _unknown_enum_value(error: ValueError) -> str | None:
    """Extract the rejected value from an enum ``'X' is not a valid Enum`` error."""
    message = str(error)
    if message.startswith("'") and "' is not a valid" in message:
        return message[1 : message.index("' is not a valid")]
    return None


class BaseClassifier:
    """Shared persistence and identity machinery for JABS classifiers.

    Class attributes that subclasses must set:
        ``_VERSION``: pickled-format version integer for this subclass.
        ``_MULTICLASS``: True if this subclass operates in multi-class mode.
        ``_PERSISTED_REQUIRED``: tuple of instance attribute names that ``load``
            must restore from the pickled instance.
        ``_PERSISTED_OPTIONAL``: tuple of instance attribute names that
            ``load`` should restore if present on the pickled instance (default
            to ``None`` otherwise). Used to support older pickles that may not
            have all attributes the live class now declares.
    """

    _VERSION: ClassVar[int] = 0
    _MULTICLASS: ClassVar[bool] = False
    _PERSISTED_REQUIRED: ClassVar[tuple[str, ...]] = ()
    _PERSISTED_OPTIONAL: ClassVar[tuple[str, ...]] = ()

    def __init__(self, classifier_type: ClassifierType, n_jobs: int = 1) -> None:
        self._classifier_type = classifier_type
        self._classifier: typing.Any = None
        self._project_settings: dict | None = None
        self._feature_names: list[str] | None = None
        # FEATURE_VERSION of the features this classifier was trained on; set by train()
        self._feature_version: int | None = None
        self._n_jobs = n_jobs
        self._version = self._VERSION

        self._classifier_file: str | None = None
        self._classifier_hash: str | None = None
        self._classifier_source: str | None = None

        self._supported_classifiers = self._supported_classifier_choices()
        if classifier_type not in self._supported_classifiers:
            if classifier_type in factories.OPTIONAL_CLASSIFIER_PACKAGES:
                raise ValueError(
                    f"{classifier_type.value} classifier requires the "
                    f"'{factories.OPTIONAL_CLASSIFIER_PACKAGES[classifier_type]}' package, "
                    f"which is not available. {factories.EXTRA_CLASSIFIERS_ADVICE}"
                )
            raise ValueError("Invalid classifier type")

    @property
    def classifier_name(self) -> str:
        """Return the name of the underlying algorithm."""
        return self._classifier_type.value

    @property
    def classifier_type(self) -> ClassifierType:
        """Return the underlying classifier algorithm enum value."""
        return self._classifier_type

    @property
    def classifier_file(self) -> str | None:
        """Return the filename of the saved classifier, if any."""
        return self._classifier_file

    @property
    def classifier_hash(self) -> str | None:
        """Return the content hash of the saved classifier, if any."""
        return self._classifier_hash

    @property
    def project_settings(self) -> dict:
        """Return a copy of the classifier's training settings."""
        if self._project_settings is not None:
            return dict(self._project_settings)
        return {}

    @property
    def version(self) -> int:
        """Return the serialized classifier format version."""
        return self._version

    @property
    def feature_names(self) -> list[str] | None:
        """Return the list of feature names used to train this classifier."""
        return self._feature_names

    @classmethod
    def _supported_classifier_choices(cls) -> set[ClassifierType]:
        """Return classifier types available in the current environment.

        Resolved per-call so that test code can patch
        :func:`jabs.classifier.factories.supported_classifier_types` or this
        method on the subclass without freezing state at import time.
        """
        return factories.supported_classifier_types(multiclass=cls._MULTICLASS)

    def set_classifier(self, classifier: ClassifierType) -> None:
        """Switch the underlying classifier algorithm.

        Args:
            classifier: The classifier type to switch to.

        Raises:
            ValueError: If the classifier type is not supported.
        """
        if classifier not in self._supported_classifier_choices():
            raise ValueError("Invalid Classifier Type")
        self._classifier_type = classifier

    def _apply_training_file_classifier_type(
        self, recorded_type: ClassifierType, override: ClassifierType | None
    ) -> None:
        """Switch to the algorithm a training file asks for, if this environment has it.

        A training file records the algorithm it was exported with, and the caller
        may override that. Either one can name an algorithm whose optional package
        is not installed here, which is not a reason to fail the load: the
        classifier keeps the type it was constructed with and the caller is warned,
        so training still runs with the default algorithm.

        Args:
            recorded_type: Algorithm recorded in the training file.
            override: Algorithm requested by the caller, taking precedence over
                ``recorded_type``. ``None`` to use what the file recorded.
        """
        requested = override if override is not None else recorded_type
        if requested in self._supported_classifiers:
            self.set_classifier(requested)
        else:
            logger.warning(
                "Specified classifier type %s is unavailable, using default: %s",
                requested.name,
                self.classifier_type.name,
            )

    def set_dict_settings(self, settings: dict) -> None:
        """Assign classifier settings from a dictionary.

        Args:
            settings: dict of settings (same structure as
                ``project.settings_manager.get_behavior``).
        """
        self._project_settings = dict(settings)

    def classifier_choices(self) -> dict[ClassifierType, str]:
        """Return the available classifier types as a sorted display map.

        Returns:
            dict mapping ``ClassifierType`` enum values to their string names.
        """
        return {t: t.value for t in sorted(self._supported_classifiers, key=lambda t: t.value)}

    def _create_classifier(self, random_seed: int | None = None) -> typing.Any:
        """Instantiate the underlying sklearn/xgboost/catboost classifier."""
        factory = factories.get_factory(self._classifier_type, multiclass=self._MULTICLASS)
        return factory(self._n_jobs, random_seed)

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Replace ±inf/NaN in feature matrix per classifier type."""
        return classifier_utils.clean_features(features, self._classifier_type)

    def _get_features_to_classify(self, features: pd.DataFrame) -> pd.DataFrame:
        """Reorder/select feature columns to match the trained model.

        Args:
            features: DataFrame of feature data to filter.

        Returns:
            DataFrame containing only the columns the trained model expects,
            in the order the model expects them.

        Raises:
            RuntimeError: If feature names cannot be obtained from the model.
        """
        if self._classifier_type == ClassifierType.XGBOOST:
            classifier_columns = self._classifier.get_booster().feature_names
        elif self._classifier_type == ClassifierType.LIGHTGBM:
            # LightGBM records only its own sanitized copies of the feature names, so
            # select the columns by the names recorded at training, in training order.
            classifier_columns = self._feature_names
        elif hasattr(self._classifier, "feature_names_in_"):
            classifier_columns = list(self._classifier.feature_names_in_)
        elif hasattr(self._classifier, "feature_names_"):
            classifier_columns = list(self._classifier.feature_names_)
        else:
            raise RuntimeError("Error obtaining feature names from classifier.")
        return features[classifier_columns]

    @staticmethod
    def combine_data(per_frame: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
        """Combine per-frame and window feature DataFrames into one."""
        return classifier_utils.combine_data(per_frame, window)

    @staticmethod
    def derive_predictions(
        probabilities: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.floating]]:
        """Derive class predictions and confidence from class probabilities.

        Args:
            probabilities: Array of shape ``(n_frames, n_classes)`` of predicted
                class probabilities.

        Returns:
            Tuple ``(predictions, confidence)`` where ``predictions`` is the
            argmax class index per frame (``-1`` if confidence is zero,
            indicating no pose) and ``confidence`` is the probability of the
            chosen class.
        """
        predictions = np.argmax(probabilities, axis=1).astype(np.int8)
        confidence = probabilities[np.arange(len(probabilities)), predictions]
        predictions[confidence == 0] = -1
        return predictions, confidence

    def get_feature_importance(self, limit: int = 20) -> list[tuple[str, float]]:
        """Return ranked feature importances, highest first.

        Args:
            limit: Maximum number of features to return.

        Returns:
            List of ``(feature_name, importance)`` tuples sorted by importance
            descending. Returns an empty list if the classifier is untrained or
            does not expose feature importances.
        """
        if self._classifier is None or self._feature_names is None:
            return []
        if not hasattr(self._classifier, "feature_importances_"):
            return []
        importances = np.asarray(self._classifier.feature_importances_, dtype=float).reshape(-1)
        # scale to fractions of the total so importances compare across classifier types
        total = importances.sum()
        if total > 0:
            importances = importances / total
        ranked = sorted(
            zip(self._feature_names, importances.tolist(), strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        # rank on the raw values; rounding first would tie small importances
        return [(feature, round(importance, 2)) for feature, importance in ranked[:limit]]

    @staticmethod
    def _training_feature_version(data: dict) -> int:
        """Return the feature version of training data, refusing exports from another one.

        Data collected from a project carries no ``file_version`` key and has the
        current ``FEATURE_VERSION``. Data loaded from an export carries the
        export's ``file_version``, which must equal the current one.

        Args:
            data: The training data dict passed to ``train``.

        Returns:
            The feature version to record on the trained classifier.

        Raises:
            ClassifierFeatureVersionError: If the data was exported with another
                feature version or records none.
        """
        if "file_version" in data:
            return check_training_data_feature_version(data["file_version"])
        return _current_feature_version()

    def reset_persistence_identity(self) -> None:
        """Clear the recorded file identity (path, hash, source).

        ``save()`` only (re)computes ``_classifier_hash`` when
        ``_classifier_file`` is ``None``, so an already-persisted classifier
        keeps its previous hash even if its contents change. Call this after
        mutating persisted state (e.g. renaming a class) so the next ``save()``
        records a hash matching the rewritten file. ``train()`` calls this
        after refitting, since a refit classifier no longer matches any file.
        """
        self._classifier_file = None
        self._classifier_hash = None
        self._classifier_source = None

    def _record_persistence_identity(self, path: Path, source: str) -> None:
        """Record the file this classifier's persisted state came from.

        The counterpart to :meth:`reset_persistence_identity`: it sets the three
        identity fields that method clears, so the two stay in step if the set
        of fields ever changes.

        Args:
            path: File the classifier was written to or read from. Its basename
                is stored and its contents are hashed.
            source: How the classifier reached that file, e.g. ``"serialized"``,
                ``"pickle"`` or ``"training_file"``.
        """
        self._classifier_file = Path(path).name
        self._classifier_hash = hash_file(Path(path))
        self._classifier_source = source

    def save(self, path: Path) -> None:
        """Serialize the classifier to disk using joblib.

        Args:
            path: Destination file path.
        """
        joblib.dump(self, path)
        if self._classifier_file is None:
            self._record_persistence_identity(path, "serialized")

    @classmethod
    def from_pickle(cls, path: Path) -> BaseClassifier:
        """Load a classifier from a pickle file with full validation and metadata backfill.

        Applies the same version, classifier-type, and metadata checks as
        :meth:`load`, but as a classmethod factory so no dummy instance is
        required. The class of the returned object is determined by the
        calling class - ``Classifier.from_pickle(...)`` rejects pickled
        ``MultiClassClassifier`` instances and vice versa.

        Args:
            path: Path to the saved classifier pickle file.

        Returns:
            Loaded and validated classifier instance of type ``cls``.

        Raises:
            UnsupportedClassifierError: If the classifier uses an algorithm
                that is not available in this environment.
            ClassifierFeatureVersionError: If the classifier was trained on
                another feature version (or records none) and must be retrained.
            ValueError: If the file is not an instance of ``cls`` or was trained
                with an incompatible sklearn or JABS version.
        """
        c = unpickle_classifier(path)

        if not isinstance(c, cls):
            raise ValueError(f"{path} is not an instance of {cls.__name__}")

        if c._version != cls._VERSION:
            raise ValueError(
                f"Unable to deserialize pickled classifier. "
                f"File version {c._version}, expected {cls._VERSION}."
            )

        if c._classifier_type not in cls._supported_classifier_choices():
            raise UnsupportedClassifierError(c._classifier_type.value)

        check_classifier_feature_version(c, path)

        if c._classifier_file is None:
            c._record_persistence_identity(path, "pickle")

        return c

    def load(self, path: Path) -> None:
        """Deserialize a classifier from disk, updating this instance in place.

        Args:
            path: Source file path.

        Raises:
            UnsupportedClassifierError: If the classifier uses an algorithm
                that is not available in this environment.
            ClassifierFeatureVersionError: If the classifier was trained on
                another feature version (or records none) and must be retrained.
            ValueError: If the file is not an instance of this class or was saved
                with a different version.
        """
        c = unpickle_classifier(path)

        if not isinstance(c, type(self)):
            raise ValueError(f"{path} is not an instance of {type(self).__name__}")

        if c._version != self._VERSION:
            raise ValueError(
                f"Unable to deserialize pickled classifier. "
                f"File version {c._version}, expected {self._VERSION}."
            )

        if c._classifier_type not in self._supported_classifiers:
            raise UnsupportedClassifierError(c._classifier_type.value)

        check_classifier_feature_version(c, path)

        for attr in self._PERSISTED_REQUIRED:
            setattr(self, attr, getattr(c, attr))
        for attr in self._PERSISTED_OPTIONAL:
            setattr(self, attr, getattr(c, attr, None))

        if c._classifier_file is not None:
            self._classifier_file = c._classifier_file
            self._classifier_hash = c._classifier_hash
            self._classifier_source = c._classifier_source
        else:
            self._record_persistence_identity(path, "pickle")
