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
        self._n_jobs = n_jobs
        self._version = self._VERSION

        self._classifier_file: str | None = None
        self._classifier_hash: str | None = None
        self._classifier_source: str | None = None

        self._supported_classifiers = self._supported_classifier_choices()
        if classifier_type not in self._supported_classifiers:
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
        importances = list(np.asarray(self._classifier.feature_importances_).reshape(-1))
        feature_importance = [
            (feature, round(importance, 2))
            for feature, importance in zip(self._feature_names, importances, strict=True)
        ]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        return feature_importance[:limit]

    def reset_persistence_identity(self) -> None:
        """Clear the recorded file identity (path, hash, source).

        ``save()`` only (re)computes ``_classifier_hash`` when
        ``_classifier_file`` is ``None``, so an already-persisted classifier
        keeps its previous hash even if its contents change. Call this after
        mutating persisted state (e.g. renaming a class) so the next ``save()``
        records a hash matching the rewritten file. ``train()`` performs the
        same null-out after refitting.
        """
        self._classifier_file = None
        self._classifier_hash = None
        self._classifier_source = None

    def save(self, path: Path) -> None:
        """Serialize the classifier to disk using joblib.

        Args:
            path: Destination file path.
        """
        joblib.dump(self, path)
        if self._classifier_file is None:
            self._classifier_file = Path(path).name
            self._classifier_hash = hash_file(Path(path))
            self._classifier_source = "serialized"

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
            ValueError: If the file is not an instance of ``cls``, was trained
                with an incompatible sklearn or JABS version, or uses an
                unsupported classifier type.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            c = joblib.load(path)
            for warning in caught_warnings:
                if issubclass(warning.category, InconsistentVersionWarning):
                    raise ValueError("Classifier trained with different version of sklearn.")
                warnings.warn(warning.message, warning.category, stacklevel=2)

        if not isinstance(c, cls):
            raise ValueError(f"{path} is not an instance of {cls.__name__}")

        if c._version != cls._VERSION:
            raise ValueError(
                f"Unable to deserialize pickled classifier. "
                f"File version {c._version}, expected {cls._VERSION}."
            )

        if c._classifier_type not in cls._supported_classifier_choices():
            raise ValueError("Invalid classifier type")

        if c._classifier_file is None:
            c._classifier_file = Path(path).name
            c._classifier_hash = hash_file(Path(path))
            c._classifier_source = "pickle"

        return c

    def load(self, path: Path) -> None:
        """Deserialize a classifier from disk, updating this instance in place.

        Args:
            path: Source file path.

        Raises:
            ValueError: If the file is not an instance of this class, was saved
                with a different version, or uses an unsupported classifier type.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            c = joblib.load(path)
            for warning in caught_warnings:
                if issubclass(warning.category, InconsistentVersionWarning):
                    raise ValueError("Classifier trained with different version of sklearn.")
                warnings.warn(warning.message, warning.category, stacklevel=2)

        if not isinstance(c, type(self)):
            raise ValueError(f"{path} is not an instance of {type(self).__name__}")

        if c._version != self._VERSION:
            raise ValueError(
                f"Unable to deserialize pickled classifier. "
                f"File version {c._version}, expected {self._VERSION}."
            )

        if c._classifier_type not in self._supported_classifiers:
            raise ValueError("Invalid classifier type")

        for attr in self._PERSISTED_REQUIRED:
            setattr(self, attr, getattr(c, attr))
        for attr in self._PERSISTED_OPTIONAL:
            setattr(self, attr, getattr(c, attr, None))

        if c._classifier_file is not None:
            self._classifier_file = c._classifier_file
            self._classifier_hash = c._classifier_hash
            self._classifier_source = c._classifier_source
        else:
            self._classifier_file = Path(path).name
            self._classifier_hash = hash_file(Path(path))
            self._classifier_source = "pickle"
