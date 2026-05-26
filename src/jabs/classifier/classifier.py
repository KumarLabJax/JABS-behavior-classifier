"""Binary behavior classifier (behavior vs. not-behavior)."""

import logging
import warnings
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from jabs.core.enums import (
    DEFAULT_CV_GROUPING_STRATEGY,
    ClassifierType,
    CrossValidationGroupingStrategy,
)
from jabs.core.utils import hash_file
from jabs.project import Project, load_training_data

from . import classifier_utils
from .base import BaseClassifier

logger = logging.getLogger(__name__)


class Classifier(BaseClassifier):
    """A binary behavior classifier (behavior vs. not-behavior).

    Supports training, evaluating, saving, and loading classifiers for
    behavioral data using Random Forest, CatBoost, or XGBoost algorithms.
    Persistence and identity machinery are inherited from
    :class:`BaseClassifier`.

    Attributes:
        LABEL_THRESHOLD: Minimum number of labels required per group.
    """

    LABEL_THRESHOLD: ClassVar[int] = classifier_utils.LABEL_THRESHOLD

    _VERSION: ClassVar[int] = 11
    _MULTICLASS: ClassVar[bool] = False
    _PERSISTED_REQUIRED: ClassVar[tuple[str, ...]] = (
        "_classifier",
        "_behavior",
        "_project_settings",
        "_classifier_type",
    )

    def __init__(
        self,
        classifier: ClassifierType = ClassifierType.RANDOM_FOREST,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(classifier_type=classifier, n_jobs=n_jobs)
        self._behavior: str | None = None

    @classmethod
    def from_training_file(
        cls, path: Path, classifier_type: ClassifierType | None = None
    ) -> "Classifier":
        """Initialize a classifier from an exported training data file.

        This method loads the training data and trains a classifier.

        Args:
            path: exported training data file
            classifier_type: Override the classifier algorithm stored in the training
                file. If ``None``, the type recorded in the file is used.

        Returns:
            trained Classifier object
        """
        loaded_training_data, _ = load_training_data(path)
        behavior = loaded_training_data["behavior"]

        classifier = cls()
        classifier.behavior_name = behavior
        classifier.set_dict_settings(loaded_training_data["settings"])
        file_classifier_type = ClassifierType(loaded_training_data["classifier_type"])
        effective_type = classifier_type if classifier_type is not None else file_classifier_type
        if effective_type in classifier._supported_classifiers:
            classifier.set_classifier(effective_type)
        else:
            logger.warning(
                "Specified classifier type %s is unavailable, using default: %s",
                effective_type.name,
                classifier.classifier_type.name,
            )
        training_features = classifier.combine_data(
            loaded_training_data["per_frame"], loaded_training_data["window"]
        )
        classifier.train(
            {
                "training_data": training_features,
                "training_labels": loaded_training_data["labels"],
            },
            random_seed=loaded_training_data["training_seed"],
        )

        classifier._classifier_file = Path(path).name
        classifier._classifier_hash = hash_file(Path(path))
        classifier._classifier_source = "training_file"

        return classifier

    @property
    def behavior_name(self) -> str | None:
        """Return the behavior name property."""
        return self._behavior

    @behavior_name.setter
    def behavior_name(self, value: str | None) -> None:
        """Set the behavior name property."""
        self._behavior = value

    @staticmethod
    def get_leave_one_group_out_max(labels: np.ndarray, groups: np.ndarray) -> int:
        """Count the number of possible leave-one-group-out splits.

        Args:
            labels: Labels to check against the per-class threshold.
            groups: Group id corresponding to each label.

        Returns:
            Number of groups that can serve as a valid test split.

        Note: labels excludes label for frames with no identity.
        """
        return classifier_utils.count_valid_logo_splits(
            labels, groups, label_threshold=Classifier.LABEL_THRESHOLD
        )

    @staticmethod
    def leave_one_group_out(
        per_frame_features: pd.DataFrame,
        window_features: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
    ):
        """Yield "leave one group out" train/test splits.

        Args:
            per_frame_features: per frame features for all labeled data
            window_features: window features for all labeled data
            labels: labels corresponding to each feature row
            groups: group id corresponding to each feature row

        Yields:
            Dict with training_data, test_data, training_labels, test_labels,
            and feature_names.
        """
        yield from classifier_utils.leave_one_group_out(
            per_frame_features,
            window_features,
            labels,
            groups,
            label_threshold=Classifier.LABEL_THRESHOLD,
        )

    @staticmethod
    def downsample_balance(
        features: pd.DataFrame, labels: np.ndarray, random_seed: int | None = None
    ):
        """Downsample features and labels to an equal class distribution."""
        return classifier_utils.downsample_balance(features, labels, random_seed)

    @staticmethod
    def augment_symmetric(
        features: pd.DataFrame, labels: np.ndarray, random_str: str = "ASygRQDZJD"
    ):
        """Augment features with left/right reflected duplicates."""
        return classifier_utils.augment_symmetric(features, labels, random_str)

    def set_project_settings(self, project: Project) -> None:
        """Assign project settings to the classifier.

        If no behavior is currently set, uses project defaults; otherwise looks
        up the behavior-scoped settings from the project's settings manager.

        Args:
            project: Project to copy classifier-relevant settings from.
        """
        if self._behavior is None:
            self._project_settings = project.get_project_defaults()
        else:
            self._project_settings = project.settings_manager.get_behavior(self._behavior)

    def train(self, data: dict, random_seed: int | None = None) -> None:
        """Train the classifier.

        Args:
            data: dict returned from train_test_split().
            random_seed: optional random seed for reproducibility.

        Raises:
            ValueError: If project settings are unset.
        """
        if self._project_settings is None:
            raise ValueError("Project settings for classifier unset, cannot train classifier.")

        if "feature_names" in data:
            self._feature_names = data["feature_names"]
        else:
            self._feature_names = data["training_data"].columns.to_list()

        features = data["training_data"]
        labels = data["training_labels"]
        # Symmetric augmentation should occur before balancing so that the
        # class with more labels can sample from the whole set.
        if self._project_settings.get("symmetric_behavior", False):
            features, labels = self.augment_symmetric(features, labels)
        if self._project_settings.get("balance_labels", False):
            features, labels = self.downsample_balance(features, labels, random_seed)

        classifier = self._create_classifier(random_seed=random_seed)
        cleaned_features = self._clean_features(features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self._classifier = classifier.fit(cleaned_features, labels)

        self._classifier_file = None
        self._classifier_hash = None
        self._classifier_source = None

    def predict(
        self, features: pd.DataFrame, frame_indexes: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict classes for a given set of features.

        Args:
            features: DataFrame of feature data to classify.
            frame_indexes: Frame indexes to classify (default all).

        Returns:
            Predicted class vector. Frames absent from ``frame_indexes`` are
            assigned -1.
        """
        cleaned_features = self._get_features_to_classify(self._clean_features(features))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            result = self._classifier.predict(cleaned_features)

        if frame_indexes is not None:
            result_adjusted = np.full(result.shape, -1, dtype=np.int8)
            result_adjusted[frame_indexes] = result[frame_indexes]
            result = result_adjusted

        return result

    def predict_proba(
        self, features: pd.DataFrame, frame_indexes: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict probabilities for a given set of features.

        Args:
            features: DataFrame of feature data to classify.
            frame_indexes: Frame indexes to classify (default all).

        Returns:
            Prediction probability matrix. Frames absent from ``frame_indexes``
            are assigned zero probabilities.
        """
        cleaned_features = self._get_features_to_classify(self._clean_features(features))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            result = self._classifier.predict_proba(cleaned_features)

        if frame_indexes is not None:
            result_adjusted = np.full(result.shape, 0, dtype=np.float32)
            result_adjusted[frame_indexes] = result[frame_indexes]
            result = result_adjusted

        return result

    def print_feature_importance(self, limit: int = 20) -> None:
        """Print the most important features and their importance.

        Args:
            limit: Maximum number of features to print.
        """
        feature_importance = self.get_feature_importance(limit=limit)
        print(f"{'Feature Name':100} Importance")
        print("-" * 120)
        for feature, importance in feature_importance[:limit]:
            print(f"{feature:100} {importance:0.2f}")

    @staticmethod
    def accuracy_score(truth: np.ndarray, predictions: np.ndarray) -> float:
        """Return accuracy score."""
        return classifier_utils.accuracy_score(truth, predictions)

    @staticmethod
    def precision_recall_score(truth: np.ndarray, predictions: np.ndarray):
        """Return precision/recall/f-score/support."""
        return classifier_utils.precision_recall_score(truth, predictions)

    @staticmethod
    def confusion_matrix(truth: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Return the confusion matrix."""
        return classifier_utils.confusion_matrix(truth, predictions)

    @staticmethod
    def count_label_threshold(
        all_counts: dict,
        cv_grouping_strategy: CrossValidationGroupingStrategy = DEFAULT_CV_GROUPING_STRATEGY,
    ) -> int:
        """Count groups that meet the label-threshold criteria.

        Args:
            all_counts: Labeled frame and bout counts for the entire project.
                Structure is a dict[video_name][identity] of fragmented and
                unfragmented frame/bout count tuples.
            cv_grouping_strategy: Cross-validation grouping strategy.

        Returns:
            Number of groups that meet the labeling threshold criteria.

        Note:
            Uses "fragmented" label counts since these reflect labels usable
            for training.
        """
        group_count = 0
        if cv_grouping_strategy == CrossValidationGroupingStrategy.INDIVIDUAL:
            for video in all_counts:
                for identity_count in all_counts[video].values():
                    if (
                        identity_count["fragmented_frame_counts"][0] >= Classifier.LABEL_THRESHOLD
                        and identity_count["fragmented_frame_counts"][1]
                        >= Classifier.LABEL_THRESHOLD
                    ):
                        group_count += 1
        elif cv_grouping_strategy == CrossValidationGroupingStrategy.VIDEO:
            for video in all_counts:
                behavior_sum = 0
                not_behavior_sum = 0
                for identity_count in all_counts[video].values():
                    behavior_sum += identity_count["fragmented_frame_counts"][0]
                    not_behavior_sum += identity_count["fragmented_frame_counts"][1]
                if (
                    behavior_sum >= Classifier.LABEL_THRESHOLD
                    and not_behavior_sum >= Classifier.LABEL_THRESHOLD
                ):
                    group_count += 1
        else:
            raise ValueError(f"Unknown cv_grouping_strategy: {cv_grouping_strategy}")
        return group_count

    @staticmethod
    def label_threshold_met(
        all_counts: dict,
        min_groups: int,
        cv_grouping_strategy: CrossValidationGroupingStrategy = DEFAULT_CV_GROUPING_STRATEGY,
    ) -> bool:
        """Determine whether the labeling threshold is met.

        Args:
            all_counts: Labeled frame and bout counts for the entire project.
            min_groups: Minimum number of groups required.
            cv_grouping_strategy: Cross-validation grouping strategy.

        Returns:
            True if there are enough groups meeting the threshold.
        """
        group_count = Classifier.count_label_threshold(
            all_counts, cv_grouping_strategy=cv_grouping_strategy
        )
        return 1 < group_count >= min_groups
