"""Per-mode training strategies used by :class:`TrainingThread`.

Two strategies — :class:`BinaryTrainingStrategy` and
:class:`MultiClassTrainingStrategy` — implement the mode-specific pieces of
the training pipeline (feature collection, final-train data dict, classifier
save target, report content, and the secondary CV metric reported to the
session tracker). The orchestrator in :class:`TrainingThread` consumes the
strategy and stays mode-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from jabs.classifier import (
    Classifier,
    MultiClassClassifier,
    TrainingReportData,
    classifier_utils,
)

if TYPE_CHECKING:
    from jabs.classifier import CrossValidationResult
    from jabs.core.enums import CrossValidationGroupingStrategy
    from jabs.project import Project


class TrainingStrategy:
    """Per-mode hooks for the classifier training pipeline."""

    def __init__(
        self,
        classifier: Classifier | MultiClassClassifier,
        project: Project,
        behavior: str,
    ) -> None:
        self._classifier = classifier
        self._project = project
        self._behavior = behavior

    def collect_features(
        self,
        progress_callable: Callable[[], None],
        should_terminate_callable: Callable[[], None],
    ) -> tuple[dict, dict]:
        """Collect labeled features and group mapping for this mode."""
        raise NotImplementedError

    def effective_settings(self) -> dict:
        """Return the settings used for both feature extraction and training."""
        raise NotImplementedError

    def final_train_data(
        self,
        features: dict,
        full_dataset: pd.DataFrame,
        feature_names: list[str],
    ) -> dict:
        """Build the data dict passed to ``classifier.train`` for the final model."""
        raise NotImplementedError

    def save_classifier(self) -> None:
        """Persist the trained classifier."""
        raise NotImplementedError

    def build_report_data(
        self,
        features: dict,
        cv_results: list[CrossValidationResult],
        final_top_features: list[tuple[str, float]],
        elapsed_ms: int,
        timestamp: datetime,
        cv_grouping_strategy: CrossValidationGroupingStrategy,
        distance_unit: str,
        settings: dict,
    ) -> TrainingReportData:
        """Assemble the ``TrainingReportData`` for the trained model."""
        raise NotImplementedError

    def cv_secondary_metric(self, cv_results: list[CrossValidationResult]) -> float | None:
        """Mean of the mode-specific secondary CV metric, or ``None``."""
        raise NotImplementedError


class BinaryTrainingStrategy(TrainingStrategy):
    """Training pipeline for the binary behavior-vs-not-behavior classifier."""

    def __init__(
        self,
        classifier: Classifier,
        project: Project,
        behavior: str,
        bout_counts: tuple[int, int],
    ) -> None:
        super().__init__(classifier, project, behavior)
        self._bout_counts = bout_counts

    def collect_features(
        self,
        progress_callable: Callable[[], None],
        should_terminate_callable: Callable[[], None],
    ) -> tuple[dict, dict]:
        """Collect labeled features for the configured behavior."""
        return self._project.get_labeled_features(
            self._behavior,
            progress_callable=progress_callable,
            should_terminate_callable=should_terminate_callable,
        )

    def effective_settings(self) -> dict:
        """Return the behavior-scoped settings from the project settings manager."""
        return self._project.settings_manager.get_behavior(self._behavior)

    def final_train_data(
        self,
        features: dict,
        full_dataset: pd.DataFrame,
        feature_names: list[str],
    ) -> dict:
        """Build the binary ``classifier.train`` payload from the combined dataset."""
        return {
            "training_data": full_dataset,
            "training_labels": features["labels"],
            "feature_names": feature_names,
        }

    def save_classifier(self) -> None:
        """Persist the classifier under its behavior-scoped pickle name."""
        self._project.save_classifier(self._classifier, self._behavior)

    def build_report_data(
        self,
        features: dict,
        cv_results: list[CrossValidationResult],
        final_top_features: list[tuple[str, float]],
        elapsed_ms: int,
        timestamp: datetime,
        cv_grouping_strategy: CrossValidationGroupingStrategy,
        distance_unit: str,
        settings: dict,
    ) -> TrainingReportData:
        """Build the binary-mode training report with frame and bout counts."""
        behavior_count = int(np.sum(features["labels"] == 1))
        not_behavior_count = int(np.sum(features["labels"] == 0))
        behavior_bouts, not_behavior_bouts = self._bout_counts
        return TrainingReportData(
            behavior_name=self._behavior,
            classifier_type=self._classifier.classifier_name,
            balance_training_labels=settings.get("balance_labels", False),
            symmetric_behavior=settings.get("symmetric_behavior", False),
            distance_unit=distance_unit,
            cv_results=cv_results,
            final_top_features=final_top_features,
            frames_behavior=behavior_count,
            frames_not_behavior=not_behavior_count,
            bouts_behavior=behavior_bouts,
            bouts_not_behavior=not_behavior_bouts,
            training_time_ms=elapsed_ms,
            timestamp=timestamp,
            window_size=settings["window_size"],
            cv_grouping_strategy=cv_grouping_strategy,
        )

    def cv_secondary_metric(self, cv_results: list[CrossValidationResult]) -> float | None:
        """Mean f1 for the behavior class across CV folds."""
        if not cv_results:
            return None
        return float(np.mean([cv.f1_behavior for cv in cv_results]))


class MultiClassTrainingStrategy(TrainingStrategy):
    """Training pipeline for the multi-class behavior classifier."""

    def __init__(
        self,
        classifier: MultiClassClassifier,
        project: Project,
        behavior: str,
    ) -> None:
        super().__init__(classifier, project, behavior)
        # Resolve effective settings once so feature extraction and the final
        # training call see identical window/balance parameters even if the
        # classifier mutates its own ``project_settings`` during training.
        self._settings = classifier.project_settings or project.get_project_defaults()

    def collect_features(
        self,
        progress_callable: Callable[[], None],
        should_terminate_callable: Callable[[], None],
    ) -> tuple[dict, dict]:
        """Collect labeled features across every behavior in one pass."""
        return self._project.get_multiclass_labeled_features(
            progress_callable=progress_callable,
            should_terminate_callable=should_terminate_callable,
            behavior_settings=self._settings,
        )

    def effective_settings(self) -> dict:
        """Return the captured project settings used at construction time."""
        return self._settings

    def final_train_data(
        self,
        features: dict,
        full_dataset: pd.DataFrame,
        feature_names: list[str],
    ) -> dict:
        """Build the multi-class ``classifier.train`` payload (per-behavior labels)."""
        return {
            "per_frame": features["per_frame"],
            "window": features["window"],
            "labels_by_behavior": features["labels_by_behavior"],
            "settings": self._settings,
            "feature_names": feature_names,
        }

    def save_classifier(self) -> None:
        """Persist the classifier under the shared multi-class pickle name."""
        self._project.save_classifier(self._classifier)

    def build_report_data(
        self,
        features: dict,
        cv_results: list[CrossValidationResult],
        final_top_features: list[tuple[str, float]],
        elapsed_ms: int,
        timestamp: datetime,
        cv_grouping_strategy: CrossValidationGroupingStrategy,
        distance_unit: str,
        settings: dict,
    ) -> TrainingReportData:
        """Build the multi-class training report with per-class frame and bout counts."""
        class_names = self._classifier.get_class_names()
        behavior_names = self._classifier.behavior_names

        merged_labels, _ = classifier_utils.merge_labels(
            features["labels_by_behavior"], behavior_names
        )
        class_frame_counts = {
            name: int(np.sum(merged_labels == class_idx))
            for class_idx, name in enumerate(class_names)
        }
        class_bout_counts: dict[str, int] = {}
        for class_name in class_names:
            bouts = 0
            for video_counts in self._project.counts(class_name).values():
                for identity_counts in video_counts.values():
                    bouts += identity_counts["unfragmented_bout_counts"][0]
            class_bout_counts[class_name] = bouts

        return TrainingReportData(
            behavior_name=self._behavior,
            classifier_type=self._classifier.classifier_name,
            balance_training_labels=settings.get("balance_labels", False),
            symmetric_behavior=settings.get("symmetric_behavior", False),
            distance_unit=distance_unit,
            cv_results=cv_results,
            final_top_features=final_top_features,
            training_time_ms=elapsed_ms,
            timestamp=timestamp,
            window_size=settings.get("window_size", 0),
            cv_grouping_strategy=cv_grouping_strategy,
            class_frame_counts=class_frame_counts,
            class_bout_counts=class_bout_counts,
        )

    def cv_secondary_metric(self, cv_results: list[CrossValidationResult]) -> float | None:
        """Mean macro F1 across CV folds, or ``None`` if no folds reported it."""
        if not cv_results:
            return None
        f1_macro = [cv.f1_macro for cv in cv_results if getattr(cv, "f1_macro", None) is not None]
        return float(np.mean(f1_macro)) if f1_macro else None
