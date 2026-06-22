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


def _included_row_mask(features: dict) -> np.ndarray | None:
    """Boolean mask selecting rows whose group is not excluded from training.

    The final classifier is trained only on included videos. Excluded videos
    still appear in ``features`` (so they can serve as cross-validation holdout
    groups) and must be filtered out before the final fit.

    Args:
        features: Feature payload from ``Project.get_*labeled_features``.

    Returns:
        A boolean mask aligned to the feature rows, or ``None`` when no groups
        are excluded (so callers can skip filtering entirely).
    """
    excluded = features.get("excluded_groups")
    if not excluded:
        return None
    return ~np.isin(features["groups"], list(excluded))


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
        cv_grouping_regex: str | None = None,
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
        """Build the binary ``classifier.train`` payload from the combined dataset.

        Rows belonging to videos excluded from training are dropped here so the
        final classifier trains only on included data.
        """
        mask = _included_row_mask(features)
        if mask is None:
            training_data = full_dataset
            training_labels = features["labels"]
        else:
            training_data = full_dataset[mask].reset_index(drop=True)
            training_labels = features["labels"][mask]
        return {
            "training_data": training_data,
            "training_labels": training_labels,
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
        cv_grouping_regex: str | None = None,
    ) -> TrainingReportData:
        """Build the binary-mode training report with frame and bout counts.

        Frame counts reflect only the videos trained on; rows from excluded
        videos are filtered out. (Bout counts arrive pre-filtered via
        ``self._bout_counts``.)
        """
        mask = _included_row_mask(features)
        labels = features["labels"] if mask is None else features["labels"][mask]
        behavior_count = int(np.sum(labels == 1))
        not_behavior_count = int(np.sum(labels == 0))
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
            cv_grouping_regex=cv_grouping_regex,
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
        """Build the multi-class ``classifier.train`` payload (per-behavior labels).

        Rows belonging to videos excluded from training are dropped here so the
        final classifier trains only on included data.
        """
        mask = _included_row_mask(features)
        if mask is None:
            per_frame = features["per_frame"]
            window = features["window"]
            labels_by_behavior = features["labels_by_behavior"]
        else:
            per_frame = features["per_frame"][mask].reset_index(drop=True)
            window = features["window"][mask].reset_index(drop=True)
            labels_by_behavior = {
                name: arr[mask] for name, arr in features["labels_by_behavior"].items()
            }
        return {
            "per_frame": per_frame,
            "window": window,
            "labels_by_behavior": labels_by_behavior,
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
        cv_grouping_regex: str | None = None,
    ) -> TrainingReportData:
        """Build the multi-class training report with per-class frame and bout counts.

        Frame and bout counts reflect only the videos trained on; rows and videos
        excluded from training are filtered out.
        """
        class_names = self._classifier.get_class_names()
        behavior_names = self._classifier.behavior_names

        mask = _included_row_mask(features)
        if mask is None:
            labels_by_behavior = features["labels_by_behavior"]
        else:
            labels_by_behavior = {
                name: arr[mask] for name, arr in features["labels_by_behavior"].items()
            }
        merged_labels, _ = classifier_utils.merge_labels(labels_by_behavior, behavior_names)
        class_frame_counts = {
            name: int(np.sum(merged_labels == class_idx))
            for class_idx, name in enumerate(class_names)
        }
        settings_manager = self._project.settings_manager
        class_bout_counts: dict[str, int] = {}
        for class_name in class_names:
            bouts = 0
            for video, video_counts in self._project.counts(class_name).items():
                if settings_manager.is_video_excluded(video):
                    continue
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
            cv_grouping_regex=cv_grouping_regex,
            class_frame_counts=class_frame_counts,
            class_bout_counts=class_bout_counts,
        )

    def cv_secondary_metric(self, cv_results: list[CrossValidationResult]) -> float | None:
        """Mean macro F1 across CV folds, or ``None`` if no folds reported it."""
        if not cv_results:
            return None
        f1_macro = [cv.f1_macro for cv in cv_results if getattr(cv, "f1_macro", None) is not None]
        return float(np.mean(f1_macro)) if f1_macro else None
