"""Cross-validation utilities for JABS classifier training."""

from collections.abc import Callable
from typing import TYPE_CHECKING, NotRequired, TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR

from . import classifier_utils
from .training_report import BinaryCVResult, CrossValidationResult, MultiClassCVResult

if TYPE_CHECKING:
    from jabs.classifier import Classifier, MultiClassClassifier
    from jabs.project import Project


class CVFeatures(TypedDict):
    """Feature payload used by cross-validation helper."""

    per_frame: pd.DataFrame
    window: pd.DataFrame
    groups: np.ndarray
    labels: NotRequired[np.ndarray]
    labels_by_behavior: NotRequired[dict[str, np.ndarray]]


def run_leave_one_group_out_cv(
    classifier: "Classifier | MultiClassClassifier",
    project: "Project",
    features: CVFeatures,
    group_mapping: dict,
    behavior: str,
    k: int = 1,
    status_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[], None] | None = None,
    terminate_callback: Callable[[], None] | None = None,
) -> list[CrossValidationResult]:
    """
    Run leave-one-group-out cross-validation for a classifier.

    Args:
        classifier: Classifier instance to train.
        project: Project instance containing data and settings.
        features: Dictionary containing features and labels
        group_mapping: Mapping of cross validation groups to labeled feature rows.
        behavior: Behavior label to train on.
        k: Number of cross-validation splits (int or np.inf for all splits).
        status_callback: Optional callback for status updates (str argument).
        progress_callback: Optional callback for progress updates (no arguments).
        terminate_callback: Optional callback to check for early termination (no arguments, should raise if terminate requested).

    Returns:
        List of CrossValidationResult instances summarizing cross-validation results.
    """

    def emit_status(msg):
        if status_callback:
            status_callback(msg)

    def emit_progress():
        if progress_callback:
            progress_callback()
        if terminate_callback:
            terminate_callback()

    is_multiclass = "labels_by_behavior" in features
    labels = features.get("labels")
    class_names: list[str] | None = None
    multiclass_settings: dict | None = None

    if is_multiclass:
        behavior_names = list(getattr(classifier, "behavior_names", []))
        labels, _ = classifier_utils.merge_labels(features["labels_by_behavior"], behavior_names)
        class_names = [MULTICLASS_NONE_BEHAVIOR, *behavior_names]
        multiclass_settings = classifier.project_settings or project.get_project_defaults()

    cv_results = []
    if k > 0:
        max_splits = classifier.get_leave_one_group_out_max(labels, features["groups"])
        if max_splits == 0:
            emit_status("No valid cross-validation splits found; skipping CV")
            return cv_results
        if k == np.inf:
            k = max_splits
        elif k > max_splits:
            emit_status(
                f"Requested {k} cross-validation splits, but only {max_splits} are valid; using {max_splits}"
            )
            k = max_splits

    emit_status("Generating train/test splits")
    data_generator = classifier.leave_one_group_out(
        features["per_frame"],
        features["window"],
        labels,
        features["groups"],
    )

    if k > 0:
        for i, data in enumerate(data_generator):
            if terminate_callback:
                terminate_callback()
            if i + 1 > k:
                break
            emit_status(f"cross validation iteration {i + 1} of {k}")
            test_info = group_mapping[data["test_group"]]
            if is_multiclass:
                if multiclass_settings is None:
                    raise RuntimeError("Internal error: multiclass settings were not initialized")
                train_idx = data["training_idx"]
                labels_by_behavior = {
                    name: arr[train_idx] for name, arr in features["labels_by_behavior"].items()
                }
                classifier.train(
                    {
                        "per_frame": features["per_frame"].iloc[train_idx],
                        "window": features["window"].iloc[train_idx],
                        "labels_by_behavior": labels_by_behavior,
                        "settings": multiclass_settings,
                        "feature_names": data["feature_names"],
                    }
                )
            else:
                classifier.set_project_settings(project)
                classifier.behavior_name = behavior
                classifier.train(data)
            predictions = classifier.predict(data["test_data"])
            accuracy = classifier_utils.accuracy_score(data["test_labels"], predictions)
            confusion = classifier_utils.confusion_matrix(data["test_labels"], predictions)
            top_features = classifier.get_feature_importance(limit=10)
            test_label = (
                f"{test_info['video']} [{test_info['identity']}]"
                if test_info["identity"] is not None
                else test_info["video"]
            )

            if is_multiclass and class_names is not None:
                class_idx = np.arange(len(class_names))
                precision, recall, f1, support = precision_recall_fscore_support(
                    data["test_labels"],
                    predictions,
                    labels=class_idx,
                    zero_division=0,
                )
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    data["test_labels"],
                    predictions,
                    average="macro",
                    zero_division=0,
                )
                precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                    data["test_labels"],
                    predictions,
                    average="micro",
                    zero_division=0,
                )
                per_class_metrics = [
                    {
                        "class_name": name,
                        "precision": float(precision[idx]),
                        "recall": float(recall[idx]),
                        "f1": float(f1[idx]),
                        "support": int(support[idx]),
                    }
                    for idx, name in enumerate(class_names)
                ]
                cv_results.append(
                    MultiClassCVResult(
                        iteration=i + 1,
                        test_label=test_label,
                        accuracy=accuracy,
                        confusion_matrix=confusion,
                        top_features=top_features,
                        class_names=class_names,
                        class_support=[int(x) for x in support],
                        per_class_metrics=per_class_metrics,
                        precision_macro=float(precision_macro),
                        recall_macro=float(recall_macro),
                        f1_macro=float(f1_macro),
                        precision_micro=float(precision_micro),
                        recall_micro=float(recall_micro),
                        f1_micro=float(f1_micro),
                    )
                )
            else:
                pr = classifier_utils.precision_recall_score(data["test_labels"], predictions)
                cv_results.append(
                    BinaryCVResult(
                        iteration=i + 1,
                        test_label=test_label,
                        accuracy=accuracy,
                        confusion_matrix=confusion,
                        top_features=top_features,
                        precision_behavior=float(pr[0][1]),
                        precision_not_behavior=float(pr[0][0]),
                        recall_behavior=float(pr[1][1]),
                        recall_not_behavior=float(pr[1][0]),
                        f1_behavior=float(pr[2][1]),
                        support_behavior=int(pr[3][1]),
                        support_not_behavior=int(pr[3][0]),
                    )
                )
            emit_progress()
    return cv_results
