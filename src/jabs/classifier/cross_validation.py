"""Cross-validation utilities for JABS classifier training."""

from collections.abc import Callable
from typing import TYPE_CHECKING, NotRequired, TypedDict

import numpy as np
import numpy.typing as npt
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


def _prepare_cv_labels(
    classifier: "Classifier | MultiClassClassifier",
    features: CVFeatures,
    project: "Project",
    is_multiclass: bool,
) -> tuple[npt.NDArray, list[str] | None, dict | None]:
    """Compute the label array, class names, and settings used for CV.

    In binary mode the labels come straight from the features payload and no
    class-name or settings preparation is needed. In multi-class mode we merge
    per-behavior label arrays into a class-index array and capture the effective
    training settings the classifier should reuse per fold.
    """
    if not is_multiclass:
        return features["labels"], None, None

    behavior_names = list(getattr(classifier, "behavior_names", []))
    labels, _ = classifier_utils.merge_labels(features["labels_by_behavior"], behavior_names)
    class_names = [MULTICLASS_NONE_BEHAVIOR, *behavior_names]
    multiclass_settings = classifier.project_settings or project.get_project_defaults()
    return labels, class_names, multiclass_settings


def _resolve_k(
    classifier: "Classifier | MultiClassClassifier",
    labels: npt.NDArray,
    groups: npt.NDArray,
    k: int | float,
    emit_status: Callable[[str], None],
) -> int:
    """Resolve the requested CV iteration count against available valid splits.

    Returns 0 when no valid splits exist or the caller asked for none, signaling
    that cross-validation should be skipped.
    """
    if k <= 0:
        return 0
    max_splits = classifier.get_leave_one_group_out_max(labels, groups)
    if max_splits == 0:
        emit_status("No valid cross-validation splits found; skipping CV")
        return 0
    if k == np.inf:
        return max_splits
    if k > max_splits:
        emit_status(
            f"Requested {k} cross-validation splits, but only {max_splits} are valid; "
            f"using {max_splits}"
        )
        return max_splits
    return int(k)


def _train_binary_fold(
    classifier: "Classifier",
    project: "Project",
    behavior: str,
    data: dict,
) -> None:
    """Train a binary classifier on the training portion of one CV fold."""
    classifier.behavior_name = behavior
    classifier.set_project_settings(project, behavior)
    classifier.train(data)


def _train_multiclass_fold(
    classifier: "MultiClassClassifier",
    data: dict,
    features: CVFeatures,
    multiclass_settings: dict,
) -> None:
    """Train a multi-class classifier on the training portion of one CV fold."""
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


def _test_label_from_group(test_info: dict) -> str:
    """Render a CV test-group label for the report (video name + optional identity)."""
    if test_info["identity"] is not None:
        return f"{test_info['video']} [{test_info['identity']}]"
    return test_info["video"]


def _build_binary_cv_result(
    iteration: int,
    test_label: str,
    accuracy: float,
    confusion: npt.NDArray,
    top_features: list[tuple[str, float]],
    data: dict,
    predictions: npt.NDArray,
) -> BinaryCVResult:
    """Construct a binary CV iteration result from prediction outputs."""
    pr = classifier_utils.precision_recall_score(data["test_labels"], predictions)
    return BinaryCVResult(
        iteration=iteration,
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


def _build_multiclass_cv_result(
    iteration: int,
    test_label: str,
    accuracy: float,
    confusion: npt.NDArray,
    top_features: list[tuple[str, float]],
    data: dict,
    predictions: npt.NDArray,
    class_names: list[str],
) -> MultiClassCVResult:
    """Construct a multi-class CV iteration result from prediction outputs."""
    class_idx = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        data["test_labels"], predictions, labels=class_idx, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        data["test_labels"], predictions, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        data["test_labels"], predictions, average="micro", zero_division=0
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
    return MultiClassCVResult(
        iteration=iteration,
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
    """Run leave-one-group-out cross-validation for a classifier.

    Args:
        classifier: Classifier instance to train.
        project: Project instance containing data and settings.
        features: Dictionary containing features and labels.
        group_mapping: Mapping of cross-validation groups to labeled feature rows.
        behavior: Behavior label to train on (binary mode only).
        k: Number of cross-validation splits (int or ``np.inf`` for all splits).
        status_callback: Optional callback for status updates (str argument).
        progress_callback: Optional callback for progress updates (no arguments).
        terminate_callback: Optional callback to check for early termination
            (no arguments, should raise if termination is requested).

    Returns:
        List of cross-validation iteration results.
    """

    def emit_status(msg: str) -> None:
        if status_callback:
            status_callback(msg)

    def emit_progress() -> None:
        if progress_callback:
            progress_callback()
        if terminate_callback:
            terminate_callback()

    is_multiclass = "labels_by_behavior" in features
    labels, class_names, multiclass_settings = _prepare_cv_labels(
        classifier, features, project, is_multiclass
    )

    cv_results: list[CrossValidationResult] = []
    k = _resolve_k(classifier, labels, features["groups"], k, emit_status)
    if k == 0:
        return cv_results

    emit_status("Generating train/test splits")
    data_generator = classifier.leave_one_group_out(
        features["per_frame"], features["window"], labels, features["groups"]
    )

    for i, data in enumerate(data_generator):
        if terminate_callback:
            terminate_callback()
        if i + 1 > k:
            break
        emit_status(f"cross validation iteration {i + 1} of {k}")

        if is_multiclass:
            if multiclass_settings is None:
                raise RuntimeError("Internal error: multiclass settings were not initialized")
            _train_multiclass_fold(classifier, data, features, multiclass_settings)
        else:
            _train_binary_fold(classifier, project, behavior, data)

        predictions = classifier.predict(data["test_data"])
        accuracy = classifier_utils.accuracy_score(data["test_labels"], predictions)
        confusion = classifier_utils.confusion_matrix(data["test_labels"], predictions)
        top_features = classifier.get_feature_importance(limit=10)
        test_label = _test_label_from_group(group_mapping[data["test_group"]])

        if is_multiclass and class_names is not None:
            cv_results.append(
                _build_multiclass_cv_result(
                    i + 1,
                    test_label,
                    accuracy,
                    confusion,
                    top_features,
                    data,
                    predictions,
                    class_names,
                )
            )
        else:
            cv_results.append(
                _build_binary_cv_result(
                    i + 1,
                    test_label,
                    accuracy,
                    confusion,
                    top_features,
                    data,
                    predictions,
                )
            )
        emit_progress()
    return cv_results
