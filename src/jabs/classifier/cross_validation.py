"""Cross-validation utilities for JABS classifier training."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .training_report import CrossValidationResult

if TYPE_CHECKING:
    from ..project import Project
    from . import Classifier


def run_leave_one_group_out_cv(
    classifier: "Classifier",
    project: "Project",
    features: dict[str, np.ndarray],
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

    emit_status("Generating train/test splits")
    data_generator = classifier.leave_one_group_out(
        features["per_frame"],
        features["window"],
        features["labels"],
        features["groups"],
    )

    cv_results = []
    if k == np.inf:
        k = classifier.get_leave_one_group_out_max(features["labels"], features["groups"])

    if k > 0:
        for i, data in enumerate(data_generator):
            if terminate_callback:
                terminate_callback()
            if i + 1 > k:
                break
            emit_status(f"cross validation iteration {i + 1} of {k}")
            test_info = group_mapping[data["test_group"]]
            classifier.behavior_name = behavior
            classifier.set_project_settings(project)
            classifier.train(data)
            predictions = classifier.predict(data["test_data"])
            accuracy = classifier.accuracy_score(data["test_labels"], predictions)
            pr = classifier.precision_recall_score(data["test_labels"], predictions)
            confusion = classifier.confusion_matrix(data["test_labels"], predictions)
            top_features = classifier.get_feature_importance(limit=10)
            cv_results.append(
                CrossValidationResult(
                    iteration=i + 1,
                    test_label=f"{test_info['video']} [{test_info['identity']}]"
                    if test_info["identity"] is not None
                    else test_info["video"],
                    accuracy=accuracy,
                    precision_behavior=pr[0][1],
                    precision_not_behavior=pr[0][0],
                    recall_behavior=pr[1][1],
                    recall_not_behavior=pr[1][0],
                    f1_behavior=pr[2][1],
                    support_behavior=int(pr[3][1]),
                    support_not_behavior=int(pr[3][0]),
                    confusion_matrix=confusion,
                    top_features=top_features,
                )
            )
            emit_progress()
    return cv_results
