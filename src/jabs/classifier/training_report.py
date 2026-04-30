"""Training report generation for classifier cross-validation results."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tabulate import tabulate

from jabs.core.enums import CrossValidationGroupingStrategy


@dataclass
class CrossValidationResult:
    """Results from a single cross-validation iteration.

    Attributes:
        iteration: The iteration number (1-indexed)
        test_label: Label of the test grouping (e.g., video filename and possibly identity index)
        accuracy: Classification accuracy (0.0 to 1.0)
        precision_behavior: Precision for behavior class (binary mode)
        precision_not_behavior: Precision for not-behavior class (binary mode)
        recall_behavior: Recall for behavior class (binary mode)
        recall_not_behavior: Recall for not-behavior class (binary mode)
        f1_behavior: F1 score for behavior class (binary mode)
        support_behavior: Number of behavior frames in test set (binary mode)
        support_not_behavior: Number of not-behavior frames in test set (binary mode)
        class_names: Ordered class names for multiclass mode
        class_support: Per-class support values for multiclass mode
        precision_macro: Macro precision for multiclass mode
        recall_macro: Macro recall for multiclass mode
        f1_macro: Macro F1 for multiclass mode
        precision_micro: Micro precision for multiclass mode
        recall_micro: Micro recall for multiclass mode
        f1_micro: Micro F1 for multiclass mode
        per_class_metrics: Per-class metric records for multiclass mode
        confusion_matrix: 2x2 confusion matrix
        top_features: List of (feature_name, importance) tuples for this iteration
    """

    iteration: int
    test_label: str
    accuracy: float
    confusion_matrix: np.ndarray
    top_features: list[tuple[str, float]] = field(default_factory=list)
    precision_behavior: float | None = None
    precision_not_behavior: float | None = None
    recall_behavior: float | None = None
    recall_not_behavior: float | None = None
    f1_behavior: float | None = None
    support_behavior: int | None = None
    support_not_behavior: int | None = None
    class_names: list[str] | None = None
    class_support: list[int] | None = None
    precision_macro: float | None = None
    recall_macro: float | None = None
    f1_macro: float | None = None
    precision_micro: float | None = None
    recall_micro: float | None = None
    f1_micro: float | None = None
    per_class_metrics: list[dict[str, float | int | str]] | None = None


@dataclass
class TrainingReportData:
    """Complete training information for generating a report.

    Attributes:
        behavior_name: Name of the behavior being trained
        classifier_type: Type/name of the classifier (e.g., "Random Forest")
        window_size: Window size used for feature extraction
        balance_training_labels: Whether training labels were balanced
        symmetric_behavior: Whether the behavior is symmetric
        distance_unit: Unit used for distance features ("cm" or "pixel")
        cv_results: List of CrossValidationResult objects, one per iteration
        final_top_features: Top features from final model (trained on all data)
        frames_behavior: Total number of frames labeled as behavior (binary mode)
        frames_not_behavior: Total number of frames labeled as not behavior (binary mode)
        bouts_behavior: Total number of behavior bouts labeled (binary mode)
        bouts_not_behavior: Total number of not-behavior bouts labeled (binary mode)
        class_frame_counts: Optional per-class frame counts (multiclass mode)
        class_bout_counts: Optional per-class bout counts (multiclass mode)
        training_time_ms: Total training time in milliseconds
        timestamp: Datetime when training was completed
        cv_grouping_strategy: Strategy used for cross-validation grouping
    """

    behavior_name: str
    classifier_type: str
    window_size: int
    balance_training_labels: bool
    symmetric_behavior: bool
    distance_unit: str
    cv_results: list[CrossValidationResult]
    final_top_features: list[tuple[str, float]]
    training_time_ms: int
    timestamp: datetime
    cv_grouping_strategy: CrossValidationGroupingStrategy
    frames_behavior: int = 0
    frames_not_behavior: int = 0
    bouts_behavior: int = 0
    bouts_not_behavior: int = 0
    class_frame_counts: dict[str, int] | None = None
    class_bout_counts: dict[str, int] | None = None


def _escape_markdown(text: str) -> str:
    """Escape markdown special characters in text.

    Args:
        text: Text that may contain markdown special characters

    Returns:
        Text with markdown special characters escaped
    """
    # Escape common markdown characters that might appear in filenames
    # Most important: _ (underscore) which creates italics
    # Also escape: * (asterisk), [ ] (brackets), ( ) (parentheses)
    chars_to_escape = ["_", "*", "[", "]", "(", ")", "`", "#"]
    for char in chars_to_escape:
        text = text.replace(char, f"\\{char}")
    return text


def generate_markdown_report(data: TrainingReportData) -> str:
    """Generate a markdown-formatted training report.

    Args:
        data: TrainingData object containing all training information

    Returns:
        Markdown-formatted string
    """
    lines = []

    lines.append(f"# Training Report: {data.behavior_name}")
    lines.append("")
    lines.append(f"**Date:** {data.timestamp.strftime('%B %d, %Y at %I:%M:%S %p')}")
    lines.append("")

    lines.append("## Training Summary")
    lines.append("")
    lines.append(f"- **Behavior:** {data.behavior_name}")
    lines.append(f"- **Classifier:** {data.classifier_type}")
    lines.append(f"- **Window Size:** {data.window_size}")
    lines.append(
        f"- **Balanced Training Labels:** {'Yes' if data.balance_training_labels else 'No'}"
    )
    lines.append(f"- **Symmetric Behavior:** {'Yes' if data.symmetric_behavior else 'No'}")
    lines.append(f"- **Distance Unit:** {data.distance_unit}")
    lines.append(f"- **Training Time:** {data.training_time_ms / 1000:.2f} seconds")
    lines.append("")

    lines.append("### Label Counts")
    lines.append("")
    if data.class_frame_counts:
        for name, count in data.class_frame_counts.items():
            lines.append(f"- **{_escape_markdown(name)} frames:** {count:,}")
        if data.class_bout_counts:
            for name, count in data.class_bout_counts.items():
                lines.append(f"- **{_escape_markdown(name)} bouts:** {count:,}")
    else:
        lines.append(f"- **Behavior frames:** {data.frames_behavior:,}")
        lines.append(f"- **Not-behavior frames:** {data.frames_not_behavior:,}")
        lines.append(f"- **Behavior bouts:** {data.bouts_behavior:,}")
        lines.append(f"- **Not-behavior bouts:** {data.bouts_not_behavior:,}")
    lines.append("")

    # Cross-validation results
    if data.cv_results:
        lines.append("## Cross-Validation Results")
        lines.append("")

        # Summary statistics
        accuracies = [r.accuracy for r in data.cv_results]
        is_multiclass_cv = data.cv_results[0].precision_macro is not None

        lines.append("### Performance Summary")
        lines.append("")
        lines.append(
            f"- **Mean Accuracy:** {np.mean(accuracies):.4f} (± {np.std(accuracies):.4f})"
        )
        if is_multiclass_cv:
            f1_macro = [r.f1_macro for r in data.cv_results if r.f1_macro is not None]
            f1_micro = [r.f1_micro for r in data.cv_results if r.f1_micro is not None]
            if f1_macro:
                lines.append(
                    f"- **Mean F1 Score (Macro):** {np.mean(f1_macro):.4f} (± {np.std(f1_macro):.4f})"
                )
            if f1_micro:
                lines.append(
                    f"- **Mean F1 Score (Micro):** {np.mean(f1_micro):.4f} (± {np.std(f1_micro):.4f})"
                )
        else:
            f1_behavior = [r.f1_behavior for r in data.cv_results if r.f1_behavior is not None]
            if f1_behavior:
                lines.append(
                    f"- **Mean F1 Score (Behavior):** {np.mean(f1_behavior):.4f} (± {np.std(f1_behavior):.4f})"
                )
        lines.append("")

        # Detailed results table
        lines.append("### Iteration Details")
        lines.append(f"CV Grouping Strategy: {data.cv_grouping_strategy.value}")
        lines.append("")

        table_data = []
        if is_multiclass_cv:
            for result in data.cv_results:
                escaped_video = _escape_markdown(result.test_label)
                table_data.append(
                    [
                        result.iteration,
                        f"{result.accuracy:.4f}",
                        f"{(result.precision_macro if result.precision_macro is not None else 0.0):.4f}",
                        f"{(result.recall_macro if result.recall_macro is not None else 0.0):.4f}",
                        f"{(result.f1_macro if result.f1_macro is not None else 0.0):.4f}",
                        f"{(result.f1_micro if result.f1_micro is not None else 0.0):.4f}",
                        f"{escaped_video}",
                    ]
                )

            headers = [
                "Iter",
                "Accuracy",
                "Precision (Macro)",
                "Recall (Macro)",
                "F1 Score (Macro)",
                "F1 Score (Micro)",
                "Test Group",
            ]
        else:
            for result in data.cv_results:
                escaped_video = _escape_markdown(result.test_label)
                table_data.append(
                    [
                        result.iteration,
                        f"{result.accuracy:.4f}",
                        f"{(result.precision_not_behavior if result.precision_not_behavior is not None else 0.0):.4f}",
                        f"{(result.precision_behavior if result.precision_behavior is not None else 0.0):.4f}",
                        f"{(result.recall_not_behavior if result.recall_not_behavior is not None else 0.0):.4f}",
                        f"{(result.recall_behavior if result.recall_behavior is not None else 0.0):.4f}",
                        f"{(result.f1_behavior if result.f1_behavior is not None else 0.0):.4f}",
                        f"{escaped_video}",
                    ]
                )

            headers = [
                "Iter",
                "Accuracy",
                "Precision (Not Behavior)",
                "Precision (Behavior)",
                "Recall (Not Behavior)",
                "Recall (Behavior)",
                "F1 Score",
                "Test Group",
            ]

        table_markdown = tabulate(table_data, headers=headers, tablefmt="github")
        lines.append(table_markdown)
        lines.append("")
    else:
        # No cross-validation was performed
        lines.append("## Cross-Validation")
        lines.append("")
        lines.append("*No cross-validation was performed for this training.*")
        lines.append("")

    # Final model feature importance
    lines.append("## Feature Importance")
    lines.append("")
    lines.append("Top 20 features from final model (trained on all labeled data):")
    lines.append("")

    feature_table = []
    for rank, (feature_name, importance) in enumerate(data.final_top_features, start=1):
        feature_table.append([rank, _escape_markdown(feature_name), f"{importance:.2f}"])

    feature_markdown = tabulate(
        feature_table, headers=["Rank", "Feature Name", "Importance"], tablefmt="github"
    )
    lines.append(feature_markdown)
    lines.append("")

    return "\n".join(lines)


def _to_python_type(val):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, np.ndarray):
        return [_to_python_type(x) for x in val.tolist()]
    if isinstance(val, list):
        return [_to_python_type(x) for x in val]
    if isinstance(val, dict):
        return {k: _to_python_type(v) for k, v in val.items()}
    return val


def generate_json_report(data: TrainingReportData) -> dict:
    """Generate a JSON-serializable training report.

    Ensures all numpy types are converted to native Python types.

    Args:
        data: TrainingReportData object containing all training information

    Returns:
        Dictionary suitable for JSON serialization
    """
    # Convert timestamp to UTC and use ISO format with 'Z' suffix
    if data.timestamp.tzinfo is None:
        # Assume naive datetime is local time; convert to UTC
        timestamp_utc = data.timestamp.astimezone(timezone.utc)
    else:
        timestamp_utc = data.timestamp.astimezone(timezone.utc)
    timestamp_str = timestamp_utc.replace(tzinfo=None).isoformat() + "Z"

    report = {
        "behavior_name": data.behavior_name,
        "classifier_type": data.classifier_type,
        "window_size": int(data.window_size),
        "balance_training_labels": bool(data.balance_training_labels),
        "symmetric_behavior": bool(data.symmetric_behavior),
        "distance_unit": data.distance_unit,
        "training_time_ms": int(data.training_time_ms),
        "timestamp": timestamp_str,
        "cv_grouping_strategy": data.cv_grouping_strategy.value,
        "frames_behavior": int(data.frames_behavior),
        "frames_not_behavior": int(data.frames_not_behavior),
        "bouts_behavior": int(data.bouts_behavior),
        "bouts_not_behavior": int(data.bouts_not_behavior),
        "class_frame_counts": (
            {str(name): int(count) for name, count in data.class_frame_counts.items()}
            if data.class_frame_counts is not None
            else None
        ),
        "class_bout_counts": (
            {str(name): int(count) for name, count in data.class_bout_counts.items()}
            if data.class_bout_counts is not None
            else None
        ),
        "cv_results": [],
        "final_top_features": [
            {"feature_name": str(name), "importance": float(importance)}
            for name, importance in data.final_top_features
        ],
    }

    for result in data.cv_results:
        result_dict = {
            "iteration": int(result.iteration),
            "test_label": str(result.test_label),
            "accuracy": float(result.accuracy),
            "confusion_matrix": _to_python_type(result.confusion_matrix),
            "top_features": [
                {"feature_name": str(name), "importance": float(importance)}
                for name, importance in result.top_features
            ],
        }
        if result.precision_behavior is not None:
            result_dict["precision_behavior"] = float(result.precision_behavior)
        if result.precision_not_behavior is not None:
            result_dict["precision_not_behavior"] = float(result.precision_not_behavior)
        if result.recall_behavior is not None:
            result_dict["recall_behavior"] = float(result.recall_behavior)
        if result.recall_not_behavior is not None:
            result_dict["recall_not_behavior"] = float(result.recall_not_behavior)
        if result.f1_behavior is not None:
            result_dict["f1_behavior"] = float(result.f1_behavior)
        if result.support_behavior is not None:
            result_dict["support_behavior"] = int(result.support_behavior)
        if result.support_not_behavior is not None:
            result_dict["support_not_behavior"] = int(result.support_not_behavior)
        if result.class_names is not None:
            result_dict["class_names"] = [str(name) for name in result.class_names]
        if result.class_support is not None:
            result_dict["class_support"] = [int(v) for v in result.class_support]
        if result.precision_macro is not None:
            result_dict["precision_macro"] = float(result.precision_macro)
        if result.recall_macro is not None:
            result_dict["recall_macro"] = float(result.recall_macro)
        if result.f1_macro is not None:
            result_dict["f1_macro"] = float(result.f1_macro)
        if result.precision_micro is not None:
            result_dict["precision_micro"] = float(result.precision_micro)
        if result.recall_micro is not None:
            result_dict["recall_micro"] = float(result.recall_micro)
        if result.f1_micro is not None:
            result_dict["f1_micro"] = float(result.f1_micro)
        if result.per_class_metrics is not None:
            result_dict["per_class_metrics"] = _to_python_type(result.per_class_metrics)
        report["cv_results"].append(result_dict)

    return report


def save_training_report(data: TrainingReportData, output_path: Path) -> None:
    """Generate and save a training report.

    The report format is determined by the file extension of output_path.
    Currently, markdown (.md) and JSON (.json) are supported.

    Args:
        data: TrainingData object containing all training information
        output_path: Path where the report file should be saved
    """
    if output_path.suffix.lower() == ".md":
        markdown_content = generate_markdown_report(data)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
    elif output_path.suffix.lower() == ".json":
        json_content = generate_json_report(data)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, indent=4)
    else:
        raise ValueError(
            "Only markdown (.md) or JSON (.json) output formats are currently supported."
        )
