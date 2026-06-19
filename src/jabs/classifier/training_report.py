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
    """Common fields shared by binary and multi-class CV iteration results.

    Attributes:
        iteration: The iteration number (1-indexed).
        test_label: Label of the test grouping (e.g., video filename and
            possibly identity index).
        accuracy: Classification accuracy (0.0 to 1.0).
        confusion_matrix: Confusion matrix for this iteration. Shape ``(2, 2)``
            for binary mode, ``(n_classes, n_classes)`` for multi-class.
        top_features: List of ``(feature_name, importance)`` tuples for this
            iteration.
    """

    iteration: int
    test_label: str
    accuracy: float
    confusion_matrix: np.ndarray
    top_features: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class BinaryCVResult(CrossValidationResult):
    """Binary cross-validation iteration result.

    Attributes:
        precision_behavior: Precision for the behavior class.
        precision_not_behavior: Precision for the not-behavior class.
        recall_behavior: Recall for the behavior class.
        recall_not_behavior: Recall for the not-behavior class.
        f1_behavior: F1 score for the behavior class.
        support_behavior: Number of behavior frames in the test set.
        support_not_behavior: Number of not-behavior frames in the test set.
    """

    precision_behavior: float = 0.0
    precision_not_behavior: float = 0.0
    recall_behavior: float = 0.0
    recall_not_behavior: float = 0.0
    f1_behavior: float = 0.0
    support_behavior: int = 0
    support_not_behavior: int = 0


@dataclass
class MultiClassCVResult(CrossValidationResult):
    """Multi-class cross-validation iteration result.

    Attributes:
        class_names: Ordered class names (e.g., ``["None", "Walk", "Run"]``).
        class_support: Per-class support values in the test set.
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        f1_macro: Macro-averaged F1 score.
        precision_micro: Micro-averaged precision.
        recall_micro: Micro-averaged recall.
        f1_micro: Micro-averaged F1 score.
        per_class_metrics: Per-class metric records.
    """

    class_names: list[str] = field(default_factory=list)
    class_support: list[int] = field(default_factory=list)
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    precision_micro: float = 0.0
    recall_micro: float = 0.0
    f1_micro: float = 0.0
    per_class_metrics: list[dict[str, float | int | str]] = field(default_factory=list)


@dataclass
class TrainingReportData:
    """Complete training information for generating a report.

    Attributes:
        behavior_name: Name of the behavior being trained.
        classifier_type: Type/name of the classifier (e.g., "Random Forest").
        window_size: Window size used for feature extraction.
        balance_training_labels: Whether training labels were balanced.
        symmetric_behavior: Whether the behavior is symmetric.
        distance_unit: Unit used for distance features ("cm" or "pixel").
        cv_results: List of cross-validation iteration results.
        final_top_features: Top features from final model (trained on all data).
        frames_behavior: Total behavior frames (binary mode).
        frames_not_behavior: Total not-behavior frames (binary mode).
        bouts_behavior: Total behavior bouts (binary mode).
        bouts_not_behavior: Total not-behavior bouts (binary mode).
        class_frame_counts: Per-class frame counts (multi-class mode).
        class_bout_counts: Per-class bout counts (multi-class mode).
        training_time_ms: Total training time in milliseconds.
        timestamp: Datetime when training was completed.
        cv_grouping_strategy: Strategy used for cross-validation grouping.
        cv_grouping_regex: Filename-pattern regex used for grouping. Only set when
            the grouping strategy is "Filename Pattern".
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
    cv_grouping_regex: str | None = None


def _escape_markdown(text: str) -> str:
    """Escape markdown special characters that might appear in identifiers."""
    chars_to_escape = ["_", "*", "[", "]", "(", ")", "`", "#"]
    for char in chars_to_escape:
        text = text.replace(char, f"\\{char}")
    return text


def _is_multiclass_cv(cv_results: list[CrossValidationResult]) -> bool:
    """Return True if the CV result list belongs to multi-class mode."""
    return bool(cv_results) and isinstance(cv_results[0], MultiClassCVResult)


def _format_label_counts(data: TrainingReportData) -> list[str]:
    """Return markdown lines for the label-counts section."""
    lines: list[str] = []
    if data.class_frame_counts is not None:
        for name, count in data.class_frame_counts.items():
            lines.append(f"- **{_escape_markdown(name)} frames:** {count:,}")
        if data.class_bout_counts is not None:
            for name, count in data.class_bout_counts.items():
                lines.append(f"- **{_escape_markdown(name)} bouts:** {count:,}")
    else:
        lines.append(f"- **Behavior frames:** {data.frames_behavior:,}")
        lines.append(f"- **Not-behavior frames:** {data.frames_not_behavior:,}")
        lines.append(f"- **Behavior bouts:** {data.bouts_behavior:,}")
        lines.append(f"- **Not-behavior bouts:** {data.bouts_not_behavior:,}")
    return lines


def _format_performance_summary(cv_results: list[CrossValidationResult]) -> list[str]:
    """Return markdown lines summarizing accuracy and F1 across iterations."""
    accuracies = [r.accuracy for r in cv_results]
    lines = [f"- **Mean Accuracy:** {np.mean(accuracies):.4f} (± {np.std(accuracies):.4f})"]
    if _is_multiclass_cv(cv_results):
        f1_macro = [r.f1_macro for r in cv_results if isinstance(r, MultiClassCVResult)]
        f1_micro = [r.f1_micro for r in cv_results if isinstance(r, MultiClassCVResult)]
        if f1_macro:
            lines.append(
                f"- **Mean F1 Score (Macro):** {np.mean(f1_macro):.4f} (± {np.std(f1_macro):.4f})"
            )
        if f1_micro:
            lines.append(
                f"- **Mean F1 Score (Micro):** {np.mean(f1_micro):.4f} (± {np.std(f1_micro):.4f})"
            )
    else:
        f1_behavior = [r.f1_behavior for r in cv_results if isinstance(r, BinaryCVResult)]
        if f1_behavior:
            lines.append(
                f"- **Mean F1 Score (Behavior):** {np.mean(f1_behavior):.4f} "
                f"(± {np.std(f1_behavior):.4f})"
            )
    return lines


def _binary_iteration_row(result: BinaryCVResult) -> list[str | int]:
    """Return a single iteration row for the binary CV table."""
    return [
        result.iteration,
        f"{result.accuracy:.4f}",
        f"{result.precision_not_behavior:.4f}",
        f"{result.precision_behavior:.4f}",
        f"{result.recall_not_behavior:.4f}",
        f"{result.recall_behavior:.4f}",
        f"{result.f1_behavior:.4f}",
        _escape_markdown(result.test_label),
    ]


def _multiclass_iteration_row(result: MultiClassCVResult) -> list[str | int]:
    """Return a single iteration row for the multi-class CV table."""
    return [
        result.iteration,
        f"{result.accuracy:.4f}",
        f"{result.precision_macro:.4f}",
        f"{result.recall_macro:.4f}",
        f"{result.f1_macro:.4f}",
        f"{result.f1_micro:.4f}",
        _escape_markdown(result.test_label),
    ]


_BINARY_HEADERS = [
    "Iter",
    "Accuracy",
    "Precision (Not Behavior)",
    "Precision (Behavior)",
    "Recall (Not Behavior)",
    "Recall (Behavior)",
    "F1 Score",
    "Test Group",
]

_MULTICLASS_HEADERS = [
    "Iter",
    "Accuracy",
    "Precision (Macro)",
    "Recall (Macro)",
    "F1 Score (Macro)",
    "F1 Score (Micro)",
    "Test Group",
]


def _format_iteration_table(cv_results: list[CrossValidationResult]) -> str:
    """Return the markdown iteration-details table."""
    if _is_multiclass_cv(cv_results):
        rows = [
            _multiclass_iteration_row(r) for r in cv_results if isinstance(r, MultiClassCVResult)
        ]
        headers = _MULTICLASS_HEADERS
    else:
        rows = [_binary_iteration_row(r) for r in cv_results if isinstance(r, BinaryCVResult)]
        headers = _BINARY_HEADERS
    return tabulate(rows, headers=headers, tablefmt="github")


def generate_markdown_report(data: TrainingReportData) -> str:
    """Generate a markdown-formatted training report.

    Args:
        data: ``TrainingReportData`` object containing all training information.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

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
    lines.extend(_format_label_counts(data))
    lines.append("")

    if data.cv_results:
        lines.append("## Cross-Validation Results")
        lines.append("")
        lines.append("### Performance Summary")
        lines.append("")
        lines.extend(_format_performance_summary(data.cv_results))
        lines.append("")

        lines.append("### Iteration Details")
        lines.append(f"CV Grouping Strategy: {data.cv_grouping_strategy.value}")
        if data.cv_grouping_regex:
            lines.append(f"CV Grouping Pattern: `{data.cv_grouping_regex}`")
        lines.append("")
        lines.append(_format_iteration_table(data.cv_results))
        lines.append("")
    else:
        lines.append("## Cross-Validation")
        lines.append("")
        lines.append("*No cross-validation was performed for this training.*")
        lines.append("")

    lines.append("## Feature Importance")
    lines.append("")
    lines.append("Top 20 features from final model (trained on all labeled data):")
    lines.append("")

    feature_table = [
        [rank, _escape_markdown(feature_name), f"{importance:.2f}"]
        for rank, (feature_name, importance) in enumerate(data.final_top_features, start=1)
    ]
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


def _common_cv_dict(result: CrossValidationResult) -> dict:
    """Return the dict fields common to all CV result types."""
    return {
        "iteration": int(result.iteration),
        "test_label": str(result.test_label),
        "accuracy": float(result.accuracy),
        "confusion_matrix": _to_python_type(result.confusion_matrix),
        "top_features": [
            {"feature_name": str(name), "importance": float(importance)}
            for name, importance in result.top_features
        ],
    }


def _binary_cv_to_dict(result: BinaryCVResult) -> dict:
    """Convert a binary CV result to its JSON-serializable dict."""
    payload = _common_cv_dict(result)
    payload.update(
        {
            "precision_behavior": float(result.precision_behavior),
            "precision_not_behavior": float(result.precision_not_behavior),
            "recall_behavior": float(result.recall_behavior),
            "recall_not_behavior": float(result.recall_not_behavior),
            "f1_behavior": float(result.f1_behavior),
            "support_behavior": int(result.support_behavior),
            "support_not_behavior": int(result.support_not_behavior),
        }
    )
    return payload


def _multiclass_cv_to_dict(result: MultiClassCVResult) -> dict:
    """Convert a multi-class CV result to its JSON-serializable dict."""
    payload = _common_cv_dict(result)
    payload.update(
        {
            "class_names": [str(name) for name in result.class_names],
            "class_support": [int(v) for v in result.class_support],
            "precision_macro": float(result.precision_macro),
            "recall_macro": float(result.recall_macro),
            "f1_macro": float(result.f1_macro),
            "precision_micro": float(result.precision_micro),
            "recall_micro": float(result.recall_micro),
            "f1_micro": float(result.f1_micro),
            "per_class_metrics": _to_python_type(result.per_class_metrics),
        }
    )
    return payload


def _cv_result_to_dict(result: CrossValidationResult) -> dict:
    """Dispatch a CV result to its type-specific dict converter."""
    if isinstance(result, MultiClassCVResult):
        return _multiclass_cv_to_dict(result)
    if isinstance(result, BinaryCVResult):
        return _binary_cv_to_dict(result)
    return _common_cv_dict(result)


def generate_json_report(data: TrainingReportData) -> dict:
    """Generate a JSON-serializable training report.

    Ensures all numpy types are converted to native Python types.

    Args:
        data: ``TrainingReportData`` object containing all training information.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    timestamp_utc = data.timestamp.astimezone(timezone.utc)
    timestamp_str = timestamp_utc.replace(tzinfo=None).isoformat() + "Z"

    return {
        "behavior_name": data.behavior_name,
        "classifier_type": data.classifier_type,
        "window_size": int(data.window_size),
        "balance_training_labels": bool(data.balance_training_labels),
        "symmetric_behavior": bool(data.symmetric_behavior),
        "distance_unit": data.distance_unit,
        "training_time_ms": int(data.training_time_ms),
        "timestamp": timestamp_str,
        "cv_grouping_strategy": data.cv_grouping_strategy.value,
        "cv_grouping_regex": data.cv_grouping_regex,
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
        "cv_results": [_cv_result_to_dict(r) for r in data.cv_results],
        "final_top_features": [
            {"feature_name": str(name), "importance": float(importance)}
            for name, importance in data.final_top_features
        ],
    }


def save_training_report(data: TrainingReportData, output_path: Path) -> None:
    """Generate and save a training report.

    The report format is determined by the file extension of ``output_path``.
    Currently, markdown (``.md``) and JSON (``.json``) are supported.

    Args:
        data: ``TrainingReportData`` object containing all training information.
        output_path: Path where the report file should be saved.

    Raises:
        ValueError: If the output path has an unsupported extension.
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
