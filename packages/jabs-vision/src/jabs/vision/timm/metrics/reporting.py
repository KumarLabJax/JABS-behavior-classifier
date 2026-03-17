"""Formatting helpers for presenting metrics results to developers."""

from __future__ import annotations

from collections.abc import Sequence

from .types import DetectionAPResult, PCKResult, PoseAPResult

__all__ = ["format_results"]


def format_results(
    pose: PoseAPResult | None = None,
    detection: DetectionAPResult | None = None,
    pck_results: Sequence[PCKResult] | None = None,
    keypoint_names: Sequence[str] | None = None,
) -> str:
    """Format evaluation results as a human-readable multi-line report."""
    lines: list[str] = []

    if pose is not None:
        lines.append("=" * 60)
        lines.append("Pose Evaluation (OKS-based)")
        lines.append("=" * 60)
        lines.append(f"  AP   (mean) : {pose.ap:.4f}")
        lines.append(f"  AP50        : {pose.ap_50:.4f}")
        lines.append(f"  AP75        : {pose.ap_75:.4f}")
        lines.append(f"  AR   (mean) : {pose.ar:.4f}")
        lines.append(f"  AR50        : {pose.ar_50:.4f}")
        lines.append(f"  AR75        : {pose.ar_75:.4f}")

        if pose.per_threshold:
            lines.append("")
            lines.append("  Per-threshold breakdown:")
            lines.append(f"  {'Threshold':>10}  {'AP':>8}  {'AR':>8}")
            lines.append(f"  {'-' * 10}  {'-' * 8}  {'-' * 8}")
            for result in pose.per_threshold:
                lines.append(f"  {result.threshold:>10.2f}  {result.ap:>8.4f}  {result.ar:>8.4f}")
        lines.append("")

    if detection is not None:
        lines.append("=" * 60)
        lines.append("Detection Evaluation (IoU-based)")
        lines.append("=" * 60)
        lines.append(f"  AP   (mean) : {detection.ap:.4f}")
        lines.append(f"  AP50        : {detection.ap_50:.4f}")
        lines.append(f"  AP75        : {detection.ap_75:.4f}")
        lines.append(f"  AR   (mean) : {detection.ar:.4f}")
        lines.append(f"  AR50        : {detection.ar_50:.4f}")
        lines.append(f"  AR75        : {detection.ar_75:.4f}")

        if detection.per_threshold:
            lines.append("")
            lines.append("  Per-threshold breakdown:")
            lines.append(f"  {'Threshold':>10}  {'AP':>8}  {'AR':>8}")
            lines.append(f"  {'-' * 10}  {'-' * 8}  {'-' * 8}")
            for result in detection.per_threshold:
                lines.append(f"  {result.threshold:>10.2f}  {result.ap:>8.4f}  {result.ar:>8.4f}")
        lines.append("")

    if pck_results:
        lines.append("=" * 60)
        lines.append("PCK Evaluation")
        lines.append("=" * 60)

        for pck in pck_results:
            lines.append(f"  PCK @ {pck.threshold:.2f} : {pck.pck:.4f}")

            if pck.per_keypoint:
                lines.append("")
                lines.append(f"  {'Keypoint':>20}  {'Correct':>8}  {'Total':>8}  {'PCK':>8}")
                lines.append(f"  {'-' * 20}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

                for keypoint_index in sorted(pck.per_keypoint):
                    detail = pck.per_keypoint[keypoint_index]
                    name = keypoint_name(keypoint_index, keypoint_names)
                    lines.append(
                        f"  {name:>20}  {detail.correct:>8}  {detail.total:>8}  {detail.pck:>8.4f}"
                    )

            if pck.excluded_indices:
                excluded = ", ".join(
                    keypoint_name(index, keypoint_names) for index in sorted(pck.excluded_indices)
                )
                lines.append(f"  Excluded: {excluded}")
            lines.append("")

    return "\n".join(lines)


def keypoint_name(index: int, names: Sequence[str] | None) -> str:
    """Return a human-readable name for a keypoint index."""
    if names is not None and index < len(names):
        return names[index]
    return f"keypoint_{index}"
