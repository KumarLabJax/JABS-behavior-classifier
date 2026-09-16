"""Rendering for classifier evaluation results.

Turns an :class:`~jabs.scripts.cli.evaluate_results.EvaluationResult` into the
four output forms the evaluate command offers: Rich tables on the console, a
JSON metrics summary, a per-bout CSV, and a markdown report.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from jabs.behavior.evaluation import BoutMetrics, FrameMetrics

from .evaluate_results import (
    BoutRecord,
    EvaluationResult,
    IdentityResult,
    aggregate_bout_metrics,
    aggregate_frame_metrics,
    format_rate,
    stage_label,
)


def _frame_row(label: str, metrics: FrameMetrics) -> list[str]:
    """Build a frame-metrics table row.

    Args:
        label: Row label (a stage, video, or identity).
        metrics: Frame counts to render.

    Returns:
        Cell values in the order the frame table declares its columns.
    """
    return [
        label,
        f"{metrics.evaluated_frames:,}",
        format_rate(metrics.accuracy),
        format_rate(metrics.precision_behavior),
        format_rate(metrics.recall_behavior),
        format_rate(metrics.f1_behavior),
        f"{metrics.true_positive:,}",
        f"{metrics.false_positive:,}",
        f"{metrics.false_negative:,}",
    ]


def _bout_row(label: str, metrics: BoutMetrics) -> list[str]:
    """Build a bout-metrics table row.

    Args:
        label: Row label (a stage, video, or identity).
        metrics: Bout counts to render.

    Returns:
        Cell values in the order the bout table declares its columns.
    """
    return [
        label,
        f"{metrics.evaluable_truth_bouts:,}",
        f"{metrics.detected_truth_bouts:,}",
        format_rate(metrics.detection_rate),
        f"{metrics.evaluable_predicted_bouts:,}",
        format_rate(metrics.precision),
        format_rate(metrics.f1),
        f"{metrics.fragmented_truth_bouts:,}",
        f"{metrics.merged_predicted_bouts:,}",
    ]


def _frame_table(title: str) -> Table:
    """Create an empty frame-metrics table with its columns declared.

    Args:
        title: Table title.

    Returns:
        A Rich table ready for ``_frame_row`` values.
    """
    table = Table(title=title, title_justify="left")
    table.add_column("", justify="left")
    table.add_column("Frames", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    return table


def _bout_table(title: str) -> Table:
    """Create an empty bout-metrics table with its columns declared.

    Args:
        title: Table title.

    Returns:
        A Rich table ready for ``_bout_row`` values.
    """
    table = Table(title=title, title_justify="left")
    table.add_column("", justify="left")
    table.add_column("GT\nbouts", justify="right")
    table.add_column("Detected", justify="right")
    table.add_column("Detection\nrate", justify="right")
    table.add_column("Pred\nbouts", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Frag-\nmented", justify="right")
    table.add_column("Merged", justify="right")
    return table


def print_console_report(result: EvaluationResult, console: Console, per_video: bool) -> None:
    """Print the evaluation tables to the console.

    Args:
        result: Evaluation to render.
        console: Rich console to print to.
        per_video: Also print a per-video breakdown under each summary table.
    """
    console.print()
    console.print(f"[bold]Behavior:[/bold] {result.behavior}")
    console.print(f"[bold]Classifier:[/bold] {result.classifier_path}")
    console.print(
        f"[bold]Type:[/bold] {result.classifier_type}   "
        f"[bold]Window size:[/bold] {result.window_size}"
    )
    console.print(f"[bold]Project:[/bold] {result.project_dir}")
    if result.postprocess_stages:
        console.print(f"[bold]Postprocessing:[/bold] {', '.join(result.postprocess_stages)}")
    console.print()

    # --- frame-level ---------------------------------------------------------
    table = _frame_table("Frame-level agreement")
    for stage in result.stages:
        table.add_row(
            *_frame_row(stage_label(stage), aggregate_frame_metrics(result.for_stage(stage)))
        )
    console.print(table)

    totals = aggregate_frame_metrics(result.for_stage(result.stages[0]))
    excluded = []
    if totals.unlabeled_frames:
        excluded.append(f"{totals.unlabeled_frames:,} unlabeled")
    if totals.unpredicted_frames:
        excluded.append(f"{totals.unpredicted_frames:,} labeled but unscored (no pose)")
    if excluded:
        console.print(f"[dim]Frames excluded from the comparison: {', '.join(excluded)}.[/dim]")
    console.print()

    if per_video:
        for stage in result.stages:
            table = _frame_table(f"Frame-level agreement by video - {stage_label(stage)}")
            for video in result.videos:
                table.add_row(
                    *_frame_row(video, aggregate_frame_metrics(result.for_video(stage, video)))
                )
            console.print(table)
            console.print()

    # --- bout-level ----------------------------------------------------------
    for criterion in result.criteria:
        table = _bout_table(f"Bout-level agreement - {criterion}")
        for stage in result.stages:
            table.add_row(
                *_bout_row(
                    stage_label(stage),
                    aggregate_bout_metrics(result.for_stage(stage), criterion),
                )
            )
        console.print(table)
        console.print()

        if per_video:
            for stage in result.stages:
                table = _bout_table(f"Bouts by video - {criterion} - {stage_label(stage)}")
                for video in result.videos:
                    table.add_row(
                        *_bout_row(
                            video,
                            aggregate_bout_metrics(result.for_video(stage, video), criterion),
                        )
                    )
                console.print(table)
                console.print()

    bouts = aggregate_bout_metrics(result.for_stage(result.stages[0]), result.criteria[0])
    notes = []
    if bouts.unevaluable_truth_bouts:
        notes.append(
            f"{bouts.unevaluable_truth_bouts:,} ground-truth bout(s) lay entirely in unscored "
            "frames and are excluded from the detection rate"
        )
    if bouts.unevaluable_predicted_bouts:
        notes.append(
            f"{bouts.unevaluable_predicted_bouts:,} predicted bout(s) lay entirely in unlabeled "
            "frames and are excluded from precision"
        )
    for note in notes:
        console.print(f"[dim]Note: {note}.[/dim]")

    for video, reason in result.prediction_write_errors:
        console.print(f"[red]Could not write predictions for {video}: {reason}[/red]")
    if result.prediction_files:
        console.print(
            f"[dim]Wrote {len(result.prediction_files)} prediction file(s) to "
            f"{result.prediction_files[0].parent}.[/dim]"
        )

    for video, reason in result.skipped_videos:
        console.print(f"[yellow]Skipped {video}: {reason}[/yellow]")
    if result.unlabeled_identities:
        console.print(
            f"[dim]{len(result.unlabeled_identities)} identity/identities had no "
            f"'{result.behavior}' ground truth and contributed nothing.[/dim]"
        )


def _metrics_block(results: Iterable[IdentityResult], criteria: Iterable[str]) -> dict:
    """Build the JSON block for one aggregation scope.

    Args:
        results: Results to aggregate.
        criteria: Match-criterion labels to include.

    Returns:
        Mapping with ``frames`` and ``bouts`` sub-blocks.
    """
    results = list(results)
    return {
        "frames": aggregate_frame_metrics(results).as_dict(),
        "bouts": {c: aggregate_bout_metrics(results, c).as_dict() for c in criteria},
    }


def build_summary(result: EvaluationResult, timestamp: datetime) -> dict:
    """Build the JSON-serializable summary of an evaluation.

    Args:
        result: Evaluation to summarize.
        timestamp: Time the evaluation completed.

    Returns:
        Nested mapping: run metadata, then per stage an overall block, a
        per-video block, and a per-identity block within each video.
    """
    summary: dict = {
        "generated": timestamp.isoformat(timespec="seconds"),
        "behavior": result.behavior,
        "project_dir": str(result.project_dir),
        "classifier": {
            "path": str(result.classifier_path),
            "type": result.classifier_type,
            "window_size": result.window_size,
        },
        "postprocessing_stages": list(result.postprocess_stages),
        "criteria": list(result.criteria),
        "skipped_videos": [{"video": v, "reason": r} for v, r in result.skipped_videos],
        "unlabeled_identities": [
            {"video": v, "identity": i} for v, i in result.unlabeled_identities
        ],
        "prediction_files": [str(p) for p in result.prediction_files],
        "prediction_write_errors": [
            {"video": v, "reason": r} for v, r in result.prediction_write_errors
        ],
        "stages": {},
    }

    for stage in result.stages:
        stage_results = result.for_stage(stage)
        videos: dict[str, dict] = {}
        for video in result.videos:
            video_results = result.for_video(stage, video)
            block = _metrics_block(video_results, result.criteria)
            block["identities"] = {
                str(r.identity): _metrics_block([r], result.criteria) for r in video_results
            }
            videos[video] = block
        summary["stages"][stage] = {
            "overall": _metrics_block(stage_results, result.criteria),
            "videos": videos,
        }

    return summary


def write_json(result: EvaluationResult, path: Path, timestamp: datetime) -> None:
    """Write the JSON metrics summary.

    Args:
        result: Evaluation to serialize.
        path: Destination file.
        timestamp: Time the evaluation completed.

    Raises:
        OSError: If the file cannot be written.
    """
    path.write_text(
        json.dumps(build_summary(result, timestamp), indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(records: list[BoutRecord], path: Path) -> None:
    """Write the per-bout detail rows.

    Args:
        records: Bout rows to write. An empty list still writes the header, so
            the file is readable rather than zero-length.
        path: Destination file.

    Raises:
        OSError: If the file cannot be written.
    """
    columns = [f.name for f in fields(BoutRecord)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a markdown table.

    Args:
        headers: Column headings.
        rows: Row cell values, each the same length as ``headers``.

    Returns:
        Lines of the rendered table, with a trailing blank line.
    """
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    return lines


def render_markdown(result: EvaluationResult, timestamp: datetime) -> str:
    """Render the evaluation as a markdown report.

    Args:
        result: Evaluation to render.
        timestamp: Time the evaluation completed.

    Returns:
        The report as a single string.
    """
    frame_headers = [
        "",
        "Frames",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "TP",
        "FP",
        "FN",
    ]
    bout_headers = [
        "",
        "GT bouts",
        "Detected",
        "Detection rate",
        "Pred bouts",
        "Precision",
        "F1",
        "Fragmented",
        "Merged",
    ]

    lines = [
        f"# Classifier Evaluation: {result.behavior}",
        "",
        f"- **Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Project:** `{result.project_dir}`",
        f"- **Classifier:** `{result.classifier_path}`",
        f"- **Classifier type:** {result.classifier_type}",
        f"- **Window size:** {result.window_size}",
        f"- **Videos evaluated:** {len(result.videos)}",
    ]
    if result.postprocess_stages:
        lines.append(f"- **Postprocessing:** {', '.join(result.postprocess_stages)}")
    lines.append("")

    lines.append("## Frame-level agreement")
    lines.append("")
    lines.extend(
        _markdown_table(
            frame_headers,
            [
                _frame_row(stage_label(s), aggregate_frame_metrics(result.for_stage(s)))
                for s in result.stages
            ],
        )
    )

    totals = aggregate_frame_metrics(result.for_stage(result.stages[0]))
    lines.append(
        f"Excluded from the comparison: {totals.unlabeled_frames:,} unlabeled frame(s) and "
        f"{totals.unpredicted_frames:,} labeled frame(s) the classifier could not score."
    )
    lines.append("")

    lines.append("## Bout-level agreement")
    lines.append("")
    lines.append(
        "A ground-truth bout counts as detected when at least one predicted bout satisfies the "
        "criterion against it, so a true bout split into several predictions is still detected; "
        "the split is reported as *Fragmented*. *Merged* counts predicted bouts spanning two or "
        "more true bouts."
    )
    lines.append("")

    for criterion in result.criteria:
        lines.append(f"### {criterion}")
        lines.append("")
        lines.extend(
            _markdown_table(
                bout_headers,
                [
                    _bout_row(
                        stage_label(s), aggregate_bout_metrics(result.for_stage(s), criterion)
                    )
                    for s in result.stages
                ],
            )
        )

    lines.append("## Per-video breakdown")
    lines.append("")
    for stage in result.stages:
        lines.append(f"### {stage_label(stage)} - frames")
        lines.append("")
        lines.extend(
            _markdown_table(
                frame_headers,
                [
                    _frame_row(v, aggregate_frame_metrics(result.for_video(stage, v)))
                    for v in result.videos
                ],
            )
        )
        for criterion in result.criteria:
            lines.append(f"### {stage_label(stage)} - bouts ({criterion})")
            lines.append("")
            lines.extend(
                _markdown_table(
                    bout_headers,
                    [
                        _bout_row(v, aggregate_bout_metrics(result.for_video(stage, v), criterion))
                        for v in result.videos
                    ],
                )
            )

    if result.prediction_write_errors:
        lines.append("## Prediction files that could not be written")
        lines.append("")
        lines.extend(f"- `{video}`: {reason}" for video, reason in result.prediction_write_errors)
        lines.append("")

    if result.prediction_files:
        lines.append("## Saved predictions")
        lines.append("")
        lines.append(
            f"{len(result.prediction_files)} prediction file(s) written to "
            f"`{result.prediction_files[0].parent}`."
        )
        lines.append("")

    if result.skipped_videos:
        lines.append("## Skipped videos")
        lines.append("")
        lines.extend(f"- `{video}`: {reason}" for video, reason in result.skipped_videos)
        lines.append("")

    if result.unlabeled_identities:
        lines.append("## Identities without ground truth")
        lines.append("")
        lines.extend(
            f"- `{video}` identity {identity}" for video, identity in result.unlabeled_identities
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def write_markdown(result: EvaluationResult, path: Path, timestamp: datetime) -> None:
    """Write the markdown report.

    Args:
        result: Evaluation to render.
        path: Destination file.
        timestamp: Time the evaluation completed.

    Raises:
        OSError: If the file cannot be written.
    """
    path.write_text(render_markdown(result, timestamp), encoding="utf-8")
