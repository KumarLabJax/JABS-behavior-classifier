"""Result types for a classifier evaluation run.

Results are kept at their finest grain - one entry per (video, identity, stage)
- and aggregated on demand by summing counts. Aggregating counts rather than
averaging rates keeps a short video from carrying the same weight as a long one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from jabs.behavior.evaluation import BoutMetrics, FrameMetrics

#: Stage keys used to distinguish raw classifier output from postprocessed output.
#: A single postprocessing config uses ``POSTPROCESSED_STAGE``; a sweep uses one
#: ``sweep_<n>`` key per combination, named by :func:`sweep_stage_key`.
RAW_STAGE = "raw"
POSTPROCESSED_STAGE = "postprocessed"


def sweep_stage_key(index: int) -> str:
    """Return the stage key for one sweep combination.

    Args:
        index: Position of the combination in the expanded grid.

    Returns:
        A stage key distinct from ``RAW_STAGE`` and ``POSTPROCESSED_STAGE``.
    """
    return f"sweep_{index}"


@dataclass(frozen=True)
class BoutRecord:
    """One ground-truth or predicted bout and how well it matched the other side.

    These are the rows of the per-bout CSV. Ground-truth and predicted bouts
    share a row shape so the file can be sorted and filtered as one table.

    Attributes:
        video: Video the bout belongs to.
        identity: Identity index within that video.
        stage: ``RAW_STAGE`` or ``POSTPROCESSED_STAGE``.
        source: ``"ground_truth"`` or ``"predicted"``.
        start: First frame of the bout, inclusive.
        end: Last frame of the bout, inclusive.
        duration: Length of the bout in frames.
        evaluable: False when the bout lies entirely in frames the other side
            could not speak to, so it is excluded from rates.
        overlapping_bouts: How many bouts on the other side it intersects at all.
        best_overlap_frames: Most frames shared with any single opposing bout.
        best_iou: Highest IoU against any single opposing bout.
        matched_overlap: Whether it matched under the frame-overlap criterion.
        matched_iou: Whether it matched under the IoU criterion.
    """

    video: str
    identity: int
    stage: str
    source: str
    start: int
    end: int
    duration: int
    evaluable: bool
    overlapping_bouts: int
    best_overlap_frames: int
    best_iou: float
    matched_overlap: bool
    matched_iou: bool


@dataclass(frozen=True)
class IdentityResult:
    """Metrics for a single identity in a single video at a single stage.

    Attributes:
        video: Video name.
        identity: Identity index within that video.
        stage: ``RAW_STAGE`` or ``POSTPROCESSED_STAGE``.
        frame_metrics: Frame-by-frame agreement counts.
        bout_metrics: Bout-level counts keyed by match-criterion label.
    """

    video: str
    identity: int
    stage: str
    frame_metrics: FrameMetrics
    bout_metrics: dict[str, BoutMetrics]


@dataclass(frozen=True)
class EvaluationResult:
    """Everything one evaluation run produced.

    Attributes:
        project_dir: Project the ground truth came from.
        behavior: Behavior that was evaluated.
        classifier_path: Classifier that produced the predictions.
        classifier_type: Human-readable classifier algorithm name.
        window_size: Window size the classifier was trained with.
        stages: Stages present in ``identity_results``, in report order.
        criteria: Bout match-criterion labels, in report order.
        identity_results: One entry per (video, identity, stage).
        bout_records: Per-bout detail rows, empty unless requested.
        skipped_videos: ``(video, reason)`` for videos that were not evaluated.
        unlabeled_identities: ``(video, identity)`` pairs that had no ground
            truth for this behavior, so contributed nothing.
        postprocess_stages: Names of the postprocessing stages that ran, empty
            when postprocessing was not requested.
        prediction_files: Prediction HDF5 files written, empty unless saving
            was requested.
        prediction_write_errors: ``(video, reason)`` for prediction files that
            could not be written. Recorded rather than raised so a failure to
            save does not cost the metrics the run just spent hours computing.
        stage_labels: Display label per stage key. Sweep keys map to their
            combination, e.g. ``"min_duration=60, max_stitch_gap=30"``.
        sweep_axis_names: Column headings for the swept parameters, in column
            order. Empty when this run was not a sweep.
        sweep_values: Per sweep stage key, the value taken on each axis,
            parallel to ``sweep_axis_names``.
        best_stage: Stage key of the best-scoring sweep combination, or None
            when this run was not a sweep. Chosen by bout F1 under the strictest
            criterion, so the detailed tables have one postprocessed stage to
            show rather than all of them.
    """

    project_dir: Path
    behavior: str
    classifier_path: Path
    classifier_type: str
    window_size: int
    stages: tuple[str, ...]
    criteria: tuple[str, ...]
    identity_results: list[IdentityResult] = field(default_factory=list)
    bout_records: list[BoutRecord] = field(default_factory=list)
    skipped_videos: list[tuple[str, str]] = field(default_factory=list)
    unlabeled_identities: list[tuple[str, int]] = field(default_factory=list)
    postprocess_stages: tuple[str, ...] = ()
    prediction_files: list[Path] = field(default_factory=list)
    prediction_write_errors: list[tuple[str, str]] = field(default_factory=list)
    stage_labels: dict[str, str] = field(default_factory=dict)
    sweep_axis_names: tuple[str, ...] = ()
    sweep_values: dict[str, tuple[object, ...]] = field(default_factory=dict)
    best_stage: str | None = None

    @property
    def videos(self) -> list[str]:
        """Evaluated video names, in first-seen order."""
        seen: dict[str, None] = {}
        for result in self.identity_results:
            seen.setdefault(result.video, None)
        return list(seen)

    @property
    def is_sweep(self) -> bool:
        """Whether this run varied postprocessing parameters."""
        return bool(self.sweep_axis_names)

    @property
    def sweep_stages(self) -> tuple[str, ...]:
        """Sweep combination stage keys, in grid order."""
        return tuple(s for s in self.stages if s in self.sweep_values)

    @property
    def detail_stages(self) -> tuple[str, ...]:
        """Stages the detailed per-stage tables cover.

        A sweep has too many combinations to tabulate in full detail, so the
        detail views cover the raw predictions plus the best combination; the
        sweep table carries every combination instead.
        """
        if not self.is_sweep:
            return self.stages
        return tuple(s for s in (RAW_STAGE, self.best_stage) if s is not None)

    def label_for(self, stage: str) -> str:
        """Return the display label for a stage key.

        Args:
            stage: Stage key.

        Returns:
            The configured label, falling back to a generic one.
        """
        return self.stage_labels.get(stage) or stage_label(stage)

    def for_stage(self, stage: str) -> list[IdentityResult]:
        """Return the results belonging to one stage.

        Args:
            stage: ``RAW_STAGE`` or ``POSTPROCESSED_STAGE``.

        Returns:
            Matching results, in the order they were produced.
        """
        return [r for r in self.identity_results if r.stage == stage]

    def for_video(self, stage: str, video: str) -> list[IdentityResult]:
        """Return the results for one video at one stage.

        Args:
            stage: ``RAW_STAGE`` or ``POSTPROCESSED_STAGE``.
            video: Video name.

        Returns:
            Matching results, in the order they were produced.
        """
        return [r for r in self.identity_results if r.stage == stage and r.video == video]


def aggregate_frame_metrics(results: Iterable[IdentityResult]) -> FrameMetrics:
    """Sum frame-level counts over a set of results.

    Args:
        results: Results to combine.

    Returns:
        Summed counts, or an all-zero ``FrameMetrics`` when ``results`` is empty.
    """
    total = FrameMetrics()
    for result in results:
        total = total + result.frame_metrics
    return total


def aggregate_bout_metrics(
    results: Iterable[IdentityResult],
    criterion: str,
) -> BoutMetrics:
    """Sum bout-level counts for one criterion over a set of results.

    Args:
        results: Results to combine.
        criterion: Match-criterion label to select within each result.

    Returns:
        Summed counts, or an all-zero ``BoutMetrics`` when no result carries
        that criterion.
    """
    total = BoutMetrics(criterion_label=criterion)
    for result in results:
        metrics = result.bout_metrics.get(criterion)
        if metrics is not None:
            total = total + metrics
    return total


def stage_label(stage: str) -> str:
    """Return a display label for a stage key.

    Args:
        stage: ``RAW_STAGE`` or ``POSTPROCESSED_STAGE``.

    Returns:
        Capitalized label for tables and headings.
    """
    return {RAW_STAGE: "Raw", POSTPROCESSED_STAGE: "Postprocessed"}.get(stage, stage)


def format_rate(value: float | None, places: int = 3) -> str:
    """Format a rate for display, showing undefined rates as ``n/a``.

    Args:
        value: Rate in ``[0, 1]``, or None when its denominator was empty.
        places: Decimal places to show.

    Returns:
        Formatted string.
    """
    return "n/a" if value is None else f"{value:.{places}f}"


def format_counts(values: Sequence[int]) -> str:
    """Format a sequence of counts as a slash-separated string.

    Args:
        values: Counts to join.

    Returns:
        e.g. ``"12/15"`` for ``(12, 15)``.
    """
    return "/".join(str(v) for v in values)
