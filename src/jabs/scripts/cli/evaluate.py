"""Evaluate a trained JABS classifier against a densely labeled JABS project.

Runs a trained binary classifier over every pose file in a project, then
compares the predictions against the project's own labels. Intended for a
project where every frame (or nearly every frame) is labeled for the behavior,
so the labels can stand in as ground truth.

The comparison is reported two ways:

\b
  Frame-level  Frame-by-frame agreement. Frames the labels leave unlabeled, and
               frames the classifier could not score because the identity has no
               pose there, are excluded and reported separately.
  Bout-level   Whether each labeled bout was found at all, allowing the
               predicted start and end frames to disagree with the labeled ones.

Bout matching is reported under two criteria side by side, so you can see how
much of the detection rate survives a boundary-quality requirement:

\b
  overlap >= N frames   Permissive - did the classifier notice the bout.
  IoU >= T              Intersection-over-union of the frame ranges, which
                        penalizes a prediction much longer or shorter than the
                        labeled bout.

Matching is many-to-many, so a labeled bout the classifier split in two still
counts as detected and both fragments count as hits. Splitting is reported as
'Fragmented', and the opposite error - one prediction spanning several labeled
bouts - as 'Merged'.

With --postprocess-config, the same comparison is run a second time against the
postprocessed predictions and both are shown side by side.

\b
Examples:
  jabs-cli evaluate /path/to/project --classifier grooming.pickle
  jabs-cli evaluate /path/to/project --classifier grooming.pickle \\
      --postprocess-config pipeline.yaml --out-dir results/
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from jabs.behavior.evaluation import (
    Bout,
    BoutMatchResult,
    BoutMetrics,
    BoutOverlap,
    FrameMetrics,
    IoUCriterion,
    MatchCriterion,
    OverlapCriterion,
    bouts_are_evaluable,
    compute_bout_metrics,
    compute_frame_metrics,
    extract_bouts,
    match_bouts,
    overlapping_pairs,
)
from jabs.behavior.events import ClassLabels
from jabs.behavior.postprocessing import PostprocessingPipeline
from jabs.classifier import Classifier, MultiClassClassifier
from jabs.core.utils import pose_file_stem
from jabs.feature_extraction import IdentityFeatures
from jabs.project import Project
from jabs.project.prediction_manager import PredictionManager
from jabs.scripts.classify import load_classifier_from_pickle
from jabs.video_reader.utilities import get_fps_and_nframes

from . import evaluate_report
from .evaluate_results import (
    POSTPROCESSED_STAGE,
    RAW_STAGE,
    BoutRecord,
    EvaluationResult,
    IdentityResult,
    aggregate_bout_metrics,
    aggregate_frame_metrics,
    sweep_stage_key,
)
from .evaluate_sweep import (
    DEFAULT_MAX_COMBINATIONS,
    SweepAxis,
    SweepPoint,
    axis_column_names,
    expand_sweep,
    format_point_label,
)
from .postprocessing import load_config_file

logger = logging.getLogger(__name__)

#: Source labels used in the per-bout CSV.
GROUND_TRUTH_SOURCE = "ground_truth"
PREDICTED_SOURCE = "predicted"


@dataclass(frozen=True)
class IdentityComparison:
    """Everything one (identity, stage) comparison produced.

    Carries the intermediate bout lists and overlaps alongside the metrics so
    the per-bout CSV rows can be built without repeating the matching work.

    Attributes:
        frame_metrics: Frame-by-frame counts.
        bout_metrics: Bout counts keyed by match-criterion label.
        truth_bouts: Ground-truth behavior bouts.
        predicted_bouts: Predicted behavior bouts.
        truth_evaluable: One flag per ground-truth bout.
        predicted_evaluable: One flag per predicted bout.
        overlaps: Every intersecting bout pair.
        matches: Match results keyed by match-criterion label.
    """

    frame_metrics: FrameMetrics
    bout_metrics: dict[str, BoutMetrics]
    truth_bouts: list[Bout]
    predicted_bouts: list[Bout]
    truth_evaluable: list[bool]
    predicted_evaluable: list[bool]
    overlaps: list[BoutOverlap]
    matches: dict[str, BoutMatchResult]


def compare_identity(
    truth: npt.NDArray[np.integer],
    predicted: npt.NDArray[np.integer],
    criteria: Sequence[MatchCriterion],
) -> IdentityComparison:
    """Compare one identity's predictions against its ground-truth labels.

    Args:
        truth: Per-frame ground-truth labels.
        predicted: Per-frame predicted classes, the same length as ``truth``.
        criteria: Bout match criteria to evaluate, each reported separately.

    Returns:
        Frame and bout metrics plus the intermediate bout data.
    """
    truth_bouts = extract_bouts(truth)
    predicted_bouts = extract_bouts(predicted)

    # a labeled bout the classifier could not score, or a predicted bout in
    # unlabeled frames, is excluded rather than charged to the classifier
    truth_evaluable = bouts_are_evaluable(truth_bouts, predicted)
    predicted_evaluable = bouts_are_evaluable(predicted_bouts, truth)

    # the sweep is criterion-independent, so run it once and match against it
    overlaps = overlapping_pairs(truth_bouts, predicted_bouts)

    matches: dict[str, BoutMatchResult] = {}
    bout_metrics: dict[str, BoutMetrics] = {}
    for criterion in criteria:
        result = match_bouts(truth_bouts, predicted_bouts, criterion, overlaps=overlaps)
        matches[criterion.label] = result
        bout_metrics[criterion.label] = compute_bout_metrics(
            result, truth_evaluable, predicted_evaluable
        )

    return IdentityComparison(
        frame_metrics=compute_frame_metrics(truth, predicted),
        bout_metrics=bout_metrics,
        truth_bouts=truth_bouts,
        predicted_bouts=predicted_bouts,
        truth_evaluable=truth_evaluable,
        predicted_evaluable=predicted_evaluable,
        overlaps=overlaps,
        matches=matches,
    )


def build_bout_records(
    comparison: IdentityComparison,
    video: str,
    identity: int,
    stage: str,
    overlap_criterion: str,
    iou_criterion: str,
) -> list[BoutRecord]:
    """Build the per-bout CSV rows for one comparison.

    Args:
        comparison: Result of :func:`compare_identity`.
        video: Video the bouts belong to.
        identity: Identity index within that video.
        stage: Stage key the comparison belongs to.
        overlap_criterion: Label of the frame-overlap criterion.
        iou_criterion: Label of the IoU criterion.

    Returns:
        One row per ground-truth bout followed by one per predicted bout.
    """
    # best overlap and IoU per bout, on each side, from the single sweep
    best_frames: dict[tuple[str, int], int] = {}
    best_iou: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}
    for overlap in comparison.overlaps:
        for key in (
            (GROUND_TRUTH_SOURCE, overlap.truth_index),
            (PREDICTED_SOURCE, overlap.predicted_index),
        ):
            best_frames[key] = max(best_frames.get(key, 0), overlap.frames)
            best_iou[key] = max(best_iou.get(key, 0.0), overlap.iou)
            counts[key] = counts.get(key, 0) + 1

    overlap_matches = comparison.matches[overlap_criterion]
    iou_matches = comparison.matches[iou_criterion]

    records: list[BoutRecord] = []
    for source, bouts, evaluable, overlap_side, iou_side in (
        (
            GROUND_TRUTH_SOURCE,
            comparison.truth_bouts,
            comparison.truth_evaluable,
            overlap_matches.truth_matches,
            iou_matches.truth_matches,
        ),
        (
            PREDICTED_SOURCE,
            comparison.predicted_bouts,
            comparison.predicted_evaluable,
            overlap_matches.predicted_matches,
            iou_matches.predicted_matches,
        ),
    ):
        for index, bout in enumerate(bouts):
            key = (source, index)
            records.append(
                BoutRecord(
                    video=video,
                    identity=identity,
                    stage=stage,
                    source=source,
                    start=bout.start,
                    end=bout.end,
                    duration=bout.duration,
                    evaluable=evaluable[index],
                    overlapping_bouts=counts.get(key, 0),
                    best_overlap_frames=best_frames.get(key, 0),
                    best_iou=round(best_iou.get(key, 0.0), 4),
                    matched_overlap=bool(overlap_side[index]),
                    matched_iou=bool(iou_side[index]),
                )
            )
    return records


class _PredictionAccumulator:
    """Collects per-identity predictions into the per-video arrays a file needs.

    ``PredictionManager.write_predictions`` writes one record per video covering
    every identity, but the evaluation loop produces one identity at a time.
    Identities that fail to classify are simply never added, leaving their rows
    at ``ClassLabels.NONE``.

    Args:
        num_identities: Identities in the pose file.
        num_frames: Frames in the pose file.
        with_postprocessed: Whether to also collect postprocessed predictions.
    """

    def __init__(self, num_identities: int, num_frames: int, with_postprocessed: bool) -> None:
        shape = (num_identities, num_frames)
        self.predictions: npt.NDArray[np.int8] = np.full(shape, ClassLabels.NONE, dtype=np.int8)
        self.probabilities: npt.NDArray[np.float32] = np.zeros(shape, dtype=np.float32)
        self.postprocessed: npt.NDArray[np.int8] | None = (
            np.full(shape, ClassLabels.NONE, dtype=np.int8) if with_postprocessed else None
        )
        self.has_predictions = False

    def add(
        self,
        identity: int,
        predicted: npt.NDArray[np.integer],
        confidence: npt.NDArray[np.floating],
        postprocessed: npt.NDArray[np.integer] | None,
    ) -> None:
        """Record one identity's full-length prediction vectors.

        Args:
            identity: Identity index, used as the row.
            predicted: Per-frame predicted classes.
            confidence: Per-frame confidence in the predicted class.
            postprocessed: Per-frame postprocessed classes, or None.
        """
        self.predictions[identity] = predicted
        self.probabilities[identity] = confidence
        if self.postprocessed is not None and postprocessed is not None:
            self.postprocessed[identity] = postprocessed
        self.has_predictions = True


@dataclass(frozen=True)
class PostprocessingPlan:
    """The postprocessing pipelines to evaluate alongside the raw predictions.

    A plan holds one pipeline per combination of swept parameter values. Because
    a pipeline is a pure function of the predictions, every combination reuses a
    single classification pass - the grid costs stage arithmetic, not features.

    Attributes:
        axes: Swept parameters, in column order. Empty for a single config.
        points: One expanded combination per pipeline, parallel to ``pipelines``.
        pipelines: Built pipelines, parallel to ``points``.
        stage_keys: Result stage key per pipeline, parallel to ``points``.
        labels: Display label per stage key.
        stage_names: Class names of the stages in the pipelines, for reporting.
    """

    axes: list[SweepAxis]
    points: list[SweepPoint]
    pipelines: list[PostprocessingPipeline]
    stage_keys: list[str]
    labels: dict[str, str]
    stage_names: tuple[str, ...]

    @property
    def is_sweep(self) -> bool:
        """Whether more than one parameter combination is being evaluated."""
        return bool(self.axes)


def build_postprocessing_plan(
    config: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    behavior: str,
    max_combinations: int = DEFAULT_MAX_COMBINATIONS,
) -> PostprocessingPlan:
    """Expand a postprocessing config into the pipelines to evaluate.

    A parameter holding a list is a sweep axis. That syntax is specific to this
    command; the expansion produces single-valued configs before any of them
    reaches a pipeline, which is what every other consumer requires.

    Args:
        config: Parsed config - a stage list, or a mapping of behavior name to
            stage list.
        behavior: Behavior being evaluated, used to select from a mapping.
        max_combinations: Ceiling on the size of the expanded grid.

    Returns:
        The plan to hand to :func:`run_evaluation`.

    Raises:
        click.ClickException: If the config is malformed, names no pipeline for
            this behavior, or expands past ``max_combinations``.
    """
    stage_config = _resolve_pipeline_config(config, behavior)
    try:
        axes, points = expand_sweep(stage_config, max_combinations=max_combinations)
    except ValueError as exc:
        raise click.ClickException(f"Invalid postprocessing config: {exc}") from exc

    pipelines: list[PostprocessingPipeline] = []
    for point in points:
        try:
            pipelines.append(PostprocessingPipeline(point.config))
        except ValueError as exc:
            detail = format_point_label(axes, point)
            where = f" for {detail}" if axes else ""
            raise click.ClickException(f"Invalid postprocessing config{where}: {exc}") from exc

    sweeping = bool(axes)
    stage_keys = [sweep_stage_key(p.index) if sweeping else POSTPROCESSED_STAGE for p in points]
    labels = {
        key: format_point_label(axes, point) if sweeping else "Postprocessed"
        for key, point in zip(stage_keys, points, strict=True)
    }
    stage_names = tuple(type(s).__name__ for s in pipelines[0].stages) if pipelines else ()

    return PostprocessingPlan(
        axes=axes,
        points=points,
        pipelines=pipelines,
        stage_keys=stage_keys,
        labels=labels,
        stage_names=stage_names,
    )


def select_best_stage(result: EvaluationResult, candidates: Sequence[str]) -> str | None:
    """Pick the best-scoring stage among a set of candidates.

    Ranked by bout F1 under the strictest criterion available - the last one,
    which is the IoU criterion - because that is the measure that reflects
    boundary quality rather than mere detection. Frame F1 breaks ties. A
    candidate whose F1 is undefined ranks last rather than as zero.

    Args:
        result: Populated evaluation result.
        candidates: Stage keys to choose between.

    Returns:
        The winning stage key, or None when ``candidates`` is empty.
    """
    if not candidates:
        return None
    criterion = result.criteria[-1]

    def score(stage: str) -> tuple[float, float]:
        rows = result.for_stage(stage)
        bout_f1 = aggregate_bout_metrics(rows, criterion).f1
        frame_f1 = aggregate_frame_metrics(rows).f1_behavior
        return (
            bout_f1 if bout_f1 is not None else -1.0,
            frame_f1 if frame_f1 is not None else -1.0,
        )

    return max(candidates, key=score)


def _predict_identity(
    classifier: Classifier,
    pose_path: Path,
    pose_est: Any,
    identity: int,
    feature_dir: Path | None,
    fps: int,
    cache_format: Any,
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.floating]]:
    """Run the classifier over one identity of one pose file.

    Args:
        classifier: Trained binary classifier.
        pose_path: Path to the pose file, used as the feature-cache key.
        pose_est: Open ``PoseEstimation`` for that file.
        identity: Identity index to classify.
        feature_dir: Feature cache directory, or None to compute without caching.
        fps: Frames per second, used for time-derived features.
        cache_format: Feature cache format to read and write.

    Returns:
        Tuple of ``(predicted_class, confidence)``, each of length
        ``pose_est.num_frames``. Frames with no pose are ``-1`` in
        ``predicted_class``.
    """
    settings = classifier.project_settings
    features = IdentityFeatures(
        pose_path,
        identity,
        feature_dir,
        pose_est,
        fps=fps,
        op_settings=settings,
        cache_window=True,
        cache_format=cache_format,
    ).get_features(settings["window_size"])

    predictions = np.full(pose_est.num_frames, ClassLabels.NONE, dtype=np.int8)
    confidence = np.zeros(pose_est.num_frames, dtype=np.float32)

    data = classifier.combine_data(
        pd.DataFrame(features["per_frame"]), pd.DataFrame(features["window"])
    )
    if data.shape[0] > 0:
        probabilities = classifier.predict_proba(data, features["frame_indexes"])
        predictions, confidence = classifier.derive_predictions(probabilities)

    return predictions, confidence


def _align(
    truth: npt.NDArray[np.integer],
    predicted: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], int]:
    """Truncate a label and prediction vector to a common length.

    The project validates that a video's frame count matches its pose file, but
    a project assembled by hand can still disagree. Truncating to the shorter of
    the two keeps one bad video from aborting the run; the caller reports the
    discrepancy.

    Args:
        truth: Per-frame ground-truth labels.
        predicted: Per-frame predicted classes.

    Returns:
        Tuple of ``(truth, predicted, dropped_frames)``.
    """
    n = min(len(truth), len(predicted))
    return truth[:n], predicted[:n], abs(len(truth) - len(predicted))


def _resolve_pipeline_config(
    config: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    behavior: str,
) -> list[dict[str, Any]]:
    """Pick this behavior's stage list out of a postprocessing config.

    Args:
        config: Parsed config - a stage list, or a mapping of behavior name to
            stage list.
        behavior: Behavior being evaluated.

    Returns:
        The stage list to build a pipeline from.

    Raises:
        click.ClickException: If the config is a mapping that does not contain
            the behavior, or is neither a list nor a mapping.
    """
    if isinstance(config, list):
        return config
    if isinstance(config, dict):
        if behavior not in config:
            available = ", ".join(repr(b) for b in config) or "none"
            raise click.ClickException(
                f"Behavior '{behavior}' not found in the postprocessing config. "
                f"Behaviors in config: {available}"
            )
        return config[behavior]
    raise click.ClickException(
        "Postprocessing config must contain a JSON/YAML list or object at the top level."
    )


def load_binary_classifier(classifier_path: Path) -> Classifier:
    """Load a trained binary classifier, rejecting multi-class ones.

    Args:
        classifier_path: Path to the saved classifier pickle.

    Returns:
        The loaded binary classifier.

    Raises:
        click.ClickException: If the file cannot be loaded or holds a
            multi-class classifier.
    """
    try:
        classifier = load_classifier_from_pickle(classifier_path)
    except Exception as exc:
        raise click.ClickException(
            f"Unable to load classifier from {classifier_path}: {exc}"
        ) from exc

    if isinstance(classifier, MultiClassClassifier):
        raise click.ClickException(
            f"This command evaluates binary classifiers only, but {classifier_path} holds a "
            "multi-class classifier. Bout comparison treats one class as the behavior and "
            "everything else as background, which has no meaning for multi-class predictions."
        )
    return classifier


def resolve_behavior(classifier: Classifier, behavior: str | None) -> str:
    """Determine which behavior's labels are the ground truth.

    Args:
        classifier: Loaded binary classifier.
        behavior: Explicit behavior name, or None to use the classifier's own.

    Returns:
        The behavior name to look for in the project's annotations.

    Raises:
        click.ClickException: If no behavior name is available from either
            source.
    """
    resolved = behavior or classifier.behavior_name
    if not resolved:
        raise click.ClickException(
            "The classifier does not record a behavior name. Pass --behavior to name the "
            "behavior whose labels should be used as ground truth."
        )
    return resolved


def run_evaluation(
    project_dir: Path,
    classifier: Classifier,
    classifier_path: Path,
    behavior: str,
    criteria: Sequence[MatchCriterion],
    plan: PostprocessingPlan | None,
    feature_dir: Path | None,
    fps_override: int | None,
    collect_bout_records: bool,
    save_predictions_dir: Path | None = None,
    progress_callback: Any = None,
) -> EvaluationResult:
    """Classify every pose file in a project and compare against its labels.

    Args:
        project_dir: JABS project holding the ground-truth labels and pose files.
        classifier: Loaded binary classifier, from :func:`load_binary_classifier`.
        classifier_path: Path the classifier was loaded from, recorded in the
            result for reporting.
        behavior: Behavior whose labels are the ground truth, from
            :func:`resolve_behavior`.
        criteria: Bout match criteria, each reported separately. The first must
            be the frame-overlap criterion and the second the IoU criterion,
            which is the order the per-bout records assume.
        plan: Postprocessing pipelines to evaluate alongside the raw
            predictions, from :func:`build_postprocessing_plan`, or None to
            evaluate raw predictions only. A plan with several combinations is
            a sweep: all of them are applied to one classification pass.
        feature_dir: Feature cache directory. Defaults to the project's own.
        fps_override: Frames per second to use for every video, skipping the
            per-video lookup. None reads it from each video.
        collect_bout_records: Whether to build the per-bout detail rows.
        save_predictions_dir: Directory to write one prediction HDF5 file per
            video into, in the same format `jabs-classify` produces. None
            skips saving. The directory must already exist.
        progress_callback: Called after each identity with the number of
            identities to advance by, defaulting to one.

    Returns:
        The completed evaluation.

    Raises:
        ValueError: If fewer than two criteria are given.
        click.ClickException: If the project is not a JABS project, or nothing
            in it could be evaluated.
    """
    # the per-bout records label their two match columns from criteria[0] and
    # criteria[1], so a shorter list would fail further down with an IndexError
    if len(criteria) < 2:
        raise ValueError(
            f"run_evaluation expects at least two criteria (overlap, IoU), got {len(criteria)}"
        )

    if not Project.is_valid_project_directory(project_dir):
        raise click.ClickException(f"Not a valid JABS project directory: {project_dir}")

    project = Project(project_dir, enable_session_tracker=False)
    if behavior not in project.settings_manager.behavior_names:
        logger.warning(
            "Behavior '%s' is not listed in the project settings; looking for labels anyway",
            behavior,
        )

    stages = (RAW_STAGE, *(plan.stage_keys if plan is not None else ()))
    result = EvaluationResult(
        project_dir=project_dir,
        behavior=behavior,
        classifier_path=classifier_path,
        classifier_type=classifier.classifier_name,
        window_size=classifier.project_settings["window_size"],
        stages=stages,
        criteria=tuple(c.label for c in criteria),
        postprocess_stages=plan.stage_names if plan is not None else (),
        stage_labels={RAW_STAGE: "Raw", **(plan.labels if plan is not None else {})},
        sweep_axis_names=tuple(axis_column_names(plan.axes)) if plan is not None else (),
        sweep_values=(
            {key: point.values for key, point in zip(plan.stage_keys, plan.points, strict=True)}
            if plan is not None and plan.is_sweep
            else {}
        ),
    )

    cache_dir = feature_dir if feature_dir is not None else project.feature_dir
    overlap_label, iou_label = result.criteria[0], result.criteria[1]

    def skip_video(video: str, reason: str) -> None:
        """Record a skipped video and advance the progress bar past its identities."""
        result.skipped_videos.append((video, reason))
        if progress_callback is not None:
            remaining = project.video_manager.get_video_identity_count(video)
            if remaining:
                progress_callback(remaining)

    for video in project.video_manager.videos:
        try:
            pose_path = project.video_manager.get_cached_pose_path(video)
            pose_est = project.load_pose_est(project.video_manager.video_path(video))
            video_labels = project.video_manager.load_video_labels(video, pose_est)
        except Exception as exc:
            skip_video(video, str(exc))
            logger.warning("Skipping %s: %s", video, exc, exc_info=True)
            continue

        if video_labels is None:
            skip_video(video, "no annotations in the project")
            continue

        if fps_override is not None:
            fps = fps_override
        else:
            try:
                fps, _ = get_fps_and_nframes(project.video_manager.video_path(video))
            except OSError as exc:
                skip_video(video, f"cannot read frame rate: {exc}")
                continue

        # accumulated across identities so the saved file matches the shape
        # jabs-classify writes: one record per video, all identities in it
        # a sweep has no single postprocessed array to save, so only the raw
        # predictions are written; see the note in the command's help
        save_postprocessed = plan is not None and not plan.is_sweep
        saved = (
            _PredictionAccumulator(
                pose_est.num_identities, pose_est.num_frames, save_postprocessed
            )
            if save_predictions_dir is not None
            else None
        )

        for identity in pose_est.identities:
            try:
                predicted, confidence = _predict_identity(
                    classifier, pose_path, pose_est, identity, cache_dir, fps, project.cache_format
                )
            except Exception as exc:
                result.skipped_videos.append((f"{video} (identity {identity})", str(exc)))
                logger.warning(
                    "Skipping identity %s of %s: %s", identity, video, exc, exc_info=True
                )
                if progress_callback is not None:
                    progress_callback()
                continue

            truth = video_labels.get_track_labels(str(identity), behavior).get_labels()
            truth, predicted_aligned, dropped = _align(truth, predicted)
            if dropped:
                logger.warning(
                    "%s identity %s: label and prediction lengths differ by %s frame(s); "
                    "compared the first %s",
                    video,
                    identity,
                    dropped,
                    len(truth),
                )

            if not np.any(truth != ClassLabels.NONE):
                result.unlabeled_identities.append((video, int(identity)))

            stage_vectors = {RAW_STAGE: predicted_aligned}
            postprocessed = None
            if plan is not None:
                for key, pipeline in zip(plan.stage_keys, plan.pipelines, strict=True):
                    vector = predicted.copy()
                    for stage in pipeline.stages:
                        vector = stage.apply(vector, confidence)
                    stage_vectors[key] = vector[: len(truth)]
                    if not plan.is_sweep:
                        postprocessed = vector

            # the saved file records the full pose-length vectors, not the ones
            # truncated to the labels, so it stands on its own as a prediction file
            if saved is not None:
                saved.add(int(identity), predicted, confidence, postprocessed)

            for stage_name, vector in stage_vectors.items():
                comparison = compare_identity(truth, vector, criteria)
                result.identity_results.append(
                    IdentityResult(
                        video=video,
                        identity=int(identity),
                        stage=stage_name,
                        frame_metrics=comparison.frame_metrics,
                        bout_metrics=comparison.bout_metrics,
                    )
                )
                if collect_bout_records:
                    result.bout_records.extend(
                        build_bout_records(
                            comparison,
                            video,
                            int(identity),
                            stage_name,
                            overlap_label,
                            iou_label,
                        )
                    )

            if progress_callback is not None:
                progress_callback()

        if saved is not None and saved.has_predictions:
            output_path = save_predictions_dir / f"{pose_file_stem(pose_path)}_behavior.h5"
            try:
                PredictionManager.write_predictions(
                    behavior,
                    output_path,
                    saved.predictions,
                    saved.probabilities,
                    pose_est,
                    classifier,
                    postprocessed_predictions=saved.postprocessed,
                )
            except Exception as exc:
                # the metrics this run just computed are worth more than the
                # side output, so record the failure and keep going
                result.prediction_write_errors.append((video, str(exc)))
                logger.error("Cannot write predictions for %s: %s", video, exc, exc_info=True)
            else:
                result.prediction_files.append(output_path)
                logger.info("Wrote predictions for %s to %s", video, output_path)

    if plan is not None and plan.is_sweep:
        best = select_best_stage(result, plan.stage_keys)
        result = dataclasses.replace(result, best_stage=best)

    if not result.identity_results:
        raise click.ClickException(
            "Nothing was evaluated. Check that the project has pose files and that its "
            f"annotations contain labels for '{behavior}'."
        )

    return result


@click.command(
    name="evaluate",
    context_settings={"max_content_width": 120},
    help=__doc__,
)
@click.argument(
    "project_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--classifier",
    "classifier_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Trained binary classifier, as produced by `jabs-classify train` or saved by the GUI.",
)
@click.option(
    "--behavior",
    default=None,
    type=str,
    help=(
        "Behavior whose project labels are the ground truth. Defaults to the behavior name "
        "recorded in the classifier."
    ),
)
@click.option(
    "--postprocess-config",
    "postprocess_config",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "JSON or YAML postprocessing pipeline config. When given, the postprocessed predictions "
        "are compared against the ground truth alongside the raw ones. Same format as "
        "`jabs-cli postprocess --config`."
    ),
)
@click.option(
    "--min-overlap",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="Frames two bouts must share to match under the frame-overlap criterion.",
)
@click.option(
    "--iou-threshold",
    default=0.5,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0, min_open=True),
    help="Intersection-over-union two bouts must reach to match under the IoU criterion.",
)
@click.option(
    "--max-sweep-combinations",
    default=DEFAULT_MAX_COMBINATIONS,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        "Ceiling on the number of parameter combinations a swept config may expand to. Each "
        "combination reuses the single classification pass, but still costs a full pass of "
        "every stage over every identity."
    ),
)
@click.option(
    "--feature-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Feature cache directory. Defaults to the project's own feature cache.",
)
@click.option(
    "--fps",
    default=None,
    type=int,
    help=(
        "Frames per second to assume for every video, skipping the per-video lookup. "
        "Defaults to reading it from each video file."
    ),
)
@click.option(
    "--save-predictions",
    "save_predictions",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help=(
        "Write one prediction HDF5 file per video into this directory, in the same format "
        "`jabs-classify` produces. Lets you re-run `jabs-cli postprocess` against these "
        "predictions without recomputing features. Created if it does not exist."
    ),
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help=(
        "Write the JSON summary, per-bout CSV, and markdown report into this directory using "
        "generated filenames. Created if it does not exist."
    ),
)
@click.option(
    "--json-out",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Write the JSON metrics summary to this path.",
)
@click.option(
    "--csv-out",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Write the per-bout detail rows to this path.",
)
@click.option(
    "--report-out",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Write the markdown report to this path.",
)
@click.option(
    "--per-video/--no-per-video",
    default=False,
    show_default=True,
    help="Also print a per-video breakdown to the console.",
)
@click.pass_context
def evaluate_command(
    ctx: click.Context,
    project_dir: Path,
    classifier_path: Path,
    behavior: str | None,
    postprocess_config: Path | None,
    min_overlap: int,
    iou_threshold: float,
    max_sweep_combinations: int,
    feature_dir: Path | None,
    fps: int | None,
    save_predictions: Path | None,
    out_dir: Path | None,
    json_out: Path | None,
    csv_out: Path | None,
    report_out: Path | None,
    per_video: bool,
) -> None:
    """Evaluate a trained classifier against a densely labeled JABS project."""
    console = Console()

    # the criterion order here is load-bearing: run_evaluation labels the
    # per-bout CSV's matched_overlap/matched_iou columns from criteria[0]/[1]
    criteria: list[MatchCriterion] = [
        OverlapCriterion(min_frames=min_overlap),
        IoUCriterion(threshold=iou_threshold),
    ]

    # load the classifier first: the behavior name it records is what the
    # postprocessing config lookup and the generated filenames key off
    classifier = load_binary_classifier(classifier_path)
    behavior = resolve_behavior(classifier, behavior)

    plan = None
    if postprocess_config is not None:
        plan = build_postprocessing_plan(
            load_config_file(postprocess_config),
            behavior,
            max_combinations=max_sweep_combinations,
        )
        if not plan.stage_names:
            console.print(
                "[yellow]Warning: every stage in the postprocessing config is disabled; "
                "the postprocessed results will match the raw ones.[/yellow]"
            )
        if plan.is_sweep:
            console.print(
                f"Sweeping {len(plan.points)} parameter combination(s) over "
                f"{', '.join(axis_column_names(plan.axes))} - the project is classified once and "
                "every combination is applied to those predictions."
            )
            if save_predictions is not None:
                console.print(
                    "[yellow]Note: a sweep has no single postprocessed result, so "
                    "--save-predictions will write the raw predictions only. Re-run with scalar "
                    "parameter values to save a postprocessed file for your chosen "
                    "combination.[/yellow]"
                )

    # resolve output paths before the expensive work, so a bad path fails fast
    timestamp = datetime.now()
    if save_predictions is not None:
        try:
            save_predictions.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise click.ClickException(f"Cannot create prediction directory: {exc}") from exc

    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise click.ClickException(f"Cannot create output directory: {exc}") from exc
        stem = f"{behavior}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        json_out = json_out or out_dir / f"{stem}_evaluation.json"
        csv_out = csv_out or out_dir / f"{stem}_bouts.csv"
        report_out = report_out or out_dir / f"{stem}_evaluation.md"

    if ctx.obj["VERBOSE"]:
        console.print(f"Project:    {project_dir}")
        console.print(f"Classifier: {classifier_path}")
        console.print(f"Behavior:   {behavior}")
        console.print(f"Criteria:   {', '.join(c.label for c in criteria)}")
        if save_predictions is not None:
            console.print(f"Predictions: {save_predictions}")

    project = Project(project_dir, enable_session_tracker=False)
    total = sum(
        project.video_manager.get_video_identity_count(v) for v in project.video_manager.videos
    )

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed} of {task.total} identities"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        task_id = progress.add_task("Classifying", total=total or None)
        result = run_evaluation(
            project_dir=project_dir,
            classifier=classifier,
            classifier_path=classifier_path,
            behavior=behavior,
            criteria=criteria,
            plan=plan,
            feature_dir=feature_dir,
            fps_override=fps,
            collect_bout_records=csv_out is not None,
            save_predictions_dir=save_predictions,
            progress_callback=lambda advance=1: progress.advance(task_id, advance),
        )

    evaluate_report.print_console_report(result, console, per_video=per_video)

    for path, writer, description in (
        (json_out, lambda p: evaluate_report.write_json(result, p, timestamp), "JSON summary"),
        (csv_out, lambda p: evaluate_report.write_csv(result.bout_records, p), "per-bout CSV"),
        (
            report_out,
            lambda p: evaluate_report.write_markdown(result, p, timestamp),
            "markdown report",
        ),
    ):
        if path is None:
            continue
        try:
            writer(path)
        except OSError as exc:
            raise click.ClickException(f"Cannot write {description}: {exc}") from exc
        console.print(f"Wrote {description} to [bold]{path}[/bold]")
