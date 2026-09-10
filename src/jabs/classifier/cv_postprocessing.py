"""Postprocessed evaluation of a cross-validation fold.

Cross-validation normally scores a fold using only the labeled frames of the
held-out group, in the order they happen to be stacked in the feature matrix.
That is fine for raw per-frame metrics, but it cannot be used to evaluate the
prediction postprocessing pipeline: stitching, duration filtering, and gap
interpolation all reason about *contiguous* frames, and the labeled rows are a
sparse, gap-collapsed subset of the video (two labeled bouts thousands of
frames apart end up as adjacent rows).

So this module re-predicts the held-out group's full tracks the way the
classify path does, applies the pipeline to those full-length prediction
vectors, and only then restricts to the labeled frames where ground truth
exists. That makes the reported numbers reflect what postprocessing actually
does at inference time, including the runs of frames with no prediction that
:class:`~jabs.behavior.postprocessing.stages.GapInterpolationStage` fills.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from jabs.feature_extraction import IdentityFeatures
from jabs.project.track_labels import TrackLabels

from .inference import predict_identity

if TYPE_CHECKING:
    from jabs.behavior.postprocessing import PostprocessingPipeline
    from jabs.pose_estimation import PoseEstimation
    from jabs.project import Project

    from .protocols import ClassifierProtocol

logger = logging.getLogger(__name__)


def enabled_stage_configs(config: list[dict]) -> list[dict]:
    """Return the stage configurations that are enabled.

    Mirrors the filter :class:`~jabs.behavior.postprocessing.PostprocessingPipeline`
    applies when it builds its stages, so callers can tell whether a pipeline
    would do anything at all before paying for a full-sequence evaluation.

    Args:
        config: Ordered list of stage configuration dicts.

    Returns:
        The subset of ``config`` whose stages are enabled, in order.
    """
    return [stage for stage in config if stage.get("enabled", True)]


@dataclass(frozen=True)
class FoldPostprocessingEvaluation:
    """Ground truth and predictions for one fold, restricted to labeled frames.

    All three arrays are the same length and aligned element-wise: they are the
    concatenation, over every ``(video, identity)`` in the held-out group, of
    that identity's labeled frames.

    Attributes:
        truth: Ground-truth labels.
        raw: Predictions before postprocessing.
        postprocessed: Predictions after the postprocessing pipeline.
    """

    truth: npt.NDArray[np.int8]
    raw: npt.NDArray[np.int8]
    postprocessed: npt.NDArray[np.int8]


def _identity_labels(
    project: Project,
    video: str,
    identity: int,
    behavior: str,
    pose_est: PoseEstimation,
) -> npt.NDArray[np.int8] | None:
    """Load one identity's ground-truth label vector for a behavior.

    Frames where the identity does not exist are forced to
    ``TrackLabels.Label.NONE``, mirroring
    :func:`~jabs.project.parallel_workers.collect_binary_labeled_features` so
    the frames scored here are exactly the fold's test rows.

    Returns:
        The per-frame label vector, or ``None`` when the video has no
        annotations at all.
    """
    labels_obj = project.video_manager.load_video_labels(video, pose_est)
    if labels_obj is None:
        return None
    labels = labels_obj.get_track_labels(str(identity), behavior).get_labels()
    identity_mask = pose_est.identity_mask(identity).astype(bool)
    labels[~identity_mask] = TrackLabels.Label.NONE
    return labels


def evaluate_group_with_postprocessing(
    classifier: ClassifierProtocol,
    project: Project,
    behavior: str,
    members: list[tuple[str, int]],
    pipeline: PostprocessingPipeline,
    behavior_settings: dict,
    window_size: int,
    status_callback: Callable[[str], None] | None = None,
    terminate_callback: Callable[[], None] | None = None,
) -> FoldPostprocessingEvaluation | None:
    """Re-predict a held-out group's full tracks and apply the postprocessing pipeline.

    Members are processed grouped by video so each pose file is opened once, and
    one identity's features are released before the next is loaded - a full-video
    feature matrix for a long video is large enough that holding a whole group's
    worth at once is not viable.

    Args:
        classifier: Classifier trained on this fold's training split.
        project: Project providing videos, poses, annotations, and features.
        behavior: Behavior being evaluated.
        members: ``(video, identity)`` pairs making up the held-out group.
        pipeline: Postprocessing pipeline to evaluate.
        behavior_settings: Behavior-scoped settings used for feature extraction.
        window_size: Window size to use for window features.
        status_callback: Optional callback for status updates.
        terminate_callback: Optional callback that raises if the caller has
            requested early termination.

    Returns:
        Ground truth and predictions restricted to labeled frames, or ``None``
        when the group yielded no labeled frames (nothing to score).
    """
    truth_parts: list[npt.NDArray[np.int8]] = []
    raw_parts: list[npt.NDArray[np.int8]] = []
    postprocessed_parts: list[npt.NDArray[np.int8]] = []

    by_video: dict[str, list[int]] = defaultdict(list)
    for video, identity in members:
        by_video[video].append(identity)

    for video, identities in by_video.items():
        if terminate_callback:
            terminate_callback()
        pose_est = project.load_pose_est(project.video_manager.video_path(video))

        for identity in identities:
            if terminate_callback:
                terminate_callback()
            if status_callback:
                status_callback(f"Postprocessing evaluation: {video} [{identity}]")

            labels = _identity_labels(project, video, identity, behavior, pose_est)
            if labels is None:
                logger.warning(
                    "No annotations found for %s while evaluating postprocessing", video
                )
                continue
            labeled = labels != TrackLabels.Label.NONE
            if not labeled.any():
                continue

            features = IdentityFeatures(
                video,
                identity,
                project.feature_dir,
                pose_est,
                fps=pose_est.fps,
                op_settings=behavior_settings,
                cache_format=project.cache_format,
            )
            prediction = predict_identity(classifier, features, window_size)
            if prediction is None:
                logger.warning(
                    "No features for %s identity %d while evaluating postprocessing",
                    video,
                    identity,
                )
                continue

            postprocessed = pipeline.run(prediction.predictions, prediction.confidence)

            truth_parts.append(labels[labeled])
            raw_parts.append(prediction.predictions[labeled])
            postprocessed_parts.append(postprocessed[labeled])

    if not truth_parts:
        return None

    return FoldPostprocessingEvaluation(
        truth=np.concatenate(truth_parts),
        raw=np.concatenate(raw_parts),
        postprocessed=np.concatenate(postprocessed_parts),
    )
