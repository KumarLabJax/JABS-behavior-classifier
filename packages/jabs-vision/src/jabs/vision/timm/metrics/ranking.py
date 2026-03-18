"""Ranking metrics for detection and pose evaluation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .matching import MatchResult

__all__ = ["COCO_RECALL_THRESHOLDS", "compute_ap_ar", "interpolate_ap"]

# COCO uses 101 recall thresholds for interpolated precision
COCO_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)


def compute_ap_ar(match_results: list[MatchResult]) -> tuple[float, float]:
    """Compute AP and AR from match results across multiple images.

    Aggregates TP/FP labels and scores across all images, computes the
    precision-recall curve, and applies 101-point COCO interpolation.

    Args:
        match_results: List of MatchResult from each image at a single threshold.

    Returns:
        Tuple of (average_precision, average_recall).
    """
    total_gt = sum(mr.num_gt for mr in match_results)

    if total_gt == 0:
        return 0.0, 0.0

    # Concatenate all results, maintaining score-descending order via merge sort
    non_empty_scores = [mr.scores for mr in match_results if len(mr.scores) > 0]
    non_empty_tp = [mr.tp_flags for mr in match_results if len(mr.tp_flags) > 0]
    non_empty_ignore = [mr.ignore_flags for mr in match_results if len(mr.ignore_flags) > 0]

    if len(non_empty_scores) == 0:
        return 0.0, 0.0

    all_scores = np.concatenate(non_empty_scores)
    all_tp = np.concatenate(non_empty_tp)

    if len(all_scores) == 0:
        return 0.0, 0.0

    # Filter out crowd-matched (ignored) detections before building PR curve
    if len(non_empty_ignore) > 0:
        all_ignore = np.concatenate(non_empty_ignore)
        if len(all_ignore) == len(all_scores):
            keep = ~all_ignore
            all_scores = all_scores[keep]
            all_tp = all_tp[keep]

    if len(all_scores) == 0:
        return 0.0, 0.0

    # Sort by score descending (stable sort preserves order for equal scores)
    sorted_indices = np.argsort(-all_scores, kind="stable")
    all_tp = all_tp[sorted_indices]

    # Cumulative TP and FP counts
    cum_tp = np.cumsum(all_tp).astype(np.float64)
    cum_fp = np.cumsum(~all_tp).astype(np.float64)

    # Precision and recall curves
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / total_gt

    # Maximum recall achieved = AR
    ar = float(recall[-1]) if len(recall) > 0 else 0.0

    # 101-point COCO interpolation
    ap = interpolate_ap(precision, recall)

    return ap, ar


def interpolate_ap(
    precision: npt.NDArray[np.float64],
    recall: npt.NDArray[np.float64],
) -> float:
    """Compute AP using 101-point COCO-style interpolation.

    For each of the 101 recall thresholds [0.0, 0.01, ..., 1.0], the
    interpolated precision is the maximum precision at any recall >= threshold.

    Args:
        precision: Precision values at each detection, shape (N,).
        recall: Recall values at each detection, shape (N,).

    Returns:
        Interpolated average precision.
    """
    # Append sentinel values
    recall_with_sentinel = np.concatenate(([0.0], recall, [recall[-1] + 1e-6]))
    precision_with_sentinel = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing (right to left)
    for i in range(len(precision_with_sentinel) - 2, -1, -1):
        precision_with_sentinel[i] = max(
            precision_with_sentinel[i], precision_with_sentinel[i + 1]
        )

    # Interpolate at 101 recall points
    interpolated = np.zeros(len(COCO_RECALL_THRESHOLDS), dtype=np.float64)
    for i, r_thresh in enumerate(COCO_RECALL_THRESHOLDS):
        # Find the insertion point: first recall value >= r_thresh
        idx = np.searchsorted(recall_with_sentinel, r_thresh)
        if idx < len(precision_with_sentinel):
            interpolated[i] = precision_with_sentinel[idx]
        else:
            interpolated[i] = 0.0

    return float(np.mean(interpolated))
