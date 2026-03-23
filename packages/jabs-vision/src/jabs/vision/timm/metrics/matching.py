"""Matching utilities for detection and pose evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .types import ImageEvalData, KeypointDetection, KeypointGroundTruth

__all__ = ["MatchResult", "greedy_match", "match_image"]


@dataclass(frozen=True)
class MatchResult:
    """Result of matching detections to ground truths for a single image.

    Attributes:
        tp_flags: Boolean array of shape (N,) indicating true positives.
        scores: Detection confidence scores of shape (N,), sorted descending.
        num_gt: Number of non-crowd ground truths in this image.
        ignore_flags: Boolean array of shape (N,) indicating detections matched
            to crowd GTs that should be excluded from AP/AR computation.
        det_indices: Original detection indices in score-descending order, shape (N,).
        gt_assignments: GT index assigned to each detection (-1 if unmatched), shape (N,).
    """

    tp_flags: npt.NDArray[np.bool_]
    scores: npt.NDArray[np.float64]
    num_gt: int
    ignore_flags: npt.NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
    det_indices: npt.NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    gt_assignments: npt.NDArray[np.intp] = field(
        default_factory=lambda: np.array([], dtype=np.intp)
    )

    def __post_init__(self) -> None:
        """Validate that all arrays have consistent lengths."""
        n = len(self.tp_flags)
        if len(self.scores) != n:
            raise ValueError(
                f"scores length ({len(self.scores)}) must match tp_flags length ({n})"
            )
        if len(self.ignore_flags) > 0 and len(self.ignore_flags) != n:
            raise ValueError(
                f"ignore_flags length ({len(self.ignore_flags)}) must match tp_flags length ({n})"
            )
        if len(self.det_indices) > 0 and len(self.det_indices) != n:
            raise ValueError(
                f"det_indices length ({len(self.det_indices)}) must match tp_flags length ({n})"
            )
        if len(self.gt_assignments) > 0 and len(self.gt_assignments) != n:
            raise ValueError(
                f"gt_assignments length ({len(self.gt_assignments)}) must match "
                f"tp_flags length ({n})"
            )


def greedy_match(
    similarity_matrix: npt.NDArray[np.float64],
    detections: list[KeypointDetection],
    ground_truths: list[KeypointGroundTruth],
    threshold: float,
) -> MatchResult:
    """Perform COCO-style greedy matching between detections and ground truths.

    Detections are sorted by confidence score in descending order. Each detection
    is greedily assigned to the highest-similarity unmatched ground truth above
    the given threshold. Crowd ground truths absorb detections without incurring
    a false positive penalty.

    Args:
        similarity_matrix: Similarity scores of shape (N_det, N_gt).
        detections: List of N_det predicted detections.
        ground_truths: List of N_gt ground-truth annotations.
        threshold: Minimum similarity for a valid match.

    Returns:
        MatchResult containing TP/FP labels, scores, and match assignments.
    """
    n_det = len(detections)
    n_gt = len(ground_truths)

    expected_shape = (n_det, n_gt)
    if similarity_matrix.shape != expected_shape:
        raise ValueError(
            f"similarity_matrix shape {similarity_matrix.shape} does not match "
            f"expected ({n_det}, {n_gt}) from detections and ground_truths"
        )

    if n_det == 0:
        num_non_crowd = sum(1 for gt in ground_truths if not gt.is_crowd)
        return MatchResult(
            tp_flags=np.array([], dtype=np.bool_),
            scores=np.array([], dtype=np.float64),
            num_gt=num_non_crowd,
            ignore_flags=np.array([], dtype=np.bool_),
            det_indices=np.array([], dtype=np.intp),
            gt_assignments=np.array([], dtype=np.intp),
        )

    # Sort detections by score descending
    scores = np.array([d.score for d in detections], dtype=np.float64)
    sorted_det_indices = np.argsort(-scores, kind="stable")

    # Separate crowd and non-crowd GTs
    crowd_flags = np.array([gt.is_crowd for gt in ground_truths], dtype=bool)
    num_non_crowd = int(np.sum(~crowd_flags))

    gt_matched = np.zeros(n_gt, dtype=bool)
    tp_flags = np.zeros(n_det, dtype=np.bool_)
    ignore_flags = np.zeros(n_det, dtype=np.bool_)
    gt_assignments = np.full(n_det, -1, dtype=np.intp)

    for det_idx in sorted_det_indices:
        sim_row = similarity_matrix[det_idx]

        # Try non-crowd GTs first (use >= for inclusive threshold matching)
        best_gt = -1
        best_sim = threshold
        for gt_idx in range(n_gt):
            if gt_matched[gt_idx] or crowd_flags[gt_idx]:
                continue
            if sim_row[gt_idx] >= best_sim:
                best_sim = sim_row[gt_idx]
                best_gt = gt_idx

        if best_gt >= 0:
            tp_flags[det_idx] = True
            gt_matched[best_gt] = True
            gt_assignments[det_idx] = best_gt
            continue

        # Try crowd GTs - absorb detection (excluded from AP/AR computation)
        best_crowd_sim = threshold
        best_crowd_gt = -1
        for gt_idx in range(n_gt):
            if not crowd_flags[gt_idx]:
                continue
            if sim_row[gt_idx] >= best_crowd_sim:
                best_crowd_sim = sim_row[gt_idx]
                best_crowd_gt = gt_idx

        if best_crowd_gt >= 0:
            # Matched to crowd: ignore this detection in AP/AR computation
            ignore_flags[det_idx] = True
            gt_assignments[det_idx] = best_crowd_gt

    # Reorder outputs by descending score
    sorted_tp = tp_flags[sorted_det_indices]
    sorted_ignore = ignore_flags[sorted_det_indices]
    sorted_scores = scores[sorted_det_indices]
    sorted_gt_assignments = gt_assignments[sorted_det_indices]

    return MatchResult(
        tp_flags=sorted_tp,
        scores=sorted_scores,
        num_gt=num_non_crowd,
        ignore_flags=sorted_ignore,
        det_indices=sorted_det_indices,
        gt_assignments=sorted_gt_assignments,
    )


def match_image(
    image: ImageEvalData,
    similarity_matrix: npt.NDArray[np.float64],
    threshold: float,
) -> MatchResult:
    """Match detections to ground truths for a single image.

    Convenience wrapper around ``greedy_match`` that takes an ``ImageEvalData``.

    Args:
        image: Evaluation data for a single image.
        similarity_matrix: Similarity matrix of shape (N_det, N_gt).
        threshold: Minimum similarity for a valid match.

    Returns:
        MatchResult for this image.
    """
    return greedy_match(
        similarity_matrix=similarity_matrix,
        detections=image.detections,
        ground_truths=image.ground_truths,
        threshold=threshold,
    )
