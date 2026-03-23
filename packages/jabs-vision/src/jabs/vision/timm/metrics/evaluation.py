"""High-level evaluation entrypoints for the metrics SDK."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .matching import MatchResult, greedy_match
from .pck import compute_pck
from .ranking import compute_ap_ar
from .similarity import compute_bbox_iou_matrix, compute_oks_matrix
from .types import (
    DetectionAPResult,
    ImageEvalData,
    PCKResult,
    PerThresholdResult,
    PoseAPResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "THRESHOLDS",
    "evaluate_detection",
    "evaluate_pck",
    "evaluate_pose",
]

THRESHOLDS = np.arange(0.50, 1.00, 0.05)
"""Standard COCO IoU/OKS thresholds: [0.50, 0.55, ..., 0.95]."""


def evaluate_pose(
    images: Sequence[ImageEvalData],
    sigmas: npt.NDArray[np.float64],
) -> PoseAPResult:
    """Evaluate pose estimation using OKS-based COCO metrics.

    Computes AP and AR across the standard COCO OKS thresholds
    [0.50, 0.55, ..., 0.95] using 101-point precision-recall interpolation.

    Args:
        images: Sequence of per-image evaluation data.
        sigmas: Per-keypoint OKS standard deviations, shape (K,). No default
            values are provided; the caller must supply sigmas appropriate
            for their keypoint definition.

    Returns:
        PoseAPResult with AP, AR, and per-threshold breakdown.

    Raises:
        ValueError: If images is empty or sigmas has wrong shape.
    """
    if len(images) == 0:
        raise ValueError("images must not be empty")

    if sigmas.ndim != 1:
        raise ValueError(f"sigmas must be 1D, got shape {sigmas.shape}")
    if len(sigmas) == 0:
        raise ValueError("sigmas must not be empty")

    thresholds = THRESHOLDS

    logger.info(
        "Evaluating pose AP across %d images at %d thresholds", len(images), len(thresholds)
    )

    # Compute OKS matrices for all images once
    oks_matrices = [
        compute_oks_matrix(img.detections, img.ground_truths, sigmas) for img in images
    ]

    # Evaluate at each threshold
    per_threshold_results: list[PerThresholdResult] = []
    all_aps: list[float] = []
    all_ars: list[float] = []

    for thresh in thresholds:
        match_results: list[MatchResult] = []
        for img, oks_mat in zip(images, oks_matrices, strict=True):
            mr = greedy_match(oks_mat, img.detections, img.ground_truths, float(thresh))
            match_results.append(mr)

        ap, ar = compute_ap_ar(match_results)
        all_aps.append(ap)
        all_ars.append(ar)
        per_threshold_results.append(PerThresholdResult(threshold=float(thresh), ap=ap, ar=ar))

    # AP/AR at the standard 0.50 and 0.75 thresholds (indices 0 and 5)
    result = PoseAPResult(
        ap=float(np.mean(all_aps)),
        ap_50=per_threshold_results[0].ap,
        ap_75=per_threshold_results[5].ap,
        ar=float(np.mean(all_ars)),
        ar_50=per_threshold_results[0].ar,
        ar_75=per_threshold_results[5].ar,
        per_threshold=per_threshold_results,
    )

    logger.info("Pose evaluation complete: AP=%.4f, AR=%.4f", result.ap, result.ar)
    return result


def evaluate_detection(
    images: Sequence[ImageEvalData],
) -> DetectionAPResult:
    """Evaluate detection using bbox IoU-based COCO metrics.

    Computes AP and AR across the standard COCO IoU thresholds
    [0.50, 0.55, ..., 0.95] using 101-point precision-recall interpolation.

    Args:
        images: Sequence of per-image evaluation data.

    Returns:
        DetectionAPResult with AP, AR, and per-threshold breakdown.

    Raises:
        ValueError: If images is empty.
    """
    if len(images) == 0:
        raise ValueError("images must not be empty")

    thresholds = THRESHOLDS

    logger.info(
        "Evaluating detection AP across %d images at %d thresholds",
        len(images),
        len(thresholds),
    )

    # Compute IoU matrices for all images once
    iou_matrices = [compute_bbox_iou_matrix(img.detections, img.ground_truths) for img in images]

    # Evaluate at each threshold
    per_threshold_results: list[PerThresholdResult] = []
    all_aps: list[float] = []
    all_ars: list[float] = []

    for thresh in thresholds:
        match_results: list[MatchResult] = []
        for img, iou_mat in zip(images, iou_matrices, strict=True):
            mr = greedy_match(iou_mat, img.detections, img.ground_truths, float(thresh))
            match_results.append(mr)

        ap, ar = compute_ap_ar(match_results)
        all_aps.append(ap)
        all_ars.append(ar)
        per_threshold_results.append(PerThresholdResult(threshold=float(thresh), ap=ap, ar=ar))

    # AP/AR at the standard 0.50 and 0.75 thresholds (indices 0 and 5)
    result = DetectionAPResult(
        ap=float(np.mean(all_aps)),
        ap_50=per_threshold_results[0].ap,
        ap_75=per_threshold_results[5].ap,
        ar=float(np.mean(all_ars)),
        ar_50=per_threshold_results[0].ar,
        ar_75=per_threshold_results[5].ar,
        per_threshold=per_threshold_results,
    )

    logger.info("Detection evaluation complete: AP=%.4f, AR=%.4f", result.ap, result.ar)
    return result


def evaluate_pck(
    images: Sequence[ImageEvalData],
    sigmas: npt.NDArray[np.float64],
    threshold: float,
    exclude_keypoint_indices: Sequence[int] | None = None,
    match_oks_threshold: float = 0.5,
) -> PCKResult:
    """Evaluate Percentage of Correct Keypoints.

    First matches detections to ground truths using OKS at the given
    ``match_oks_threshold``, then computes PCK using bbox diagonal as the
    reference distance.

    Args:
        images: Sequence of per-image evaluation data.
        sigmas: Per-keypoint OKS standard deviations for matching, shape (K,).
        threshold: PCK distance threshold as fraction of bbox diagonal.
        exclude_keypoint_indices: Optional keypoint indices to exclude from
            PCK computation.
        match_oks_threshold: OKS threshold for matching detections to GTs.

    Returns:
        PCKResult with overall and per-keypoint breakdown.

    Raises:
        ValueError: If images is empty.
    """
    if len(images) == 0:
        raise ValueError("images must not be empty")

    if sigmas.ndim != 1:
        raise ValueError(f"sigmas must be 1D, got shape {sigmas.shape}")
    if len(sigmas) == 0:
        raise ValueError("sigmas must not be empty")

    logger.info("Evaluating PCK across %d images at threshold=%.2f", len(images), threshold)

    exclude_set = (
        frozenset(exclude_keypoint_indices) if exclude_keypoint_indices is not None else None
    )

    # Match and collect paired detections/GTs
    all_matched_dets = []
    all_matched_gts = []

    for img in images:
        if len(img.detections) == 0 or len(img.ground_truths) == 0:
            continue

        oks_matrix = compute_oks_matrix(img.detections, img.ground_truths, sigmas)
        match_result = greedy_match(
            oks_matrix, img.detections, img.ground_truths, match_oks_threshold
        )

        # Collect matched pairs
        for i, (tp, gt_idx) in enumerate(
            zip(match_result.tp_flags, match_result.gt_assignments, strict=True)
        ):
            if tp and gt_idx >= 0:
                det_idx = match_result.det_indices[i]
                gt = img.ground_truths[gt_idx]
                # Skip crowd GTs for PCK
                if not gt.is_crowd:
                    all_matched_dets.append(img.detections[det_idx])
                    all_matched_gts.append(gt)

    result = compute_pck(all_matched_dets, all_matched_gts, threshold, exclude_set)

    logger.info("PCK evaluation complete: PCK=%.4f", result.pck)
    return result
