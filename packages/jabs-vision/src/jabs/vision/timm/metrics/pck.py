"""Percentage of Correct Keypoints (PCK) utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .types import KeypointDetection, KeypointGroundTruth, KeypointPCKDetail, PCKResult

__all__ = ["bbox_diagonal", "compute_pck"]


def bbox_diagonal(bbox: npt.NDArray[np.float64]) -> float:
    """Compute the diagonal length of a bounding box.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2], shape (4,).

    Returns:
        Diagonal length: sqrt((x2-x1)^2 + (y2-y1)^2).
    """
    dx = bbox[2] - bbox[0]
    dy = bbox[3] - bbox[1]
    return float(np.sqrt(dx**2 + dy**2))


def compute_pck(
    matched_detections: list[KeypointDetection],
    matched_ground_truths: list[KeypointGroundTruth],
    threshold: float,
    exclude_keypoint_indices: frozenset[int] | None = None,
) -> PCKResult:
    """Compute PCK from matched detection-ground truth pairs.

    For each matched pair, a keypoint is correct if the Euclidean distance
    between predicted and ground-truth positions is less than
    ``threshold * bbox_diagonal``, where bbox_diagonal is the diagonal of
    the ground-truth bounding box.

    Args:
        matched_detections: Predictions that were successfully matched.
        matched_ground_truths: Corresponding ground-truth annotations (same length).
        threshold: Distance threshold as a fraction of the GT bbox diagonal.
        exclude_keypoint_indices: Optional set of keypoint indices to exclude
            from evaluation.

    Returns:
        PCKResult with overall and per-keypoint breakdown.

    Raises:
        ValueError: If detection and ground truth lists have different lengths.
    """
    if len(matched_detections) != len(matched_ground_truths):
        raise ValueError(
            f"Detection and ground truth lists must have the same length, "
            f"got {len(matched_detections)} and {len(matched_ground_truths)}"
        )

    if exclude_keypoint_indices is None:
        exclude_keypoint_indices = frozenset()

    if len(matched_detections) == 0:
        return PCKResult(
            pck=0.0,
            threshold=threshold,
            per_keypoint={},
            excluded_indices=exclude_keypoint_indices,
        )

    num_keypoints = matched_detections[0].keypoints.shape[0]

    # Per-keypoint accumulators
    correct_per_kp = np.zeros(num_keypoints, dtype=np.int64)
    total_per_kp = np.zeros(num_keypoints, dtype=np.int64)

    for det, gt in zip(matched_detections, matched_ground_truths, strict=True):
        ref_dist = bbox_diagonal(gt.bbox) * threshold
        if ref_dist <= 0:
            continue

        gt_kps = gt.keypoints
        det_kps = det.keypoints

        # Determine visibility
        visible = gt_kps[:, 2] > 0 if gt_kps.shape[1] == 3 else np.ones(num_keypoints, dtype=bool)

        # Compute per-keypoint distances
        dx = det_kps[:, 0] - gt_kps[:, 0]
        dy = det_kps[:, 1] - gt_kps[:, 1]
        distances = np.sqrt(dx**2 + dy**2)

        for k in range(num_keypoints):
            if k in exclude_keypoint_indices:
                continue
            if not visible[k]:
                continue

            total_per_kp[k] += 1
            if distances[k] < ref_dist:
                correct_per_kp[k] += 1

    # Build per-keypoint details
    per_keypoint: dict[int, KeypointPCKDetail] = {}
    total_correct = 0
    total_evaluated = 0

    for k in range(num_keypoints):
        if k in exclude_keypoint_indices:
            continue
        c = int(correct_per_kp[k])
        t = int(total_per_kp[k])
        pck_k = c / t if t > 0 else 0.0
        per_keypoint[k] = KeypointPCKDetail(correct=c, total=t, pck=pck_k)
        total_correct += c
        total_evaluated += t

    overall_pck = total_correct / total_evaluated if total_evaluated > 0 else 0.0

    return PCKResult(
        pck=overall_pck,
        threshold=threshold,
        per_keypoint=per_keypoint,
        excluded_indices=exclude_keypoint_indices,
    )
