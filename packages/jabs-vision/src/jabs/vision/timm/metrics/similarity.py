"""Similarity functions for keypoint and bounding box evaluation.

Provides OKS (Object Keypoint Similarity) for pose evaluation and
IoU (Intersection over Union) for bounding box evaluation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .types import KeypointDetection, KeypointGroundTruth

__all__ = [
    "compute_bbox_iou",
    "compute_bbox_iou_matrix",
    "compute_oks",
    "compute_oks_matrix",
]


def compute_oks(
    detection: KeypointDetection,
    ground_truth: KeypointGroundTruth,
    sigmas: npt.NDArray[np.float64],
) -> float:
    """Compute Object Keypoint Similarity between a detection and ground truth.

    Follows the COCO OKS formula:
        OKS = sum(exp(-d_i^2 / (2 * s_i^2 * area * 2)) * delta(v_i > 0)) /
              sum(delta(v_i > 0))

    where d_i is the Euclidean distance for keypoint i, s_i is the per-keypoint
    sigma, area is the GT annotation area, and v_i is the GT visibility flag.

    Args:
        detection: Predicted keypoint detection.
        ground_truth: Ground-truth keypoint annotation.
        sigmas: Per-keypoint standard deviations, shape (K,).

    Returns:
        OKS score in [0, 1]. Returns 0.0 if no visible keypoints in GT.
    """
    gt_kps = ground_truth.keypoints
    det_kps = detection.keypoints
    num_keypoints = gt_kps.shape[0]

    # Determine visibility mask
    visible = gt_kps[:, 2] > 0 if gt_kps.shape[1] == 3 else np.ones(num_keypoints, dtype=bool)

    num_visible = int(np.sum(visible))
    if num_visible == 0:
        return 0.0

    area = ground_truth.area
    if area is None or area <= 0:
        return 0.0

    # Compute squared distances for each keypoint
    dx = det_kps[:, 0] - gt_kps[:, 0]
    dy = det_kps[:, 1] - gt_kps[:, 1]
    d_squared = dx**2 + dy**2

    # OKS per keypoint: exp(-d^2 / (2 * sigma^2 * area * 2))
    # The factor of 2 * area accounts for the COCO convention
    variance = (sigmas**2) * 2 * area * 2
    oks_per_kp = np.exp(-d_squared / (variance + np.finfo(np.float64).eps))

    return float(np.sum(oks_per_kp[visible]) / num_visible)


def compute_oks_matrix(
    detections: list[KeypointDetection],
    ground_truths: list[KeypointGroundTruth],
    sigmas: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute OKS similarity matrix between all detections and ground truths.

    Args:
        detections: List of N predicted detections.
        ground_truths: List of M ground-truth annotations.
        sigmas: Per-keypoint standard deviations, shape (K,).

    Returns:
        OKS matrix of shape (N, M) where entry [i, j] is OKS between
        detection i and ground truth j.
    """
    n_det = len(detections)
    n_gt = len(ground_truths)

    if n_det == 0 or n_gt == 0:
        return np.zeros((n_det, n_gt), dtype=np.float64)

    matrix = np.zeros((n_det, n_gt), dtype=np.float64)
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truths):
            matrix[i, j] = compute_oks(det, gt, sigmas)

    return matrix


def compute_bbox_iou(
    bbox_a: npt.NDArray[np.float64],
    bbox_b: npt.NDArray[np.float64],
) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        bbox_a: First bounding box as [x1, y1, x2, y2], shape (4,).
        bbox_b: Second bounding box as [x1, y1, x2, y2], shape (4,).

    Returns:
        IoU score in [0, 1].
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def compute_bbox_iou_matrix(
    detections: list[KeypointDetection],
    ground_truths: list[KeypointGroundTruth],
) -> npt.NDArray[np.float64]:
    """Compute IoU matrix between detection and ground-truth bounding boxes.

    Args:
        detections: List of N predicted detections.
        ground_truths: List of M ground-truth annotations.

    Returns:
        IoU matrix of shape (N, M) where entry [i, j] is the IoU between
        detection i's bbox and ground truth j's bbox.
    """
    n_det = len(detections)
    n_gt = len(ground_truths)

    if n_det == 0 or n_gt == 0:
        return np.zeros((n_det, n_gt), dtype=np.float64)

    matrix = np.zeros((n_det, n_gt), dtype=np.float64)
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truths):
            matrix[i, j] = compute_bbox_iou(det.bbox, gt.bbox)

    return matrix
