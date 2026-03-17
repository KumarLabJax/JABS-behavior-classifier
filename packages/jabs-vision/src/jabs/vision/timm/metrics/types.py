"""Public data models for neural network performance metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = [
    "DetectionAPResult",
    "ImageEvalData",
    "KeypointDetection",
    "KeypointGroundTruth",
    "KeypointPCKDetail",
    "PCKResult",
    "PerThresholdResult",
    "PoseAPResult",
]


@dataclass(frozen=True)
class KeypointDetection:
    """A single predicted keypoint detection.

    Attributes:
        keypoints: Predicted keypoint coordinates, shape (K, 2) or (K, 3) if visibility
            scores are included as a third column.
        score: Overall detection confidence score.
        bbox: Bounding box as [x1, y1, x2, y2], shape (4,).
    """

    keypoints: npt.NDArray[np.float64]
    score: float
    bbox: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate array shapes."""
        if self.keypoints.ndim != 2 or self.keypoints.shape[1] not in (2, 3):
            raise ValueError(
                f"keypoints must have shape (K, 2) or (K, 3), got {self.keypoints.shape}"
            )
        if self.bbox.shape != (4,):
            raise ValueError(f"bbox must have shape (4,), got {self.bbox.shape}")


@dataclass(frozen=True)
class KeypointGroundTruth:
    """A single ground-truth keypoint annotation.

    Attributes:
        keypoints: Ground-truth keypoint coordinates, shape (K, 2) or (K, 3) if
            visibility flags are included as a third column. Visibility follows COCO
            convention: 0=not labeled, 1=labeled but occluded, 2=labeled and visible.
        bbox: Bounding box as [x1, y1, x2, y2], shape (4,).
        area: Annotation area in pixels squared. If not provided, computed from bbox.
        is_crowd: Whether this is a crowd annotation. Crowd GTs absorb detections
            without incurring false positive penalty.
        num_keypoints: Number of visible keypoints. If not provided, computed from
            the visibility column (values > 0) if available, otherwise set to K.
    """

    keypoints: npt.NDArray[np.float64]
    bbox: npt.NDArray[np.float64]
    area: float | None = None
    is_crowd: bool = False
    num_keypoints: int | None = None

    def __post_init__(self) -> None:
        """Validate array shapes and compute defaults."""
        if self.keypoints.ndim != 2 or self.keypoints.shape[1] not in (2, 3):
            raise ValueError(
                f"keypoints must have shape (K, 2) or (K, 3), got {self.keypoints.shape}"
            )
        if self.bbox.shape != (4,):
            raise ValueError(f"bbox must have shape (4,), got {self.bbox.shape}")

        # Compute area from bbox if not provided
        if self.area is None:
            x1, y1, x2, y2 = self.bbox
            computed_area = float((x2 - x1) * (y2 - y1))
            object.__setattr__(self, "area", computed_area)

        # Compute num_keypoints if not provided
        if self.num_keypoints is None:
            if self.keypoints.shape[1] == 3:
                computed = int(np.sum(self.keypoints[:, 2] > 0))
            else:
                computed = self.keypoints.shape[0]
            object.__setattr__(self, "num_keypoints", computed)


@dataclass(frozen=True)
class ImageEvalData:
    """Evaluation data for a single image.

    Attributes:
        detections: List of predicted detections for this image.
        ground_truths: List of ground-truth annotations for this image.
    """

    detections: list[KeypointDetection]
    ground_truths: list[KeypointGroundTruth]

    def __post_init__(self) -> None:
        """Validate keypoint counts are consistent."""
        all_kp_counts: set[int] = set()
        for det in self.detections:
            all_kp_counts.add(det.keypoints.shape[0])
        for gt in self.ground_truths:
            all_kp_counts.add(gt.keypoints.shape[0])

        if len(all_kp_counts) > 1:
            raise ValueError(
                f"All detections and ground truths in an image must have the same number "
                f"of keypoints, got {all_kp_counts}"
            )


@dataclass(frozen=True)
class PerThresholdResult:
    """AP and AR at a single IoU/OKS threshold.

    Attributes:
        threshold: The IoU/OKS threshold value.
        ap: Average precision at this threshold.
        ar: Average recall at this threshold.
    """

    threshold: float
    ap: float
    ar: float


@dataclass(frozen=True)
class PoseAPResult:
    """Results from OKS-based pose evaluation.

    Attributes:
        ap: Mean AP across all thresholds.
        ap_50: AP at OKS threshold 0.50.
        ap_75: AP at OKS threshold 0.75.
        ar: Mean AR across all thresholds.
        ar_50: AR at OKS threshold 0.50.
        ar_75: AR at OKS threshold 0.75.
        per_threshold: Per-threshold AP and AR breakdown.
    """

    ap: float
    ap_50: float
    ap_75: float
    ar: float
    ar_50: float
    ar_75: float
    per_threshold: list[PerThresholdResult] = field(default_factory=list)


@dataclass(frozen=True)
class DetectionAPResult:
    """Results from bbox IoU-based detection evaluation.

    Attributes:
        ap: Mean AP across all thresholds.
        ap_50: AP at IoU threshold 0.50.
        ap_75: AP at IoU threshold 0.75.
        ar: Mean AR across all thresholds.
        ar_50: AR at IoU threshold 0.50.
        ar_75: AR at IoU threshold 0.75.
        per_threshold: Per-threshold AP and AR breakdown.
    """

    ap: float
    ap_50: float
    ap_75: float
    ar: float
    ar_50: float
    ar_75: float
    per_threshold: list[PerThresholdResult] = field(default_factory=list)


@dataclass(frozen=True)
class KeypointPCKDetail:
    """PCK detail for a single keypoint.

    Attributes:
        correct: Number of correct predictions within threshold.
        total: Total number of evaluated predictions.
        pck: PCK value (correct / total), or 0.0 if total is 0.
    """

    correct: int
    total: int
    pck: float


@dataclass(frozen=True)
class PCKResult:
    """Results from PCK (Percentage of Correct Keypoints) evaluation.

    Attributes:
        pck: Overall PCK across all keypoints.
        threshold: The distance threshold used (as fraction of bbox diagonal).
        per_keypoint: Per-keypoint PCK breakdown, keyed by keypoint index.
        excluded_indices: Set of keypoint indices that were excluded from evaluation.
    """

    pck: float
    threshold: float
    per_keypoint: dict[int, KeypointPCKDetail] = field(default_factory=dict)
    excluded_indices: frozenset[int] = field(default_factory=frozenset)
