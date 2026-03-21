"""Developer-facing SDK for neural network performance metrics.

Provides typed evaluation models plus public APIs for similarity scoring,
matching, ranking, PCK evaluation, and result reporting.
"""

from .evaluation import evaluate_detection, evaluate_pck, evaluate_pose
from .matching import MatchResult, greedy_match, match_image
from .pck import compute_pck
from .reporting import format_results
from .similarity import (
    compute_bbox_iou,
    compute_bbox_iou_matrix,
    compute_oks,
    compute_oks_matrix,
)
from .types import (
    DetectionAPResult,
    ImageEvalData,
    KeypointDetection,
    KeypointGroundTruth,
    KeypointPCKDetail,
    PCKResult,
    PerThresholdResult,
    PoseAPResult,
)

__all__ = [
    "DetectionAPResult",
    "ImageEvalData",
    "KeypointDetection",
    "KeypointGroundTruth",
    "KeypointPCKDetail",
    "MatchResult",
    "PCKResult",
    "PerThresholdResult",
    "PoseAPResult",
    "compute_bbox_iou",
    "compute_bbox_iou_matrix",
    "compute_oks",
    "compute_oks_matrix",
    "compute_pck",
    "evaluate_detection",
    "evaluate_pck",
    "evaluate_pose",
    "format_results",
    "greedy_match",
    "match_image",
]
