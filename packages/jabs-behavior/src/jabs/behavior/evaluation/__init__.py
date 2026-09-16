"""jabs.behavior.evaluation package.

Compares predicted behavior against ground-truth labels, frame by frame and
bout by bout.
"""

from .bouts import (
    Bout,
    BoutMatchResult,
    BoutOverlap,
    IoUCriterion,
    MatchCriterion,
    OverlapCriterion,
    extract_bouts,
    match_bouts,
    overlapping_pairs,
)
from .metrics import (
    BoutMetrics,
    FrameMetrics,
    bouts_are_evaluable,
    compute_bout_metrics,
    compute_frame_metrics,
)

__all__ = [
    "Bout",
    "BoutMatchResult",
    "BoutMetrics",
    "BoutOverlap",
    "FrameMetrics",
    "IoUCriterion",
    "MatchCriterion",
    "OverlapCriterion",
    "bouts_are_evaluable",
    "compute_bout_metrics",
    "compute_frame_metrics",
    "extract_bouts",
    "match_bouts",
    "overlapping_pairs",
]
