"""Bout extraction and matching for comparing predicted behavior against ground truth.

A bout is a contiguous run of frames assigned the same class. Comparing a
classifier against densely labeled ground truth at the bout level asks a
different question than a frame-by-frame comparison: not "what fraction of
frames agree" but "was this occurrence of the behavior found at all", allowing
the predicted start and end frames to disagree with the labeled ones.

Two bouts are considered the same occurrence when they satisfy a
:class:`MatchCriterion`. Two are provided:

- :class:`OverlapCriterion` - the bouts share at least N frames (N=1 by default).
  Permissive; answers whether the classifier noticed the bout at all.
- :class:`IoUCriterion` - intersection-over-union of the two frame ranges meets a
  threshold. Penalizes a prediction that is much longer or shorter than the true
  bout, so sloppy boundaries cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..events import BehaviorEvents, ClassLabels


@dataclass(frozen=True, order=True)
class Bout:
    """A contiguous run of frames assigned the same class.

    Both endpoints are inclusive, matching the block representation JABS uses
    elsewhere for label blocks.

    Attributes:
        start: First frame of the bout, inclusive.
        end: Last frame of the bout, inclusive.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate that the bout spans at least one frame.

        Raises:
            ValueError: If ``end`` precedes ``start``.
        """
        if self.end < self.start:
            raise ValueError(f"bout end ({self.end}) precedes start ({self.start})")

    @property
    def duration(self) -> int:
        """Number of frames spanned by this bout."""
        return self.end - self.start + 1

    def overlap(self, other: Bout) -> int:
        """Number of frames shared with another bout.

        Args:
            other: Bout to intersect with.

        Returns:
            Count of shared frames; ``0`` when the bouts are disjoint.
        """
        return max(0, min(self.end, other.end) - max(self.start, other.start) + 1)

    def union(self, other: Bout) -> int:
        """Number of frames covered by either bout.

        Args:
            other: Bout to union with.

        Returns:
            Count of frames in the union of the two ranges.
        """
        return self.duration + other.duration - self.overlap(other)

    def iou(self, other: Bout) -> float:
        """Intersection-over-union of this bout's frame range with another's.

        Args:
            other: Bout to compare against.

        Returns:
            Value in ``[0.0, 1.0]``; ``1.0`` when the ranges are identical and
            ``0.0`` when they are disjoint.
        """
        union = self.union(other)
        return self.overlap(other) / union if union else 0.0


@dataclass(frozen=True)
class BoutOverlap:
    """An intersecting (ground truth, predicted) bout pair and its agreement.

    Attributes:
        truth_index: Index into the ground-truth bout list.
        predicted_index: Index into the predicted bout list.
        frames: Number of frames the two bouts share. Always at least 1.
        iou: Intersection-over-union of the two frame ranges.
    """

    truth_index: int
    predicted_index: int
    frames: int
    iou: float


@runtime_checkable
class MatchCriterion(Protocol):
    """Decides whether an overlapping bout pair counts as the same occurrence."""

    @property
    def label(self) -> str:
        """Short human-readable description, for report headings."""
        ...

    def matches(self, overlap: BoutOverlap) -> bool:
        """Return True if this overlap is close enough to count as a match.

        Args:
            overlap: A pair of bouts that share at least one frame.

        Returns:
            True if the pair should be treated as the same occurrence.
        """
        ...


@dataclass(frozen=True)
class OverlapCriterion:
    """Match bouts that share at least ``min_frames`` frames.

    Args:
        min_frames: Minimum number of shared frames. Must be positive - a
            threshold of zero would match every pair of bouts, including
            disjoint ones.
    """

    min_frames: int = 1

    def __post_init__(self) -> None:
        """Validate the frame threshold.

        Raises:
            ValueError: If ``min_frames`` is not positive.
        """
        if self.min_frames < 1:
            raise ValueError(f"min_frames must be positive, got {self.min_frames}")

    @property
    def label(self) -> str:
        """Short human-readable description, for report headings."""
        if self.min_frames == 1:
            return "overlap >= 1 frame"
        return f"overlap >= {self.min_frames} frames"

    def matches(self, overlap: BoutOverlap) -> bool:
        """Return True if the pair shares at least ``min_frames`` frames.

        Args:
            overlap: A pair of bouts that share at least one frame.

        Returns:
            True if the shared frame count meets the threshold.
        """
        return overlap.frames >= self.min_frames


@dataclass(frozen=True)
class IoUCriterion:
    """Match bouts whose intersection-over-union meets a threshold.

    Args:
        threshold: Minimum IoU, in ``(0.0, 1.0]``. A threshold of zero is
            rejected because every overlapping pair would match, making this
            criterion indistinguishable from ``OverlapCriterion(1)`` while
            claiming to measure boundary agreement.
    """

    threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate the IoU threshold.

        Raises:
            ValueError: If ``threshold`` is outside ``(0.0, 1.0]``.
        """
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"IoU threshold must be in (0.0, 1.0], got {self.threshold}")

    @property
    def label(self) -> str:
        """Short human-readable description, for report headings."""
        return f"IoU >= {self.threshold:g}"

    def matches(self, overlap: BoutOverlap) -> bool:
        """Return True if the pair's IoU meets the threshold.

        Args:
            overlap: A pair of bouts that share at least one frame.

        Returns:
            True if the IoU meets the threshold.
        """
        return overlap.iou >= self.threshold


@dataclass(frozen=True)
class BoutMatchResult:
    """Which bouts matched which, under a single criterion.

    Attributes:
        criterion_label: The criterion's ``label``, carried for reporting.
        truth_matches: For each ground-truth bout, the indices of the predicted
            bouts it matched. Parallel to the ground-truth bout list.
        predicted_matches: For each predicted bout, the indices of the
            ground-truth bouts it matched. Parallel to the predicted bout list.
    """

    criterion_label: str
    truth_matches: tuple[tuple[int, ...], ...]
    predicted_matches: tuple[tuple[int, ...], ...]


def extract_bouts(
    vector: npt.NDArray[np.integer],
    value: int = ClassLabels.BEHAVIOR,
) -> list[Bout]:
    """Extract maximal runs of ``value`` from a per-frame class vector.

    Args:
        vector: Per-frame class assignments, e.g. a :class:`ClassLabels` vector
            of ground-truth labels or predicted classes.
        value: Class value whose runs should be returned. Defaults to
            ``ClassLabels.BEHAVIOR``.

    Returns:
        Bouts in ascending frame order. Empty when the value never occurs.
    """
    events = BehaviorEvents.from_vector(np.asarray(vector))
    return [
        Bout(int(start), int(start) + int(duration) - 1)
        for start, duration, state in zip(
            events.starts, events.durations, events.states, strict=True
        )
        if state == value
    ]


def overlapping_pairs(truth: list[Bout], predicted: list[Bout]) -> list[BoutOverlap]:
    """Find every pair of bouts that shares at least one frame.

    Both inputs are assumed sorted by start frame and internally disjoint, which
    is what :func:`extract_bouts` produces, so a single linear sweep finds all
    intersecting pairs rather than comparing every pair against every other.

    Args:
        truth: Ground-truth bouts, ascending and non-overlapping.
        predicted: Predicted bouts, ascending and non-overlapping.

    Returns:
        Every intersecting pair, with its shared frame count and IoU.
    """
    pairs: list[BoutOverlap] = []
    i = j = 0
    while i < len(truth) and j < len(predicted):
        t, p = truth[i], predicted[j]
        shared = t.overlap(p)
        if shared > 0:
            pairs.append(BoutOverlap(i, j, shared, t.iou(p)))
        # advance past whichever bout ends first: it cannot intersect anything
        # further along the other list
        if t.end < p.end:
            i += 1
        else:
            j += 1
    return pairs


def match_bouts(
    truth: list[Bout],
    predicted: list[Bout],
    criterion: MatchCriterion,
    overlaps: list[BoutOverlap] | None = None,
) -> BoutMatchResult:
    """Match predicted bouts against ground-truth bouts under a criterion.

    Matching is many-to-many: a ground-truth bout is matched if any predicted
    bout satisfies the criterion against it, and vice versa. A classifier that
    splits one true bout into several therefore still counts that bout as
    detected, and each of the fragments as a hit; the splitting shows up in the
    fragmentation counts derived from this result rather than as a penalty here.

    Args:
        truth: Ground-truth bouts, ascending and non-overlapping.
        predicted: Predicted bouts, ascending and non-overlapping.
        criterion: Rule deciding whether an overlapping pair is the same
            occurrence.
        overlaps: Precomputed result of :func:`overlapping_pairs` for the same
            two lists. Pass this when matching the same bouts under several
            criteria so the sweep runs once.

    Returns:
        The per-bout match lists.
    """
    if overlaps is None:
        overlaps = overlapping_pairs(truth, predicted)

    truth_matches: list[list[int]] = [[] for _ in truth]
    predicted_matches: list[list[int]] = [[] for _ in predicted]
    for overlap in overlaps:
        if criterion.matches(overlap):
            truth_matches[overlap.truth_index].append(overlap.predicted_index)
            predicted_matches[overlap.predicted_index].append(overlap.truth_index)

    return BoutMatchResult(
        criterion_label=criterion.label,
        truth_matches=tuple(tuple(m) for m in truth_matches),
        predicted_matches=tuple(tuple(m) for m in predicted_matches),
    )
