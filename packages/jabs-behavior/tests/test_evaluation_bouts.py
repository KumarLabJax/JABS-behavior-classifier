"""Tests for bout extraction and matching."""

import numpy as np
import pytest

from jabs.behavior.evaluation import (
    Bout,
    IoUCriterion,
    OverlapCriterion,
    extract_bouts,
    match_bouts,
    overlapping_pairs,
)
from jabs.behavior.events import ClassLabels

# -----------------------------------------------------------------------------
# Bout
# -----------------------------------------------------------------------------


def test_bout_duration_is_inclusive_of_both_endpoints() -> None:
    """Bout duration is inclusive of both endpoints."""
    assert Bout(10, 10).duration == 1
    assert Bout(10, 14).duration == 5


def test_bout_rejects_inverted_range() -> None:
    """Bout rejects inverted range."""
    with pytest.raises(ValueError, match="precedes start"):
        Bout(10, 9)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (Bout(0, 9), Bout(5, 14), 5),
        (Bout(0, 9), Bout(10, 14), 0),
        (Bout(0, 9), Bout(0, 9), 10),
        (Bout(0, 9), Bout(3, 5), 3),
        (Bout(5, 14), Bout(0, 9), 5),
    ],
    ids=["partial", "adjacent-disjoint", "identical", "contained", "reversed-args"],
)
def test_bout_overlap(a: Bout, b: Bout, expected: int) -> None:
    """Bout overlap."""
    assert a.overlap(b) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (Bout(0, 9), Bout(0, 9), 1.0),
        (Bout(0, 9), Bout(10, 19), 0.0),
        # 5 shared frames out of 15 covered
        (Bout(0, 9), Bout(5, 14), 1 / 3),
        # a 2-frame prediction inside an 8-frame bout
        (Bout(4, 11), Bout(7, 8), 0.25),
    ],
    ids=["identical", "disjoint", "half-shifted", "short-prediction"],
)
def test_bout_iou(a: Bout, b: Bout, expected: float) -> None:
    """Bout iou."""
    assert a.iou(b) == pytest.approx(expected)


def test_bout_iou_is_symmetric() -> None:
    """Bout iou is symmetric."""
    a, b = Bout(3, 20), Bout(15, 40)
    assert a.iou(b) == pytest.approx(b.iou(a))


# -----------------------------------------------------------------------------
# extract_bouts
# -----------------------------------------------------------------------------


def test_extract_bouts_finds_behavior_runs() -> None:
    """Extract bouts finds behavior runs."""
    #      0  1  2  3  4  5  6  7
    v = [0, 1, 1, 0, 0, 1, 0, 0]
    assert extract_bouts(np.array(v)) == [Bout(1, 2), Bout(5, 5)]


def test_extract_bouts_handles_runs_at_both_edges() -> None:
    """Extract bouts handles runs at both edges."""
    v = [1, 1, 0, 0, 1, 1]
    assert extract_bouts(np.array(v)) == [Bout(0, 1), Bout(4, 5)]


def test_extract_bouts_returns_empty_when_behavior_absent() -> None:
    """Extract bouts returns empty when behavior absent."""
    assert extract_bouts(np.zeros(20, dtype=np.int8)) == []


def test_extract_bouts_on_empty_vector() -> None:
    """Extract bouts on empty vector."""
    assert extract_bouts(np.array([], dtype=np.int8)) == []


def test_extract_bouts_ignores_unlabeled_frames() -> None:
    """NONE frames break a run rather than joining it to the next."""
    v = [1, 1, ClassLabels.NONE, ClassLabels.NONE, 1]
    assert extract_bouts(np.array(v)) == [Bout(0, 1), Bout(4, 4)]


def test_extract_bouts_can_select_another_class() -> None:
    """Extract bouts can select another class."""
    v = [0, 1, 1, 0, 0]
    assert extract_bouts(np.array(v), value=ClassLabels.NOT_BEHAVIOR) == [Bout(0, 0), Bout(3, 4)]


# -----------------------------------------------------------------------------
# overlapping_pairs
# -----------------------------------------------------------------------------


def test_overlapping_pairs_finds_every_intersection() -> None:
    """Overlapping pairs finds every intersection."""
    truth = [Bout(0, 9), Bout(20, 29)]
    predicted = [Bout(5, 24), Bout(26, 40)]
    pairs = {(p.truth_index, p.predicted_index) for p in overlapping_pairs(truth, predicted)}
    assert pairs == {(0, 0), (1, 0), (1, 1)}


def test_overlapping_pairs_reports_frames_and_iou() -> None:
    """Overlapping pairs reports frames and iou."""
    (pair,) = overlapping_pairs([Bout(0, 9)], [Bout(5, 14)])
    assert pair.frames == 5
    assert pair.iou == pytest.approx(1 / 3)


def test_overlapping_pairs_empty_when_disjoint() -> None:
    """Overlapping pairs empty when disjoint."""
    assert overlapping_pairs([Bout(0, 9)], [Bout(10, 19)]) == []


def test_overlapping_pairs_handles_empty_sides() -> None:
    """Overlapping pairs handles empty sides."""
    assert overlapping_pairs([], [Bout(0, 9)]) == []
    assert overlapping_pairs([Bout(0, 9)], []) == []


def test_overlapping_pairs_does_not_skip_a_pair_after_a_match() -> None:
    """A single long prediction spanning several true bouts must yield every pair.

    This is the case a naive two-pointer sweep drops: after emitting (0, 0) it
    advances both sides and never sees (1, 0) or (2, 0).
    """
    truth = [Bout(0, 4), Bout(10, 14), Bout(20, 24)]
    predicted = [Bout(0, 24)]
    pairs = {(p.truth_index, p.predicted_index) for p in overlapping_pairs(truth, predicted)}
    assert pairs == {(0, 0), (1, 0), (2, 0)}


def test_overlapping_pairs_scales_to_many_bouts() -> None:
    """Interleaved bouts, to exercise the sweep's advance logic at length."""
    truth = [Bout(i * 10, i * 10 + 4) for i in range(50)]
    predicted = [Bout(i * 10 + 2, i * 10 + 7) for i in range(50)]
    pairs = overlapping_pairs(truth, predicted)
    assert len(pairs) == 50
    assert all(p.truth_index == p.predicted_index for p in pairs)
    assert all(p.frames == 3 for p in pairs)


# -----------------------------------------------------------------------------
# criteria
# -----------------------------------------------------------------------------


def test_overlap_criterion_rejects_non_positive_threshold() -> None:
    """Overlap criterion rejects non positive threshold."""
    with pytest.raises(ValueError, match="must be positive"):
        OverlapCriterion(min_frames=0)


def test_iou_criterion_rejects_out_of_range_threshold() -> None:
    """Iou criterion rejects out of range threshold."""
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"must be in \(0.0, 1.0\]"):
            IoUCriterion(threshold=bad)


@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        (OverlapCriterion(1), "overlap >= 1 frame"),
        (OverlapCriterion(5), "overlap >= 5 frames"),
        (IoUCriterion(0.5), "IoU >= 0.5"),
        (IoUCriterion(0.25), "IoU >= 0.25"),
    ],
    ids=["one-frame", "n-frames", "iou-half", "iou-quarter"],
)
def test_criterion_labels(criterion, expected: str) -> None:
    """Criterion labels."""
    assert criterion.label == expected


# -----------------------------------------------------------------------------
# match_bouts
# -----------------------------------------------------------------------------


def test_match_bouts_one_to_one() -> None:
    """Match bouts one to one."""
    result = match_bouts([Bout(0, 9)], [Bout(1, 8)], OverlapCriterion(1))
    assert result.truth_matches == ((0,),)
    assert result.predicted_matches == ((0,),)


def test_match_bouts_reports_no_match_when_disjoint() -> None:
    """Match bouts reports no match when disjoint."""
    result = match_bouts([Bout(0, 9)], [Bout(20, 29)], OverlapCriterion(1))
    assert result.truth_matches == ((),)
    assert result.predicted_matches == ((),)


def test_match_bouts_is_many_to_many_for_a_fragmented_bout() -> None:
    """One true bout split into two predictions: both count, and the truth is detected."""
    result = match_bouts([Bout(0, 20)], [Bout(0, 5), Bout(10, 20)], OverlapCriterion(1))
    assert result.truth_matches == ((0, 1),)
    assert result.predicted_matches == ((0,), (0,))


def test_match_bouts_is_many_to_many_for_a_merged_prediction() -> None:
    """One prediction spanning two true bouts matches both."""
    result = match_bouts([Bout(0, 5), Bout(10, 20)], [Bout(0, 20)], OverlapCriterion(1))
    assert result.truth_matches == ((0,), (0,))
    assert result.predicted_matches == ((0, 1),)


def test_iou_criterion_rejects_a_sloppy_match_that_overlap_accepts() -> None:
    """The distinction the two criteria exist to draw."""
    truth = [Bout(4, 11)]
    predicted = [Bout(7, 8)]  # 2 frames inside an 8-frame bout -> IoU 0.25

    assert match_bouts(truth, predicted, OverlapCriterion(1)).truth_matches == ((0,),)
    assert match_bouts(truth, predicted, IoUCriterion(0.5)).truth_matches == ((),)


def test_min_overlap_threshold_filters_brief_intersections() -> None:
    """Min overlap threshold filters brief intersections."""
    truth = [Bout(0, 9)]
    predicted = [Bout(9, 20)]  # shares exactly 1 frame

    assert match_bouts(truth, predicted, OverlapCriterion(1)).truth_matches == ((0,),)
    assert match_bouts(truth, predicted, OverlapCriterion(2)).truth_matches == ((),)


def test_match_bouts_accepts_precomputed_overlaps() -> None:
    """Match bouts accepts precomputed overlaps."""
    truth = [Bout(0, 9), Bout(20, 29)]
    predicted = [Bout(5, 24)]
    overlaps = overlapping_pairs(truth, predicted)

    with_precomputed = match_bouts(truth, predicted, OverlapCriterion(1), overlaps=overlaps)
    without = match_bouts(truth, predicted, OverlapCriterion(1))
    assert with_precomputed == without


def test_match_bouts_carries_the_criterion_label() -> None:
    """Match bouts carries the criterion label."""
    result = match_bouts([Bout(0, 9)], [Bout(0, 9)], IoUCriterion(0.75))
    assert result.criterion_label == "IoU >= 0.75"
