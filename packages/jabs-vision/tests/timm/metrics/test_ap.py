"""Tests for AP and AR computation."""

from __future__ import annotations

import numpy as np
import pytest

from jabs.vision.timm.metrics.matching import MatchResult
from jabs.vision.timm.metrics.ranking import compute_ap_ar


def _make_match_result(
    tp_flags: list[bool],
    scores: list[float],
    num_gt: int,
) -> MatchResult:
    """Create a MatchResult from simple lists."""
    n = len(tp_flags)
    return MatchResult(
        tp_flags=np.array(tp_flags, dtype=np.bool_),
        scores=np.array(scores, dtype=np.float64),
        num_gt=num_gt,
        det_indices=np.arange(n, dtype=np.intp),
        gt_assignments=np.full(n, -1, dtype=np.intp),
    )


class TestComputeAPAR:
    """Tests for compute_ap_ar."""

    def test_perfect_detection(self) -> None:
        """All TPs with one GT gives AP=1.0 and AR=1.0."""
        mr = _make_match_result(tp_flags=[True], scores=[0.9], num_gt=1)
        ap, ar = compute_ap_ar([mr])
        assert ap == pytest.approx(1.0)
        assert ar == pytest.approx(1.0)

    def test_all_false_positives(self) -> None:
        """All FPs gives AP=0.0."""
        mr = _make_match_result(tp_flags=[False, False], scores=[0.9, 0.8], num_gt=1)
        ap, ar = compute_ap_ar([mr])
        assert ap == pytest.approx(0.0)
        assert ar == pytest.approx(0.0)

    def test_no_ground_truths(self) -> None:
        """No GTs returns AP=0, AR=0."""
        mr = _make_match_result(tp_flags=[False], scores=[0.9], num_gt=0)
        ap, ar = compute_ap_ar([mr])
        assert ap == pytest.approx(0.0)
        assert ar == pytest.approx(0.0)

    def test_no_detections(self) -> None:
        """No detections with GTs returns AP=0, AR=0."""
        mr = _make_match_result(tp_flags=[], scores=[], num_gt=5)
        ap, ar = compute_ap_ar([mr])
        assert ap == pytest.approx(0.0)
        assert ar == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        """One TP, one missed GT: AR=0.5."""
        mr = _make_match_result(tp_flags=[True], scores=[0.9], num_gt=2)
        ap, ar = compute_ap_ar([mr])
        assert ar == pytest.approx(0.5)
        # AP should be > 0 since we have a TP
        assert ap > 0.0

    def test_multi_image_aggregation(self) -> None:
        """Results from multiple images are aggregated correctly."""
        mr1 = _make_match_result(tp_flags=[True], scores=[0.9], num_gt=1)
        mr2 = _make_match_result(tp_flags=[True], scores=[0.8], num_gt=1)
        ap, ar = compute_ap_ar([mr1, mr2])
        assert ap == pytest.approx(1.0)
        assert ar == pytest.approx(1.0)

    def test_mixed_tp_fp_ordering(self) -> None:
        """Higher-scored TP followed by FP produces reasonable AP."""
        # TP at high score, FP at low score, 1 GT
        mr = _make_match_result(tp_flags=[True, False], scores=[0.9, 0.3], num_gt=1)
        ap, ar = compute_ap_ar([mr])
        assert ar == pytest.approx(1.0)
        # Perfect recall, but precision degrades. AP should still be 1.0
        # because the TP comes first and COCO interpolation takes max precision
        # at each recall level.
        assert ap == pytest.approx(1.0)

    def test_fp_before_tp(self) -> None:
        """FP at high score followed by TP reduces AP below 1.0."""
        mr = _make_match_result(tp_flags=[False, True], scores=[0.9, 0.8], num_gt=1)
        ap, ar = compute_ap_ar([mr])
        assert ar == pytest.approx(1.0)
        # Precision at recall=1.0 is 1/2, interpolated AP should be less than 1.0
        assert ap < 1.0

    def test_multiple_images_partial(self) -> None:
        """Multiple images with partial detections."""
        # Image 1: TP
        mr1 = _make_match_result(tp_flags=[True], scores=[0.9], num_gt=1)
        # Image 2: FP
        mr2 = _make_match_result(tp_flags=[False], scores=[0.5], num_gt=1)

        ap, ar = compute_ap_ar([mr1, mr2])
        assert ar == pytest.approx(0.5)  # 1 TP / 2 GT total
        assert 0.0 < ap < 1.0

    def test_empty_match_results_list(self) -> None:
        """Empty list of match results returns zeros."""
        ap, ar = compute_ap_ar([])
        assert ap == pytest.approx(0.0)
        assert ar == pytest.approx(0.0)
