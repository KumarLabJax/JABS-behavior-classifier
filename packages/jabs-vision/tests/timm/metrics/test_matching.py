"""Tests for COCO-style greedy matching."""

from __future__ import annotations

import numpy as np
import pytest

from jabs.vision.timm.metrics import (
    ImageEvalData,
    KeypointDetection,
    KeypointGroundTruth,
)
from jabs.vision.timm.metrics.matching import greedy_match, match_image


def _make_det(score: float, kps_offset: float = 0.0) -> KeypointDetection:
    """Create a detection with given score and keypoint offset."""
    kps = np.array(
        [[kps_offset, kps_offset], [kps_offset + 10, kps_offset + 10]], dtype=np.float64
    )
    bbox = np.array([kps_offset, kps_offset, kps_offset + 20, kps_offset + 20], dtype=np.float64)
    return KeypointDetection(keypoints=kps, score=score, bbox=bbox)


def _make_gt(kps_offset: float = 0.0, is_crowd: bool = False) -> KeypointGroundTruth:
    """Create a ground truth with given keypoint offset."""
    kps = np.array(
        [[kps_offset, kps_offset], [kps_offset + 10, kps_offset + 10]], dtype=np.float64
    )
    bbox = np.array([kps_offset, kps_offset, kps_offset + 20, kps_offset + 20], dtype=np.float64)
    return KeypointGroundTruth(keypoints=kps, bbox=bbox, is_crowd=is_crowd)


class TestGreedyMatch:
    """Tests for greedy_match."""

    def test_perfect_single_match(self) -> None:
        """Single detection perfectly matching single GT is a TP."""
        det = _make_det(0.9)
        gt = _make_gt()
        sim = np.array([[1.0]], dtype=np.float64)

        result = greedy_match(sim, [det], [gt], threshold=0.5)
        assert result.num_gt == 1
        assert len(result.tp_flags) == 1
        assert result.tp_flags[0]
        assert result.scores[0] == pytest.approx(0.9)

    def test_below_threshold_is_fp(self) -> None:
        """Detection below threshold is a false positive."""
        det = _make_det(0.9)
        gt = _make_gt(kps_offset=200.0)
        sim = np.array([[0.1]], dtype=np.float64)

        result = greedy_match(sim, [det], [gt], threshold=0.5)
        assert result.num_gt == 1
        assert not result.tp_flags[0]

    def test_score_ordering(self) -> None:
        """Higher-scored detection gets priority for matching."""
        det_high = _make_det(0.9)
        det_low = _make_det(0.3)
        gt = _make_gt()
        # Both detections match the GT
        sim = np.array([[0.95], [0.90]], dtype=np.float64)

        result = greedy_match(sim, [det_high, det_low], [gt], threshold=0.5)
        # First in sorted order (high score) is TP, second is FP
        assert result.tp_flags[0]  # det_high
        assert not result.tp_flags[1]  # det_low

    def test_two_dets_two_gts(self) -> None:
        """Two detections match two GTs correctly."""
        det1 = _make_det(0.9, kps_offset=0.0)
        det2 = _make_det(0.8, kps_offset=100.0)
        gt1 = _make_gt(kps_offset=0.0)
        gt2 = _make_gt(kps_offset=100.0)

        sim = np.array([[0.95, 0.01], [0.01, 0.95]], dtype=np.float64)
        result = greedy_match(sim, [det1, det2], [gt1, gt2], threshold=0.5)

        assert result.num_gt == 2
        assert np.sum(result.tp_flags) == 2

    def test_crowd_absorbs_detection(self) -> None:
        """Detection matched to crowd GT is ignored, not counted as TP or FP."""
        det = _make_det(0.9)
        crowd_gt = _make_gt(is_crowd=True)
        sim = np.array([[0.95]], dtype=np.float64)

        result = greedy_match(sim, [det], [crowd_gt], threshold=0.5)
        # Crowd GT does not count toward num_gt
        assert result.num_gt == 0
        # Detection absorbed by crowd - ignored in AP/AR computation
        assert not result.tp_flags[0]
        assert result.ignore_flags[0]

    def test_crowd_with_non_crowd(self) -> None:
        """Non-crowd GT is preferred over crowd GT."""
        det = _make_det(0.9)
        gt = _make_gt()
        crowd_gt = _make_gt(is_crowd=True)
        # Detection matches both equally
        sim = np.array([[0.95, 0.95]], dtype=np.float64)

        result = greedy_match(sim, [det], [gt, crowd_gt], threshold=0.5)
        assert result.num_gt == 1  # Only non-crowd counts
        assert result.tp_flags[0]
        # Should be assigned to non-crowd GT (index 0)
        assert result.gt_assignments[0] == 0

    def test_empty_detections(self) -> None:
        """No detections produces empty result."""
        gt = _make_gt()
        sim = np.zeros((0, 1), dtype=np.float64)

        result = greedy_match(sim, [], [gt], threshold=0.5)
        assert result.num_gt == 1
        assert len(result.tp_flags) == 0
        assert len(result.scores) == 0

    def test_empty_ground_truths(self) -> None:
        """No GTs means all detections are FP."""
        det = _make_det(0.9)
        sim = np.zeros((1, 0), dtype=np.float64)

        result = greedy_match(sim, [det], [], threshold=0.5)
        assert result.num_gt == 0
        assert not result.tp_flags[0]

    def test_gt_matched_only_once(self) -> None:
        """Each GT can only be matched once."""
        det1 = _make_det(0.9)
        det2 = _make_det(0.8)
        gt = _make_gt()
        # Both detections have high similarity to the single GT
        sim = np.array([[0.95], [0.90]], dtype=np.float64)

        result = greedy_match(sim, [det1, det2], [gt], threshold=0.5)
        assert result.num_gt == 1
        assert int(np.sum(result.tp_flags)) == 1  # Only one TP


class TestMatchImage:
    """Tests for match_image convenience wrapper."""

    def test_delegates_to_greedy_match(self) -> None:
        """match_image produces same result as greedy_match."""
        det = _make_det(0.9)
        gt = _make_gt()
        sim = np.array([[0.95]], dtype=np.float64)

        image = ImageEvalData(detections=[det], ground_truths=[gt])
        result = match_image(image, sim, threshold=0.5)

        assert result.num_gt == 1
        assert result.tp_flags[0]
