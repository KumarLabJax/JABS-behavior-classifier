"""Tests for PCK computation."""

from __future__ import annotations

import numpy as np
import pytest

from jabs.vision.timm.metrics import KeypointDetection, KeypointGroundTruth
from jabs.vision.timm.metrics.pck import bbox_diagonal, compute_pck


class TestBboxDiagonal:
    """Tests for bbox_diagonal."""

    def test_unit_square(self) -> None:
        """Unit square has diagonal of sqrt(2)."""
        bbox = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        assert bbox_diagonal(bbox) == pytest.approx(np.sqrt(2.0))

    def test_rectangle(self) -> None:
        """Rectangle diagonal is sqrt(w^2 + h^2)."""
        bbox = np.array([0.0, 0.0, 3.0, 4.0], dtype=np.float64)
        assert bbox_diagonal(bbox) == pytest.approx(5.0)

    def test_zero_area(self) -> None:
        """Zero-area box has zero diagonal."""
        bbox = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
        assert bbox_diagonal(bbox) == pytest.approx(0.0)

    def test_offset_box(self) -> None:
        """Offset box diagonal depends only on width and height."""
        bbox = np.array([100.0, 200.0, 103.0, 204.0], dtype=np.float64)
        assert bbox_diagonal(bbox) == pytest.approx(5.0)


class TestComputePCK:
    """Tests for compute_pck."""

    def test_perfect_predictions(self) -> None:
        """Identical keypoints produce PCK of 1.0."""
        kps = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        bbox = np.array([0.0, 0.0, 30.0, 40.0], dtype=np.float64)

        det = KeypointDetection(keypoints=kps, score=0.9, bbox=bbox)
        gt = KeypointGroundTruth(keypoints=kps, bbox=bbox)

        result = compute_pck([det], [gt], threshold=0.2)
        assert result.pck == pytest.approx(1.0)
        assert result.threshold == 0.2

    def test_all_incorrect(self) -> None:
        """Keypoints far away produce PCK of 0.0."""
        det_kps = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        gt_kps = np.array([[1000.0, 1000.0], [1000.0, 1000.0]], dtype=np.float64)
        bbox = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)

        det = KeypointDetection(keypoints=det_kps, score=0.9, bbox=bbox)
        gt = KeypointGroundTruth(keypoints=gt_kps, bbox=bbox)

        result = compute_pck([det], [gt], threshold=0.2)
        assert result.pck == pytest.approx(0.0)

    def test_per_keypoint_breakdown(self) -> None:
        """Per-keypoint details are computed correctly."""
        # Keypoint 0: exact match (correct), Keypoint 1: far off (incorrect)
        det_kps = np.array([[10.0, 20.0], [999.0, 999.0]], dtype=np.float64)
        gt_kps = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        bbox = np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float64)

        det = KeypointDetection(keypoints=det_kps, score=0.9, bbox=bbox)
        gt = KeypointGroundTruth(keypoints=gt_kps, bbox=bbox)

        result = compute_pck([det], [gt], threshold=0.2)
        assert result.per_keypoint[0].correct == 1
        assert result.per_keypoint[0].total == 1
        assert result.per_keypoint[0].pck == pytest.approx(1.0)
        assert result.per_keypoint[1].correct == 0
        assert result.per_keypoint[1].total == 1
        assert result.per_keypoint[1].pck == pytest.approx(0.0)
        assert result.pck == pytest.approx(0.5)

    def test_exclude_keypoints(self) -> None:
        """Excluded keypoints are not counted."""
        kps = np.array([[10.0, 20.0], [999.0, 999.0]], dtype=np.float64)
        gt_kps = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        bbox = np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float64)

        det = KeypointDetection(keypoints=kps, score=0.9, bbox=bbox)
        gt = KeypointGroundTruth(keypoints=gt_kps, bbox=bbox)

        # Exclude keypoint 1 (the bad one)
        result = compute_pck([det], [gt], threshold=0.2, exclude_keypoint_indices=frozenset({1}))
        assert result.pck == pytest.approx(1.0)
        assert 1 not in result.per_keypoint
        assert 1 in result.excluded_indices

    def test_visibility_mask(self) -> None:
        """Non-visible keypoints are not counted."""
        det_kps = np.array([[10.0, 20.0], [999.0, 999.0]], dtype=np.float64)
        gt_kps = np.array(
            [[10.0, 20.0, 2.0], [30.0, 40.0, 0.0]],  # kp1 not visible
            dtype=np.float64,
        )
        bbox = np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float64)

        det = KeypointDetection(keypoints=det_kps, score=0.9, bbox=bbox)
        gt = KeypointGroundTruth(keypoints=gt_kps, bbox=bbox)

        result = compute_pck([det], [gt], threshold=0.2)
        # Only keypoint 0 is visible and correct
        assert result.pck == pytest.approx(1.0)
        assert result.per_keypoint[0].total == 1
        assert result.per_keypoint[1].total == 0

    def test_bbox_diagonal_reference(self) -> None:
        """Threshold is relative to bbox diagonal."""
        # bbox is 30x40, diagonal = 50
        # threshold=0.1 means 5px tolerance
        bbox = np.array([0.0, 0.0, 30.0, 40.0], dtype=np.float64)

        # Keypoint offset by 4px (within 5px tolerance)
        det_close = KeypointDetection(
            keypoints=np.array([[4.0, 0.0]], dtype=np.float64),
            score=0.9,
            bbox=bbox,
        )
        # Keypoint offset by 6px (outside 5px tolerance)
        det_far = KeypointDetection(
            keypoints=np.array([[6.0, 0.0]], dtype=np.float64),
            score=0.9,
            bbox=bbox,
        )
        gt = KeypointGroundTruth(keypoints=np.array([[0.0, 0.0]], dtype=np.float64), bbox=bbox)

        result_close = compute_pck([det_close], [gt], threshold=0.1)
        result_far = compute_pck([det_far], [gt], threshold=0.1)

        assert result_close.pck == pytest.approx(1.0)
        assert result_far.pck == pytest.approx(0.0)

    def test_empty_inputs(self) -> None:
        """Empty matched pairs produce PCK of 0.0."""
        result = compute_pck([], [], threshold=0.2)
        assert result.pck == pytest.approx(0.0)
        assert result.per_keypoint == {}

    def test_mismatched_lengths(self) -> None:
        """Mismatched detection/GT lists raises ValueError."""
        det = KeypointDetection(
            keypoints=np.zeros((2, 2), dtype=np.float64),
            score=0.9,
            bbox=np.zeros(4, dtype=np.float64),
        )
        with pytest.raises(ValueError, match="same length"):
            compute_pck([det], [], threshold=0.2)

    def test_multiple_pairs(self) -> None:
        """PCK aggregated across multiple matched pairs."""
        bbox = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float64)
        gt_kps = np.array([[50.0, 50.0]], dtype=np.float64)
        gt = KeypointGroundTruth(keypoints=gt_kps, bbox=bbox)

        # Pair 1: correct
        det1 = KeypointDetection(
            keypoints=np.array([[50.0, 50.0]], dtype=np.float64),
            score=0.9,
            bbox=bbox,
        )
        # Pair 2: incorrect
        det2 = KeypointDetection(
            keypoints=np.array([[999.0, 999.0]], dtype=np.float64),
            score=0.8,
            bbox=bbox,
        )

        result = compute_pck([det1, det2], [gt, gt], threshold=0.2)
        assert result.pck == pytest.approx(0.5)
        assert result.per_keypoint[0].correct == 1
        assert result.per_keypoint[0].total == 2
