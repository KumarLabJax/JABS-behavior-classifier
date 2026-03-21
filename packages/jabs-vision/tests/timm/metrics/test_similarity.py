"""Tests for OKS and IoU similarity functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from jabs.vision.timm.metrics import KeypointDetection, KeypointGroundTruth
from jabs.vision.timm.metrics.similarity import (
    compute_bbox_iou,
    compute_bbox_iou_matrix,
    compute_oks,
    compute_oks_matrix,
)


class TestComputeOKS:
    """Tests for compute_oks."""

    def test_perfect_match(
        self,
        perfect_detection: KeypointDetection,
        matching_ground_truth: KeypointGroundTruth,
        sigmas: npt.NDArray[np.float64],
    ) -> None:
        """Identical keypoints produce OKS of 1.0."""
        oks = compute_oks(perfect_detection, matching_ground_truth, sigmas)
        assert oks == pytest.approx(1.0)

    def test_offset_keypoints(
        self,
        offset_detection: KeypointDetection,
        matching_ground_truth: KeypointGroundTruth,
        sigmas: npt.NDArray[np.float64],
    ) -> None:
        """Offset keypoints produce OKS less than 1.0."""
        oks = compute_oks(offset_detection, matching_ground_truth, sigmas)
        assert 0.0 < oks < 1.0

    def test_large_distance_low_oks(
        self,
        perfect_detection: KeypointDetection,
        distant_ground_truth: KeypointGroundTruth,
        sigmas: npt.NDArray[np.float64],
    ) -> None:
        """Large keypoint distances produce OKS near 0."""
        oks = compute_oks(perfect_detection, distant_ground_truth, sigmas)
        assert oks < 0.01

    def test_no_visible_keypoints(self, sigmas: npt.NDArray[np.float64]) -> None:
        """OKS is 0.0 when no GT keypoints are visible."""
        det = KeypointDetection(
            keypoints=np.zeros((5, 2), dtype=np.float64),
            score=0.9,
            bbox=np.zeros(4, dtype=np.float64),
        )
        gt_kps = np.zeros((5, 3), dtype=np.float64)  # All visibility = 0
        gt = KeypointGroundTruth(
            keypoints=gt_kps, bbox=np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float64)
        )
        oks = compute_oks(det, gt, sigmas)
        assert oks == 0.0

    def test_sigma_scaling(self) -> None:
        """Larger sigmas produce higher OKS for same offset."""
        det_kps = np.array([[10.0, 10.0]], dtype=np.float64)
        gt_kps = np.array([[15.0, 15.0]], dtype=np.float64)
        bbox = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float64)

        det = KeypointDetection(keypoints=det_kps, score=0.9, bbox=bbox)
        gt = KeypointGroundTruth(keypoints=gt_kps, bbox=bbox)

        small_sigma = np.array([0.01], dtype=np.float64)
        large_sigma = np.array([0.10], dtype=np.float64)

        oks_small = compute_oks(det, gt, small_sigma)
        oks_large = compute_oks(det, gt, large_sigma)
        assert oks_large > oks_small

    def test_area_scaling(self) -> None:
        """Larger GT area produces higher OKS for same offset."""
        det_kps = np.array([[10.0, 10.0]], dtype=np.float64)
        gt_kps = np.array([[15.0, 15.0]], dtype=np.float64)
        sigma = np.array([0.05], dtype=np.float64)

        small_bbox = np.array([0.0, 0.0, 20.0, 20.0], dtype=np.float64)
        large_bbox = np.array([0.0, 0.0, 200.0, 200.0], dtype=np.float64)

        det = KeypointDetection(keypoints=det_kps, score=0.9, bbox=small_bbox)
        gt_small = KeypointGroundTruth(keypoints=gt_kps, bbox=small_bbox)
        gt_large = KeypointGroundTruth(keypoints=gt_kps, bbox=large_bbox)

        oks_small_area = compute_oks(det, gt_small, sigma)
        oks_large_area = compute_oks(det, gt_large, sigma)
        assert oks_large_area > oks_small_area

    def test_visibility_mask(self, sigmas: npt.NDArray[np.float64]) -> None:
        """Only visible keypoints contribute to OKS."""
        # Detection matches on visible kps but is far on invisible ones
        det_kps = np.array(
            [[10.0, 20.0], [30.0, 40.0], [999.0, 999.0], [70.0, 80.0], [90.0, 100.0]],
            dtype=np.float64,
        )
        gt_kps = np.array(
            [
                [10.0, 20.0, 2.0],
                [30.0, 40.0, 2.0],
                [50.0, 60.0, 0.0],  # Not visible - won't count
                [70.0, 80.0, 2.0],
                [90.0, 100.0, 2.0],
            ],
            dtype=np.float64,
        )
        bbox = np.array([5.0, 15.0, 95.0, 105.0], dtype=np.float64)

        det = KeypointDetection(keypoints=det_kps, score=0.9, bbox=bbox)
        gt = KeypointGroundTruth(keypoints=gt_kps, bbox=bbox)

        oks = compute_oks(det, gt, sigmas)
        # Should be 1.0 since the invisible keypoint doesn't count
        assert oks == pytest.approx(1.0)


class TestComputeOKSMatrix:
    """Tests for compute_oks_matrix."""

    def test_shape(
        self,
        perfect_detection: KeypointDetection,
        offset_detection: KeypointDetection,
        matching_ground_truth: KeypointGroundTruth,
        distant_ground_truth: KeypointGroundTruth,
        sigmas: npt.NDArray[np.float64],
    ) -> None:
        """Matrix has shape (N_det, N_gt)."""
        dets = [perfect_detection, offset_detection]
        gts = [matching_ground_truth, distant_ground_truth]
        matrix = compute_oks_matrix(dets, gts, sigmas)
        assert matrix.shape == (2, 2)

    def test_empty_detections(
        self,
        matching_ground_truth: KeypointGroundTruth,
        sigmas: npt.NDArray[np.float64],
    ) -> None:
        """Empty detections produce (0, N_gt) matrix."""
        matrix = compute_oks_matrix([], [matching_ground_truth], sigmas)
        assert matrix.shape == (0, 1)

    def test_empty_ground_truths(
        self,
        perfect_detection: KeypointDetection,
        sigmas: npt.NDArray[np.float64],
    ) -> None:
        """Empty GTs produce (N_det, 0) matrix."""
        matrix = compute_oks_matrix([perfect_detection], [], sigmas)
        assert matrix.shape == (1, 0)

    def test_diagonal_highest_for_perfect_match(
        self,
        sigmas: npt.NDArray[np.float64],
    ) -> None:
        """Perfect det-GT pairs have highest similarity on diagonal."""
        kps1 = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
        kps2 = np.array([[100.0, 100.0], [110.0, 110.0]], dtype=np.float64)
        bbox1 = np.array([5.0, 5.0, 25.0, 25.0], dtype=np.float64)
        bbox2 = np.array([95.0, 95.0, 115.0, 115.0], dtype=np.float64)
        sigma2 = np.array([0.05, 0.05], dtype=np.float64)

        det1 = KeypointDetection(keypoints=kps1, score=0.9, bbox=bbox1)
        det2 = KeypointDetection(keypoints=kps2, score=0.8, bbox=bbox2)
        gt1 = KeypointGroundTruth(keypoints=kps1, bbox=bbox1)
        gt2 = KeypointGroundTruth(keypoints=kps2, bbox=bbox2)

        matrix = compute_oks_matrix([det1, det2], [gt1, gt2], sigma2)
        assert matrix[0, 0] > matrix[0, 1]
        assert matrix[1, 1] > matrix[1, 0]


class TestComputeBboxIoU:
    """Tests for compute_bbox_iou."""

    def test_identical_boxes(self) -> None:
        """Identical boxes have IoU of 1.0."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float64)
        assert compute_bbox_iou(bbox, bbox) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes have IoU of 0.0."""
        a = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
        b = np.array([20.0, 20.0, 30.0, 30.0], dtype=np.float64)
        assert compute_bbox_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Partially overlapping boxes have IoU between 0 and 1."""
        a = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
        b = np.array([5.0, 5.0, 15.0, 15.0], dtype=np.float64)
        iou = compute_bbox_iou(a, b)
        assert 0.0 < iou < 1.0
        # Intersection: 5x5=25, Union: 100+100-25=175
        assert iou == pytest.approx(25.0 / 175.0)

    def test_contained_box(self) -> None:
        """Smaller box contained in larger has IoU = small_area / large_area."""
        outer = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float64)
        inner = np.array([25.0, 25.0, 75.0, 75.0], dtype=np.float64)
        iou = compute_bbox_iou(outer, inner)
        # Intersection: 50*50=2500, Union: 10000+2500-2500=10000
        assert iou == pytest.approx(2500.0 / 10000.0)

    def test_zero_area_box(self) -> None:
        """Zero-area box produces IoU of 0.0."""
        a = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)
        b = np.array([0.0, 0.0, 20.0, 20.0], dtype=np.float64)
        assert compute_bbox_iou(a, b) == pytest.approx(0.0)


class TestComputeBboxIoUMatrix:
    """Tests for compute_bbox_iou_matrix."""

    def test_shape(
        self,
        perfect_detection: KeypointDetection,
        offset_detection: KeypointDetection,
        matching_ground_truth: KeypointGroundTruth,
    ) -> None:
        """Matrix has shape (N_det, N_gt)."""
        matrix = compute_bbox_iou_matrix(
            [perfect_detection, offset_detection], [matching_ground_truth]
        )
        assert matrix.shape == (2, 1)

    def test_empty_inputs(self) -> None:
        """Empty inputs produce correctly shaped zero matrix."""
        assert compute_bbox_iou_matrix([], []).shape == (0, 0)

    def test_identical_boxes_on_diagonal(self) -> None:
        """Identical det-GT bboxes produce 1.0 on diagonal."""
        bbox = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
        kps = np.zeros((3, 2), dtype=np.float64)

        det = KeypointDetection(keypoints=kps, score=0.9, bbox=bbox)
        gt = KeypointGroundTruth(keypoints=kps, bbox=bbox)

        matrix = compute_bbox_iou_matrix([det], [gt])
        assert matrix[0, 0] == pytest.approx(1.0)
