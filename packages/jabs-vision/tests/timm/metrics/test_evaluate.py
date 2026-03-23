"""Tests for high-level evaluation orchestrator functions."""

from __future__ import annotations

import numpy as np
import pytest

from jabs.vision.timm.metrics import (
    DetectionAPResult,
    ImageEvalData,
    KeypointDetection,
    KeypointGroundTruth,
    PCKResult,
    PoseAPResult,
    evaluate_detection,
    evaluate_pck,
    evaluate_pose,
    format_results,
)


def _make_perfect_image(num_keypoints: int = 5) -> ImageEvalData:
    """Create an image with a single perfect detection matching the GT."""
    kps = np.arange(num_keypoints * 2, dtype=np.float64).reshape(num_keypoints, 2)
    bbox = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float64)

    det = KeypointDetection(keypoints=kps.copy(), score=0.95, bbox=bbox.copy())
    gt = KeypointGroundTruth(keypoints=kps.copy(), bbox=bbox.copy())
    return ImageEvalData(detections=[det], ground_truths=[gt])


def _make_fp_image(num_keypoints: int = 5) -> ImageEvalData:
    """Create an image with a detection far from any GT."""
    det_kps = np.zeros((num_keypoints, 2), dtype=np.float64)
    gt_kps = np.full((num_keypoints, 2), 1000.0, dtype=np.float64)
    det_bbox = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    gt_bbox = np.array([990.0, 990.0, 1010.0, 1010.0], dtype=np.float64)

    det = KeypointDetection(keypoints=det_kps, score=0.5, bbox=det_bbox)
    gt = KeypointGroundTruth(keypoints=gt_kps, bbox=gt_bbox)
    return ImageEvalData(detections=[det], ground_truths=[gt])


class TestEvaluatePose:
    """Tests for evaluate_pose."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions produce AP and AR near 1.0."""
        images = [_make_perfect_image()]
        sigmas = np.full(5, 0.05, dtype=np.float64)

        result = evaluate_pose(images, sigmas)

        assert isinstance(result, PoseAPResult)
        assert result.ap == pytest.approx(1.0)
        assert result.ar == pytest.approx(1.0)
        assert result.ap_50 == pytest.approx(1.0)
        assert result.ap_75 == pytest.approx(1.0)

    def test_all_false_positives(self) -> None:
        """All-FP predictions produce AP of 0.0."""
        images = [_make_fp_image()]
        sigmas = np.full(5, 0.05, dtype=np.float64)

        result = evaluate_pose(images, sigmas)

        assert result.ap == pytest.approx(0.0)
        assert result.ar == pytest.approx(0.0)

    def test_per_threshold_results(self) -> None:
        """Per-threshold breakdown is populated."""
        images = [_make_perfect_image()]
        sigmas = np.full(5, 0.05, dtype=np.float64)

        result = evaluate_pose(images, sigmas)

        assert len(result.per_threshold) == 10  # 0.50 to 0.95
        for pt in result.per_threshold:
            assert 0.0 <= pt.ap <= 1.0
            assert 0.0 <= pt.ar <= 1.0

    def test_empty_images_raises(self) -> None:
        """Empty images list raises ValueError."""
        sigmas = np.full(5, 0.05, dtype=np.float64)
        with pytest.raises(ValueError, match="images must not be empty"):
            evaluate_pose([], sigmas)

    def test_multiple_images(self) -> None:
        """Multiple images are aggregated."""
        images = [_make_perfect_image(), _make_perfect_image()]
        sigmas = np.full(5, 0.05, dtype=np.float64)

        result = evaluate_pose(images, sigmas)

        assert result.ap == pytest.approx(1.0)
        assert result.ar == pytest.approx(1.0)


class TestEvaluateDetection:
    """Tests for evaluate_detection."""

    def test_perfect_bbox_match(self) -> None:
        """Perfect bbox overlap produces high AP."""
        images = [_make_perfect_image()]

        result = evaluate_detection(images)

        assert isinstance(result, DetectionAPResult)
        assert result.ap == pytest.approx(1.0)
        assert result.ar == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Non-overlapping bboxes produce AP=0."""
        images = [_make_fp_image()]

        result = evaluate_detection(images)

        assert result.ap == pytest.approx(0.0)

    def test_empty_images_raises(self) -> None:
        """Empty images list raises ValueError."""
        with pytest.raises(ValueError, match="images must not be empty"):
            evaluate_detection([])

    def test_per_threshold_results(self) -> None:
        """Per-threshold breakdown is populated."""
        images = [_make_perfect_image()]
        result = evaluate_detection(images)
        assert len(result.per_threshold) == 10


class TestEvaluatePCK:
    """Tests for evaluate_pck."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions produce PCK of 1.0."""
        images = [_make_perfect_image()]
        sigmas = np.full(5, 0.05, dtype=np.float64)

        result = evaluate_pck(images, sigmas, threshold=0.2)

        assert isinstance(result, PCKResult)
        assert result.pck == pytest.approx(1.0)
        assert result.threshold == 0.2

    def test_per_keypoint_details(self) -> None:
        """Per-keypoint details are populated."""
        images = [_make_perfect_image()]
        sigmas = np.full(5, 0.05, dtype=np.float64)

        result = evaluate_pck(images, sigmas, threshold=0.2)

        assert len(result.per_keypoint) == 5
        for detail in result.per_keypoint.values():
            assert detail.pck == pytest.approx(1.0)
            assert detail.total == 1

    def test_exclusion(self) -> None:
        """Excluded keypoints are not in per_keypoint."""
        images = [_make_perfect_image()]
        sigmas = np.full(5, 0.05, dtype=np.float64)

        result = evaluate_pck(images, sigmas, threshold=0.2, exclude_keypoint_indices=[0, 1])

        assert 0 not in result.per_keypoint
        assert 1 not in result.per_keypoint
        assert 0 in result.excluded_indices
        assert 1 in result.excluded_indices

    def test_empty_images_raises(self) -> None:
        """Empty images list raises ValueError."""
        sigmas = np.full(5, 0.05, dtype=np.float64)
        with pytest.raises(ValueError, match="images must not be empty"):
            evaluate_pck([], sigmas, threshold=0.2)

    def test_no_matches(self) -> None:
        """When nothing matches, PCK is 0.0."""
        images = [_make_fp_image()]
        sigmas = np.full(5, 0.05, dtype=np.float64)

        result = evaluate_pck(images, sigmas, threshold=0.2)
        assert result.pck == pytest.approx(0.0)


class TestFormatResults:
    """Tests for format_results."""

    def test_pose_section(self) -> None:
        """Pose results section is formatted correctly."""
        pose = PoseAPResult(ap=0.6, ap_50=0.8, ap_75=0.5, ar=0.7, ar_50=0.9, ar_75=0.6)
        output = format_results(pose=pose)
        assert "Pose Evaluation" in output
        assert "0.6000" in output
        assert "0.8000" in output

    def test_detection_section(self) -> None:
        """Detection results section is formatted correctly."""
        detection = DetectionAPResult(ap=0.5, ap_50=0.7, ap_75=0.4, ar=0.6, ar_50=0.8, ar_75=0.5)
        output = format_results(detection=detection)
        assert "Detection Evaluation" in output
        assert "0.5000" in output

    def test_pck_section(self) -> None:
        """PCK results section is formatted correctly."""
        from jabs.vision.timm.metrics import KeypointPCKDetail

        pck = PCKResult(
            pck=0.85,
            threshold=0.2,
            per_keypoint={
                0: KeypointPCKDetail(correct=8, total=10, pck=0.8),
                1: KeypointPCKDetail(correct=9, total=10, pck=0.9),
            },
        )
        output = format_results(pck_results=[pck])
        assert "PCK Evaluation" in output
        assert "0.8500" in output

    def test_keypoint_names(self) -> None:
        """Keypoint names appear in PCK output."""
        from jabs.vision.timm.metrics import KeypointPCKDetail

        pck = PCKResult(
            pck=0.9,
            threshold=0.2,
            per_keypoint={0: KeypointPCKDetail(correct=9, total=10, pck=0.9)},
        )
        output = format_results(pck_results=[pck], keypoint_names=["nose", "left_ear"])
        assert "nose" in output

    def test_all_sections(self) -> None:
        """All sections combined in one output."""
        pose = PoseAPResult(ap=0.6, ap_50=0.8, ap_75=0.5, ar=0.7, ar_50=0.9, ar_75=0.6)
        detection = DetectionAPResult(ap=0.5, ap_50=0.7, ap_75=0.4, ar=0.6, ar_50=0.8, ar_75=0.5)
        output = format_results(pose=pose, detection=detection)
        assert "Pose Evaluation" in output
        assert "Detection Evaluation" in output

    def test_empty_returns_empty(self) -> None:
        """No results produces empty string."""
        output = format_results()
        assert output == ""

    def test_excluded_indices_shown(self) -> None:
        """Excluded keypoints are listed in output."""
        pck = PCKResult(
            pck=0.9,
            threshold=0.2,
            excluded_indices=frozenset({2, 3}),
        )
        output = format_results(pck_results=[pck], keypoint_names=["a", "b", "c", "d"])
        assert "Excluded" in output
        assert "c" in output
        assert "d" in output
