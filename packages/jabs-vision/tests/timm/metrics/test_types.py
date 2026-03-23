"""Tests for input/output data types."""

from __future__ import annotations

import numpy as np
import pytest

from jabs.vision.timm.metrics import (
    DetectionAPResult,
    ImageEvalData,
    KeypointDetection,
    KeypointGroundTruth,
    KeypointPCKDetail,
    PCKResult,
    PerThresholdResult,
    PoseAPResult,
)


class TestKeypointDetection:
    """Tests for KeypointDetection dataclass."""

    def test_valid_2d_keypoints(self) -> None:
        """Valid detection with (K, 2) keypoints."""
        kps = np.zeros((5, 2), dtype=np.float64)
        bbox = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
        det = KeypointDetection(keypoints=kps, score=0.9, bbox=bbox)
        assert det.keypoints.shape == (5, 2)
        assert det.score == 0.9

    def test_valid_3d_keypoints(self) -> None:
        """Valid detection with (K, 3) keypoints (with visibility)."""
        kps = np.zeros((5, 3), dtype=np.float64)
        bbox = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
        det = KeypointDetection(keypoints=kps, score=0.5, bbox=bbox)
        assert det.keypoints.shape == (5, 3)

    def test_invalid_keypoints_shape_1d(self) -> None:
        """Reject 1D keypoints."""
        with pytest.raises(ValueError, match="keypoints must have shape"):
            KeypointDetection(
                keypoints=np.zeros(10, dtype=np.float64),
                score=0.9,
                bbox=np.zeros(4, dtype=np.float64),
            )

    def test_invalid_keypoints_shape_wrong_cols(self) -> None:
        """Reject keypoints with wrong number of columns."""
        with pytest.raises(ValueError, match="keypoints must have shape"):
            KeypointDetection(
                keypoints=np.zeros((5, 4), dtype=np.float64),
                score=0.9,
                bbox=np.zeros(4, dtype=np.float64),
            )

    def test_invalid_bbox_shape(self) -> None:
        """Reject bbox with wrong shape."""
        with pytest.raises(ValueError, match="bbox must have shape"):
            KeypointDetection(
                keypoints=np.zeros((5, 2), dtype=np.float64),
                score=0.9,
                bbox=np.zeros(5, dtype=np.float64),
            )

    def test_frozen(self) -> None:
        """Verify dataclass is frozen."""
        det = KeypointDetection(
            keypoints=np.zeros((5, 2), dtype=np.float64),
            score=0.9,
            bbox=np.zeros(4, dtype=np.float64),
        )
        with pytest.raises(AttributeError):
            det.score = 0.5  # type: ignore[misc]


class TestKeypointGroundTruth:
    """Tests for KeypointGroundTruth dataclass."""

    def test_area_computed_from_bbox(self) -> None:
        """Area defaults to bbox area when not provided."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float64)
        gt = KeypointGroundTruth(keypoints=np.zeros((5, 2), dtype=np.float64), bbox=bbox)
        expected_area = (50.0 - 10.0) * (80.0 - 20.0)
        assert gt.area == pytest.approx(expected_area)

    def test_area_provided(self) -> None:
        """Explicit area is preserved."""
        gt = KeypointGroundTruth(
            keypoints=np.zeros((5, 2), dtype=np.float64),
            bbox=np.zeros(4, dtype=np.float64),
            area=500.0,
        )
        assert gt.area == 500.0

    def test_num_keypoints_from_visibility(self) -> None:
        """num_keypoints computed from visibility column."""
        kps = np.array(
            [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        gt = KeypointGroundTruth(keypoints=kps, bbox=np.zeros(4, dtype=np.float64))
        assert gt.num_keypoints == 2  # visibility > 0: indices 0 and 2

    def test_num_keypoints_without_visibility(self) -> None:
        """num_keypoints defaults to K when no visibility column."""
        gt = KeypointGroundTruth(
            keypoints=np.zeros((7, 2), dtype=np.float64),
            bbox=np.zeros(4, dtype=np.float64),
        )
        assert gt.num_keypoints == 7

    def test_num_keypoints_provided(self) -> None:
        """Explicit num_keypoints is preserved."""
        gt = KeypointGroundTruth(
            keypoints=np.zeros((5, 2), dtype=np.float64),
            bbox=np.zeros(4, dtype=np.float64),
            num_keypoints=3,
        )
        assert gt.num_keypoints == 3

    def test_is_crowd_default_false(self) -> None:
        """is_crowd defaults to False."""
        gt = KeypointGroundTruth(
            keypoints=np.zeros((5, 2), dtype=np.float64),
            bbox=np.zeros(4, dtype=np.float64),
        )
        assert gt.is_crowd is False

    def test_frozen(self) -> None:
        """Verify dataclass is frozen."""
        gt = KeypointGroundTruth(
            keypoints=np.zeros((5, 2), dtype=np.float64),
            bbox=np.zeros(4, dtype=np.float64),
        )
        with pytest.raises(AttributeError):
            gt.is_crowd = True  # type: ignore[misc]


class TestImageEvalData:
    """Tests for ImageEvalData dataclass."""

    def test_consistent_keypoints(
        self,
        perfect_detection: KeypointDetection,
        matching_ground_truth: KeypointGroundTruth,
    ) -> None:
        """Valid image with consistent keypoint counts."""
        img = ImageEvalData(
            detections=[perfect_detection],
            ground_truths=[matching_ground_truth],
        )
        assert len(img.detections) == 1
        assert len(img.ground_truths) == 1

    def test_inconsistent_keypoints(self) -> None:
        """Reject mixed keypoint counts."""
        det = KeypointDetection(
            keypoints=np.zeros((3, 2), dtype=np.float64),
            score=0.9,
            bbox=np.zeros(4, dtype=np.float64),
        )
        gt = KeypointGroundTruth(
            keypoints=np.zeros((5, 2), dtype=np.float64),
            bbox=np.zeros(4, dtype=np.float64),
        )
        with pytest.raises(ValueError, match="same number of keypoints"):
            ImageEvalData(detections=[det], ground_truths=[gt])

    def test_empty_detections(self) -> None:
        """Valid with empty detections list."""
        gt = KeypointGroundTruth(
            keypoints=np.zeros((5, 2), dtype=np.float64),
            bbox=np.zeros(4, dtype=np.float64),
        )
        img = ImageEvalData(detections=[], ground_truths=[gt])
        assert len(img.detections) == 0

    def test_empty_both(self) -> None:
        """Valid with both lists empty."""
        img = ImageEvalData(detections=[], ground_truths=[])
        assert len(img.detections) == 0
        assert len(img.ground_truths) == 0


class TestResultTypes:
    """Tests for result dataclasses."""

    def test_per_threshold_result(self) -> None:
        """PerThresholdResult stores values correctly."""
        r = PerThresholdResult(threshold=0.5, ap=0.8, ar=0.7)
        assert r.threshold == 0.5
        assert r.ap == 0.8
        assert r.ar == 0.7

    def test_pose_ap_result(self) -> None:
        """PoseAPResult stores all fields."""
        r = PoseAPResult(ap=0.6, ap_50=0.8, ap_75=0.5, ar=0.7, ar_50=0.9, ar_75=0.6)
        assert r.ap == 0.6
        assert r.per_threshold == []

    def test_detection_ap_result(self) -> None:
        """DetectionAPResult stores all fields."""
        r = DetectionAPResult(ap=0.6, ap_50=0.8, ap_75=0.5, ar=0.7, ar_50=0.9, ar_75=0.6)
        assert r.ap == 0.6

    def test_pck_result(self) -> None:
        """PCKResult stores all fields."""
        detail = KeypointPCKDetail(correct=8, total=10, pck=0.8)
        r = PCKResult(
            pck=0.8,
            threshold=0.2,
            per_keypoint={0: detail},
            excluded_indices=frozenset({2}),
        )
        assert r.pck == 0.8
        assert r.threshold == 0.2
        assert 0 in r.per_keypoint
        assert 2 in r.excluded_indices

    def test_keypoint_pck_detail(self) -> None:
        """KeypointPCKDetail stores values correctly."""
        d = KeypointPCKDetail(correct=5, total=10, pck=0.5)
        assert d.correct == 5
        assert d.total == 10
        assert d.pck == 0.5
