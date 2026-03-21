"""Shared fixtures for keypoint evaluation metrics tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from jabs.vision.timm.metrics import (
    ImageEvalData,
    KeypointDetection,
    KeypointGroundTruth,
)

NUM_KEYPOINTS = 5


@pytest.fixture()
def sigmas() -> npt.NDArray[np.float64]:
    """Per-keypoint OKS sigmas for 5 keypoints."""
    return np.array([0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float64)


@pytest.fixture()
def perfect_detection() -> KeypointDetection:
    """A high-confidence detection at known coordinates."""
    kps = np.array(
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0], [90.0, 100.0]],
        dtype=np.float64,
    )
    bbox = np.array([5.0, 15.0, 95.0, 105.0], dtype=np.float64)
    return KeypointDetection(keypoints=kps, score=0.95, bbox=bbox)


@pytest.fixture()
def matching_ground_truth() -> KeypointGroundTruth:
    """A ground truth that exactly matches perfect_detection."""
    kps = np.array(
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0], [90.0, 100.0]],
        dtype=np.float64,
    )
    bbox = np.array([5.0, 15.0, 95.0, 105.0], dtype=np.float64)
    return KeypointGroundTruth(keypoints=kps, bbox=bbox)


@pytest.fixture()
def offset_detection() -> KeypointDetection:
    """A detection with keypoints offset by a small amount."""
    kps = np.array(
        [[12.0, 22.0], [32.0, 42.0], [52.0, 62.0], [72.0, 82.0], [92.0, 102.0]],
        dtype=np.float64,
    )
    bbox = np.array([7.0, 17.0, 97.0, 107.0], dtype=np.float64)
    return KeypointDetection(keypoints=kps, score=0.80, bbox=bbox)


@pytest.fixture()
def distant_ground_truth() -> KeypointGroundTruth:
    """A ground truth far from perfect_detection."""
    kps = np.array(
        [[200.0, 200.0], [220.0, 220.0], [240.0, 240.0], [260.0, 260.0], [280.0, 280.0]],
        dtype=np.float64,
    )
    bbox = np.array([195.0, 195.0, 285.0, 285.0], dtype=np.float64)
    return KeypointGroundTruth(keypoints=kps, bbox=bbox)


@pytest.fixture()
def crowd_ground_truth() -> KeypointGroundTruth:
    """A crowd ground truth annotation."""
    kps = np.array(
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0], [90.0, 100.0]],
        dtype=np.float64,
    )
    bbox = np.array([5.0, 15.0, 95.0, 105.0], dtype=np.float64)
    return KeypointGroundTruth(keypoints=kps, bbox=bbox, is_crowd=True)


@pytest.fixture()
def gt_with_visibility() -> KeypointGroundTruth:
    """A ground truth with visibility flags (some keypoints not visible)."""
    kps = np.array(
        [
            [10.0, 20.0, 2.0],
            [30.0, 40.0, 2.0],
            [50.0, 60.0, 0.0],  # Not labeled
            [70.0, 80.0, 1.0],  # Occluded but labeled
            [90.0, 100.0, 2.0],
        ],
        dtype=np.float64,
    )
    bbox = np.array([5.0, 15.0, 95.0, 105.0], dtype=np.float64)
    return KeypointGroundTruth(keypoints=kps, bbox=bbox)


@pytest.fixture()
def perfect_image(
    perfect_detection: KeypointDetection,
    matching_ground_truth: KeypointGroundTruth,
) -> ImageEvalData:
    """An image with a single perfect detection matching a single GT."""
    return ImageEvalData(
        detections=[perfect_detection],
        ground_truths=[matching_ground_truth],
    )


@pytest.fixture()
def multi_detection_image(
    perfect_detection: KeypointDetection,
    offset_detection: KeypointDetection,
    matching_ground_truth: KeypointGroundTruth,
    distant_ground_truth: KeypointGroundTruth,
) -> ImageEvalData:
    """An image with two detections and two GTs."""
    return ImageEvalData(
        detections=[perfect_detection, offset_detection],
        ground_truths=[matching_ground_truth, distant_ground_truth],
    )
