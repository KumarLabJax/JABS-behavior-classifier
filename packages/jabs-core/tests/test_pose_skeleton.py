"""Tests for the pose skeleton constants, PoseCapability and SegmentationWindow."""

import dataclasses

import numpy as np
import pytest

from jabs.core import pose_skeleton
from jabs.core.abstract import pose_est
from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.enums import PoseCapability
from jabs.core.types import SegmentationWindow

EXPECTED_KEYPOINTS = {
    "NOSE": 0,
    "LEFT_EAR": 1,
    "RIGHT_EAR": 2,
    "BASE_NECK": 3,
    "LEFT_FRONT_PAW": 4,
    "RIGHT_FRONT_PAW": 5,
    "CENTER_SPINE": 6,
    "LEFT_REAR_PAW": 7,
    "RIGHT_REAR_PAW": 8,
    "BASE_TAIL": 9,
    "MID_TAIL": 10,
    "TIP_TAIL": 11,
}


def test_keypoint_index_names_and_values():
    """KeypointIndex has the 12 keypoints with their historical indexes."""
    assert {k.name: k.value for k in pose_skeleton.KeypointIndex} == EXPECTED_KEYPOINTS


def test_pose_estimation_attributes_are_aliases():
    """The ABC's class attributes are the same objects as the module's."""
    assert PoseEstimation.KeypointIndex is pose_skeleton.KeypointIndex
    assert PoseEstimation.FULL_CONNECTED_SEGMENTS is pose_skeleton.FULL_CONNECTED_SEGMENTS
    assert PoseEstimation.NVSN_CONNECTED_SEGMENTS is pose_skeleton.NVSN_CONNECTED_SEGMENTS


def test_minimum_confidence():
    """MINIMUM_CONFIDENCE keeps its value and stays importable from the ABC module."""
    assert pose_skeleton.MINIMUM_CONFIDENCE == 0.3
    assert pose_est.MINIMUM_CONFIDENCE == pose_skeleton.MINIMUM_CONFIDENCE


def test_connected_segment_values():
    """Connected segments keep their keypoint sequences."""
    names = [[k.name for k in seg] for seg in pose_skeleton.FULL_CONNECTED_SEGMENTS]
    assert names == [
        ["LEFT_FRONT_PAW", "CENTER_SPINE", "RIGHT_FRONT_PAW"],
        ["LEFT_REAR_PAW", "BASE_TAIL", "RIGHT_REAR_PAW"],
        ["NOSE", "BASE_NECK", "CENTER_SPINE", "BASE_TAIL", "MID_TAIL", "TIP_TAIL"],
        ["LEFT_EAR", "NOSE", "RIGHT_EAR"],
    ]
    nvsn = [[k.name for k in seg] for seg in pose_skeleton.NVSN_CONNECTED_SEGMENTS]
    assert nvsn == [["LEFT_EAR", "NOSE", "RIGHT_EAR"], ["NOSE", "BASE_TAIL", "TIP_TAIL"]]


def test_tail_keypoints():
    """TAIL_KEYPOINTS names the tail keypoints excluded from identity presence."""
    assert pose_skeleton.TAIL_KEYPOINTS == ("MID_TAIL", "TIP_TAIL")
    for name in pose_skeleton.TAIL_KEYPOINTS:
        assert name in pose_skeleton.KeypointIndex.__members__


def test_pose_capability_values():
    """PoseCapability is a str enum with the expected members."""
    assert {c.name: c.value for c in PoseCapability} == {
        "SOCIAL": "social",
        "SEGMENTATION": "segmentation",
        "BOUNDING_BOXES": "bounding_boxes",
    }
    assert isinstance(PoseCapability.SOCIAL, str)
    assert PoseCapability("social") is PoseCapability.SOCIAL


def test_segmentation_window_is_frozen():
    """SegmentationWindow holds its arrays and rejects attribute assignment."""
    window = SegmentationWindow(
        contours=np.zeros((2, 1, 3, 2), dtype=np.int32),
        contour_count=np.zeros(2, dtype=np.uint32),
        contour_length=np.zeros((2, 1), dtype=np.uint32),
        external_flag=None,
        start=10,
    )
    assert window.start == 10
    assert window.external_flag is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        window.start = 0  # type: ignore[misc]
