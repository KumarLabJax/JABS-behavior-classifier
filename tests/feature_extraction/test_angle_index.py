"""Unit tests for the AngleIndex angle/keypoint lookup tables."""

import pytest

from jabs.feature_extraction.angle_index import AngleIndex
from jabs.pose_estimation import PoseEstimation

# The exact name of every angle. These strings become part of the feature column
# names written to the feature cache, so a change here renames cached features
# and would need a FEATURE_VERSION bump. Pinning them keeps the derivation in
# angle_index.py from silently doing that.
EXPECTED_NAMES = {
    AngleIndex.NOSE_BASE_NECK_RIGHT_FRONT_PAW: "NOSE-BASE_NECK-RIGHT_FRONT_PAW",
    AngleIndex.NOSE_BASE_NECK_LEFT_FRONT_PAW: "NOSE-BASE_NECK-LEFT_FRONT_PAW",
    AngleIndex.RIGHT_FRONT_PAW_BASE_NECK_CENTER_SPINE: "RIGHT_FRONT_PAW-BASE_NECK-CENTER_SPINE",
    AngleIndex.LEFT_FRONT_PAW_BASE_NECK_CENTER_SPINE: "LEFT_FRONT_PAW-BASE_NECK-CENTER_SPINE",
    AngleIndex.BASE_NECK_CENTER_SPINE_BASE_TAIL: "BASE_NECK-CENTER_SPINE-BASE_TAIL",
    AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_CENTER_SPINE: "RIGHT_REAR_PAW-BASE_TAIL-CENTER_SPINE",
    AngleIndex.LEFT_REAR_PAW_BASE_TAIL_CENTER_SPINE: "LEFT_REAR_PAW-BASE_TAIL-CENTER_SPINE",
    AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_MID_TAIL: "RIGHT_REAR_PAW-BASE_TAIL-MID_TAIL",
    AngleIndex.LEFT_REAR_PAW_BASE_TAIL_MID_TAIL: "LEFT_REAR_PAW-BASE_TAIL-MID_TAIL",
    AngleIndex.CENTER_SPINE_BASE_TAIL_MID_TAIL: "CENTER_SPINE-BASE_TAIL-MID_TAIL",
    AngleIndex.BASE_TAIL_MID_TAIL_TIP_TAIL: "BASE_TAIL-MID_TAIL-TIP_TAIL",
}


@pytest.mark.parametrize("angle", list(AngleIndex), ids=lambda a: a.name)
def test_get_angle_name(angle: AngleIndex) -> None:
    """Every angle maps to its documented name."""
    assert AngleIndex.get_angle_name(angle) == EXPECTED_NAMES[angle]


def test_expected_names_cover_every_angle() -> None:
    """The pinned name table has an entry for every AngleIndex member."""
    assert set(EXPECTED_NAMES) == set(AngleIndex)


@pytest.mark.parametrize("angle", list(AngleIndex), ids=lambda a: a.name)
def test_get_angle_indices_are_three_distinct_keypoints(angle: AngleIndex) -> None:
    """Every angle is defined by three distinct keypoints."""
    keypoints = AngleIndex.get_angle_indices(angle)

    assert len(keypoints) == 3
    assert all(isinstance(kp, PoseEstimation.KeypointIndex) for kp in keypoints)
    assert len(set(keypoints)) == 3


@pytest.mark.parametrize("angle", list(AngleIndex), ids=lambda a: a.name)
def test_angle_name_matches_its_keypoints(angle: AngleIndex) -> None:
    """An angle's name names the keypoints it is measured from, in order."""
    keypoints = AngleIndex.get_angle_indices(angle)
    assert AngleIndex.get_angle_name(angle).split("-") == [kp.name for kp in keypoints]


def test_get_angle_indices_returns_a_fresh_list() -> None:
    """Mutating a returned list does not corrupt the shared keypoint table."""
    angle = AngleIndex.BASE_TAIL_MID_TAIL_TIP_TAIL
    original = AngleIndex.get_angle_indices(angle)

    returned = AngleIndex.get_angle_indices(angle)
    returned.clear()

    assert AngleIndex.get_angle_indices(angle) == original
