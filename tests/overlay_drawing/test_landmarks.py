"""Tests for the shared arena landmark drawing."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from PySide6 import QtGui

    from jabs.overlay_drawing import (
        LANDMARK_CORNER_COLOR,
        LANDMARK_HOPPER_COLOR,
        LANDMARK_LIXIT_COLOR,
        draw_landmarks,
    )

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


class StubPose:
    """Pose stand-in carrying whichever static objects a test needs."""

    def __init__(self, static_objects: dict, num_lixit_keypoints: int = 1) -> None:
        self.static_objects = static_objects
        self.num_lixit_keypoints = num_lixit_keypoints


def _draw(pose: StubPose, to_output=None) -> QtGui.QImage:
    image = QtGui.QImage(120, 120, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(0, 0, 0))
    if to_output is None:

        def to_output(x, y):
            return round(x), round(y)

    painter = QtGui.QPainter(image)
    try:
        draw_landmarks(painter, pose, to_output=to_output, point_radius=2, line_width=1)
    finally:
        painter.end()
    return image


def _colors(image: QtGui.QImage) -> set[tuple[int, int, int]]:
    return {
        QtGui.QColor(image.pixel(x, y)).getRgb()[:3]
        for x in range(image.width())
        for y in range(image.height())
    } - {(0, 0, 0)}


def test_a_pose_file_with_no_static_objects_draws_nothing() -> None:
    """Static objects are optional, and older pose versions have none at all."""
    assert _colors(_draw(StubPose({}))) == set()


def test_corners_are_drawn() -> None:
    """The four arena corners get a marker each."""
    corners = np.array([[20, 20], [100, 20], [100, 100], [20, 100]], dtype=np.float32)

    assert LANDMARK_CORNER_COLOR.getRgb()[:3] in _colors(_draw(StubPose({"corners": corners})))


def test_a_single_keypoint_lixit_is_drawn() -> None:
    """Older pose files record a lixit as one point."""
    lixit = np.array([[60, 30]], dtype=np.float32)

    assert LANDMARK_LIXIT_COLOR.getRgb()[:3] in _colors(_draw(StubPose({"lixit": lixit})))


def test_a_three_keypoint_lixit_draws_every_keypoint() -> None:
    """The newer three-keypoint form draws all three, not just the first."""
    lixit = np.array([[[40, 30], [50, 30], [60, 30]]], dtype=np.float32)
    pose = StubPose({"lixit": lixit}, num_lixit_keypoints=3)

    image = _draw(pose)

    lixit_color = LANDMARK_LIXIT_COLOR.getRgb()[:3]
    assert all(QtGui.QColor(image.pixel(x, 30)).getRgb()[:3] == lixit_color for x in (40, 50, 60))


def test_the_food_hopper_outline_is_closed() -> None:
    """The hopper is a closed shape, as cv2 drew it."""
    hopper = np.array([[30, 30], [90, 30], [90, 60], [30, 60]], dtype=np.float32)

    image = _draw(StubPose({"food_hopper": hopper}))

    hopper_color = LANDMARK_HOPPER_COLOR.getRgb()[:3]
    # The left edge exists only if the last point was joined back to the first.
    assert QtGui.QColor(image.pixel(30, 45)).getRgb()[:3] == hopper_color


def test_landmarks_outside_a_crop_are_skipped() -> None:
    """A landmark outside the cropped region is not drawn at the edge instead."""
    corners = np.array([[20, 20], [100, 20]], dtype=np.float32)

    def to_output(x, y):
        return (round(x), round(y)) if x < 50 else None

    image = _draw(StubPose({"corners": corners}), to_output=to_output)

    corner_color = LANDMARK_CORNER_COLOR.getRgb()[:3]
    assert QtGui.QColor(image.pixel(20, 20)).getRgb()[:3] == corner_color
    assert QtGui.QColor(image.pixel(100, 20)).getRgb()[:3] != corner_color
