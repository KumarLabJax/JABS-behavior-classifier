"""Tests for the shared movement-track drawing."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from PySide6 import QtGui

    from jabs.overlay_drawing import (
        TRACK_FUTURE_COLOR,
        TRACK_PAST_COLOR,
        draw_identity_track,
    )
    from jabs.pose_estimation import PoseEstimation

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)

_FRAMES = 30


class StubPose:
    """Pose stand-in whose nose walks left to right, one pixel per frame."""

    def __init__(self, masked_frames: tuple[int, ...] = ()) -> None:
        self._masked = masked_frames

    def get_identity_poses(self, identity: int):
        """Return (points, mask) for every frame, as a real pose object does."""
        n_kp = len(PoseEstimation.KeypointIndex)
        points = np.zeros((_FRAMES, n_kp, 2), dtype=np.float32)
        mask = np.ones((_FRAMES, n_kp), dtype=np.uint8)
        nose = PoseEstimation.KeypointIndex.NOSE
        for frame in range(_FRAMES):
            points[frame, nose] = (10 + frame * 3, 50)
        for frame in self._masked:
            mask[frame, nose] = 0
        return points, mask


def _draw(pose: StubPose, frame_index: int = 15, to_output=None) -> QtGui.QImage:
    image = QtGui.QImage(160, 100, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(0, 0, 0))
    if to_output is None:

        def to_output(x, y):
            return round(x), round(y)

    painter = QtGui.QPainter(image)
    try:
        draw_identity_track(
            painter,
            pose,
            frame_index,
            0,
            to_output=to_output,
            point_radius=1,
            line_width=1,
        )
    finally:
        painter.end()
    return image


def _colors(image: QtGui.QImage) -> set[tuple[int, int, int]]:
    return {
        QtGui.QColor(image.pixel(x, y)).getRgb()[:3]
        for x in range(image.width())
        for y in range(image.height())
    } - {(0, 0, 0)}


def test_past_and_future_are_drawn_in_different_colors() -> None:
    """Where the animal is going reads differently from where it has been."""
    colors = _colors(_draw(StubPose()))

    assert TRACK_FUTURE_COLOR.getRgb()[:3] in colors
    assert TRACK_PAST_COLOR.getRgb()[:3] in colors


def test_the_track_follows_the_current_frame() -> None:
    """The track is anchored on the frame being shown, not fixed in place."""
    early = _draw(StubPose(), frame_index=5)
    late = _draw(StubPose(), frame_index=25)

    assert _painted_columns(early).max() < _painted_columns(late).max()


def test_a_masked_keypoint_breaks_the_line_instead_of_being_bridged() -> None:
    """A frame with no nose is a gap in the track, not a shortcut across it."""
    complete = _draw(StubPose())
    with_gap = _draw(StubPose(masked_frames=(13, 14)))

    assert _painted(with_gap) < _painted(complete), "the gap should remove drawing"


def test_points_outside_a_crop_are_dropped() -> None:
    """A track leaving the cropped region stops at the edge."""

    def to_output(x, y):
        return (round(x), round(y)) if x < 40 else None

    # At frame 10 the nose is at x=40, so the earlier half of the past track is inside
    # the crop and everything from the current frame on is outside it.
    cropped = _draw(StubPose(), frame_index=10, to_output=to_output)

    painted = _painted_columns(cropped)
    assert painted.size, "the visible part of the track should still be drawn"
    assert painted.max() < 40


def test_a_track_at_the_start_of_the_video_does_not_run_off_the_front() -> None:
    """Frame 0 has no past to draw, and asking for one must not wrap around."""
    image = _draw(StubPose(), frame_index=0)

    # Only the current frame's marker and the future track, all at x >= 10.
    assert _painted_columns(image).min() >= 10 - 2


def _painted(image: QtGui.QImage) -> int:
    return sum(
        QtGui.QColor(image.pixel(x, y)).getRgb()[:3] != (0, 0, 0)
        for x in range(image.width())
        for y in range(image.height())
    )


def _painted_columns(image: QtGui.QImage) -> np.ndarray:
    columns = [
        x
        for x in range(image.width())
        if any(
            QtGui.QColor(image.pixel(x, y)).getRgb()[:3] != (0, 0, 0)
            for y in range(image.height())
        )
    ]
    return np.array(columns)
