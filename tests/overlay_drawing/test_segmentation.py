"""Tests for the shared segmentation contour drawing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jabs.core.types import SegmentationWindow
from jabs.pose_estimation import open_pose_file
from jabs.pose_estimation.pose_est_v6 import pack_segmentation

from .conftest import POSE_KINDS

try:
    from PySide6 import QtGui

    from jabs.overlay_drawing import (
        BEHAVIOR_COLOR,
        NOT_BEHAVIOR_COLOR,
        SEGMENTATION_NO_PREDICTION_COLOR,
        draw_identity_segmentation,
        identity_contours,
        native_segmentation_line_width,
        segmentation_prediction_color,
    )

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)

# A square, and a second slot that the pose file left unused.
_SQUARE = [(10, 10), (50, 10), (50, 50), (10, 50)]


class StubPose:
    """Minimal pose stand-in returning canned segmentation data, one window per request."""

    def __init__(self, contours: np.ndarray | None, has_segmentation: bool = True) -> None:
        self._contours = contours
        self.has_segmentation = has_segmentation
        self.requests: list[tuple[int, int, int]] = []

    def segmentation_window(self, identity: int, start: int, stop: int) -> SegmentationWindow:
        """Record the request and return the canned contours packed into a window."""
        self.requests.append((identity, start, stop))
        if not self.has_segmentation:
            raise NotImplementedError("no segmentation")
        contours = self._contours if self._contours is not None else _padded()
        return pack_segmentation(contours[np.newaxis], None, start)


def _padded(*contours: list[tuple[int, int]], slots: int = 2, points: int = 6) -> np.ndarray:
    """Build segmentation data shaped the way a pose file stores it.

    A file keeps a fixed number of contour slots per identity, each with a fixed number
    of points, and pads everything it does not use with -1.
    """
    data = np.full((slots, points, 2), -1, dtype=np.int32)
    for slot, contour in enumerate(contours):
        for i, (x, y) in enumerate(contour):
            data[slot, i] = (x, y)
    return data


def test_padding_is_trimmed_from_contours() -> None:
    """Only the real points of a contour come back, not the -1 padding."""
    contours = identity_contours(StubPose(_padded(_SQUARE)), 0, 0)

    assert len(contours) == 1, "the unused second slot should be dropped"
    np.testing.assert_array_equal(contours[0], np.array(_SQUARE))


def test_multiple_contours_are_returned_separately() -> None:
    """An animal can have more than one contour on a frame."""
    second = [(60, 60), (70, 60), (70, 70)]

    contours = identity_contours(StubPose(_padded(_SQUARE, second)), 0, 0)

    assert len(contours) == 2
    np.testing.assert_array_equal(contours[1], np.array(second))


@pytest.mark.parametrize(
    ("pose", "expected_requests"),
    [
        (lambda: StubPose(_padded(_SQUARE), has_segmentation=False), []),
        (lambda: StubPose(np.full((2, 6, 2), -1, dtype=np.int32)), [(0, 0, 1)]),
    ],
    ids=["no-segmentation", "all-padding"],
)
def test_no_contours_cases(pose, expected_requests: list[tuple[int, int, int]]) -> None:
    """A frame without contours yields nothing, and a file without any is not even asked."""
    stub = pose()

    assert identity_contours(stub, 0, 0) == []
    assert stub.requests == expected_requests


def test_contours_come_from_a_one_frame_window() -> None:
    """The contours of a frame are read as the window ``[frame, frame + 1)``."""
    stub = StubPose(_padded(_SQUARE))

    identity_contours(stub, 7, 2)

    assert stub.requests == [(2, 7, 8)]


def test_a_hole_slot_before_a_contour_is_skipped() -> None:
    """Packing puts valid contours first, so a gap in the legacy slots is not a contour."""
    data = _padded([(0, 0)], _SQUARE)
    data[0] = -1

    contours = identity_contours(StubPose(data), 0, 0)

    assert len(contours) == 1
    np.testing.assert_array_equal(contours[0], np.array(_SQUARE))


@pytest.mark.parametrize("kind", POSE_KINDS)
def test_real_pose_contours_match_the_window(
    segmentation_poses: dict[str, Path], kind: str
) -> None:
    """A legacy v6 file and its unified conversion give the window's contours."""
    with open_pose_file(segmentation_poses[kind]) as pose:
        identity = 0
        window = pose.segmentation_window(identity, 0, 1)
        count = int(window.contour_count[0])
        assert count > 0, "fixture frame should have a contour"
        expected = [
            window.contours[0, c, : int(window.contour_length[0, c])] for c in range(count)
        ]

        contours = identity_contours(pose, 0, identity)

    assert len(contours) == count
    for got, want in zip(contours, expected, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("kind", POSE_KINDS)
def test_real_pose_contour_is_drawn(segmentation_poses: dict[str, Path], kind: str) -> None:
    """Drawing a real file's identity paints the first point of its contour."""
    with open_pose_file(segmentation_poses[kind]) as pose:
        contour = identity_contours(pose, 0, 0)[0]
        image = QtGui.QImage(800, 800, QtGui.QImage.Format.Format_RGB888)
        image.fill(QtGui.QColor(0, 0, 0))
        painter = QtGui.QPainter(image)
        try:
            draw_identity_segmentation(
                painter,
                pose,
                0,
                0,
                to_output=lambda x, y: (round(x), round(y)),
                line_width=1,
                active=True,
            )
        finally:
            painter.end()

    x, y = (int(v) for v in contour[0])
    assert _painted(image, x, y)


def test_line_width_grows_with_the_frame() -> None:
    """A contour was one pixel when cv2 drew it at 800x800; it stays that at that size."""
    assert native_segmentation_line_width(800, 800) == 1
    assert native_segmentation_line_width(400, 400) == 1
    assert native_segmentation_line_width(4000, 4000) > 1


def _draw(pose, *, active: bool = True, to_output=None, color=None) -> QtGui.QImage:
    """Draw one identity's contours onto a black frame."""
    image = QtGui.QImage(80, 80, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(0, 0, 0))
    if to_output is None:

        def to_output(x, y):
            return round(x), round(y)

    painter = QtGui.QPainter(image)
    try:
        draw_identity_segmentation(
            painter, pose, 0, 0, to_output=to_output, line_width=1, active=active, color=color
        )
    finally:
        painter.end()
    return image


def _painted(image: QtGui.QImage, x: int, y: int) -> bool:
    """Whether anything was drawn at this pixel."""
    return QtGui.QColor(image.pixel(x, y)).getRgb()[:3] != (0, 0, 0)


@pytest.mark.parametrize(
    ("active", "channel"),
    [(True, 0), (False, 2)],
    ids=["active-identity-is-red", "other-identities-are-blue"],
)
def test_contour_color_tracks_the_active_identity(active: bool, channel: int) -> None:
    """The active animal's contours are drawn in a different color from the rest."""
    image = _draw(StubPose(_padded(_SQUARE)), active=active)

    rgb = QtGui.QColor(image.pixel(30, 10)).getRgb()[:3]
    assert rgb[channel] == max(rgb), f"expected channel {channel} to dominate, got {rgb}"


def test_an_explicit_color_overrides_the_active_identity_color() -> None:
    """A color passed in is used in place of the active/inactive colors."""
    image = _draw(StubPose(_padded(_SQUARE)), active=True, color=QtGui.QColor(0, 255, 0))

    assert QtGui.QColor(image.pixel(30, 10)).getRgb()[:3] == (0, 255, 0)


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [(1, "behavior"), (0, "not_behavior"), (-1, "none")],
    ids=["behavior-is-orange", "not-behavior-is-blue", "no-prediction-is-white"],
)
def test_binary_prediction_colors(prediction: int, expected: str) -> None:
    """Binary predictions map to the label marker colors, or white for no prediction."""
    colors = {
        "behavior": BEHAVIOR_COLOR,
        "not_behavior": NOT_BEHAVIOR_COLOR,
        "none": SEGMENTATION_NO_PREDICTION_COLOR,
    }
    assert segmentation_prediction_color(prediction) == colors[expected]


def test_multiclass_prediction_colors_come_from_the_lut() -> None:
    """With a LUT, index 0 is no prediction and other indices are opaque LUT colors."""
    lut = np.array([[128, 128, 128, 255], [10, 20, 30, 100], [40, 50, 60, 255]], dtype=np.uint8)

    assert segmentation_prediction_color(0, lut) == SEGMENTATION_NO_PREDICTION_COLOR
    assert segmentation_prediction_color(1, lut) == QtGui.QColor(10, 20, 30)
    # an index past the end of the table is clamped to its last entry
    assert segmentation_prediction_color(9, lut) == QtGui.QColor(40, 50, 60)


def test_a_fully_visible_contour_is_closed() -> None:
    """A contour is a closed shape, so the last point joins back to the first."""
    image = _draw(StubPose(_padded(_SQUARE)))

    # The left edge only exists if the last point was joined back to the first.
    assert _painted(image, 10, 30)


def test_a_contour_cut_short_by_a_crop_is_not_closed() -> None:
    """Closing a clipped run would draw a chord across the cropped-away region."""

    def to_output(x, y):
        # Stand in for a crop that excludes the bottom-left corner of the square.
        return None if (x, y) == (10, 50) else (round(x), round(y))

    image = _draw(StubPose(_padded(_SQUARE)), to_output=to_output)

    assert _painted(image, 30, 10), "the visible run should still be drawn"
    assert not _painted(image, 10, 30), "must not close across the cropped region"


def test_nothing_is_drawn_without_contours() -> None:
    """No segmentation data means an untouched frame, not an error."""
    image = _draw(StubPose(None))

    assert not any(_painted(image, x, y) for x in range(0, 80, 5) for y in range(0, 80, 5))


def test_a_single_point_contour_is_drawn_as_a_pixel() -> None:
    """A one-point contour is degenerate, but cv2 drew it and a pose file can hold one."""
    image = _draw(StubPose(_padded([(30, 30)])))

    assert _painted(image, 30, 30)


def test_a_contour_clipped_across_its_wrap_around_stays_in_one_piece() -> None:
    """A closed shape's last point joins its first, so that edge can be visible too.

    Walking the points as a flat list would split the visible part into the head and
    tail of the list and lose the edge between them.
    """

    def to_output(x, y):
        # A crop that excludes the square's two right-hand corners, leaving the
        # remaining corners adjacent only through the closing edge.
        return None if x > 40 else (round(x), round(y))

    image = _draw(StubPose(_padded(_SQUARE)), to_output=to_output)

    # The left edge runs between the two visible corners.
    assert _painted(image, 10, 30)
