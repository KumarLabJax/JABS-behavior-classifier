"""Tests for the shared segmentation contour drawing."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from PySide6 import QtGui

    from jabs.overlay_drawing import (
        draw_identity_segmentation,
        identity_contours,
        native_segmentation_line_width,
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
    """Minimal pose stand-in returning canned segmentation data."""

    def __init__(self, contours: np.ndarray | None, version: int = 6) -> None:
        self._contours = contours
        self.format_major_version = version
        self.requests: list[tuple[int, int]] = []

    def get_segmentation_data_per_frame(self, frame_index: int, identity: int):
        """Record the request and return the canned data."""
        self.requests.append((frame_index, identity))
        return self._contours


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
        (lambda: StubPose(None), [(0, 0)]),
        (lambda: StubPose(_padded(_SQUARE), version=5), []),
        (lambda: StubPose(np.full((2, 6, 2), -1, dtype=np.int32)), [(0, 0)]),
    ],
    ids=["no-segmentation-data", "pose-predates-v6", "all-padding"],
)
def test_no_contours_cases(pose, expected_requests: list[tuple[int, int]]) -> None:
    """A file without contours yields nothing, and a pre-v6 file is not even asked."""
    stub = pose()

    assert identity_contours(stub, 0, 0) == []
    assert stub.requests == expected_requests


def test_line_width_grows_with_the_frame() -> None:
    """A contour was one pixel when cv2 drew it at 800x800; it stays that at that size."""
    assert native_segmentation_line_width(800, 800) == 1
    assert native_segmentation_line_width(400, 400) == 1
    assert native_segmentation_line_width(4000, 4000) > 1


def _draw(pose, *, active: bool = True, to_output=None) -> QtGui.QImage:
    """Draw one identity's contours onto a black frame."""
    image = QtGui.QImage(80, 80, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(0, 0, 0))
    if to_output is None:

        def to_output(x, y):
            return round(x), round(y)

    painter = QtGui.QPainter(image)
    try:
        draw_identity_segmentation(
            painter, pose, 0, 0, to_output=to_output, line_width=1, active=active
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
