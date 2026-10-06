"""Tests for the shared behavior label/prediction marker drawing."""

import numpy as np
import pytest

try:
    from PySide6 import QtGui

    from jabs.overlay_drawing import (
        BACKGROUND_COLOR,
        BEHAVIOR_COLOR,
        LABEL_MARKER_PAIR_GAP,
        LABEL_MARKER_SIZE,
        NOT_BEHAVIOR_COLOR,
        draw_label_marker,
        label_marker_color,
        native_label_marker_sizes,
    )

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


@pytest.mark.parametrize(
    ("label_value", "expected"),
    [(1, "behavior"), (0, "not_behavior"), (-1, "background"), (7, "background")],
    ids=["behavior", "not-behavior", "no-label", "unexpected-value"],
)
def test_binary_label_colors(label_value: int, expected: str) -> None:
    """Without a color table the binary scheme applies, and anything else is gray."""
    colors = {
        "behavior": BEHAVIOR_COLOR,
        "not_behavior": NOT_BEHAVIOR_COLOR,
        "background": BACKGROUND_COLOR,
    }

    assert label_marker_color(label_value) == colors[expected]


def test_lut_index_selects_the_table_color() -> None:
    """With a table, the label value is an index into it."""
    lut = np.array([[1, 2, 3, 255], [4, 5, 6, 255]], dtype=np.uint8)

    assert label_marker_color(1, lut).getRgb() == (4, 5, 6, 255)


@pytest.mark.parametrize(
    ("label_value", "expected"),
    [(9, (4, 5, 6, 255)), (-3, (1, 2, 3, 255))],
    ids=["above-the-table", "below-the-table"],
)
def test_out_of_range_lut_index_is_clamped(label_value: int, expected: tuple) -> None:
    """A label array that outgrew its table still renders rather than crashing."""
    lut = np.array([[1, 2, 3, 255], [4, 5, 6, 255]], dtype=np.uint8)

    assert label_marker_color(label_value, lut).getRgb() == expected


def test_marker_grows_with_the_frame_but_never_shrinks() -> None:
    """Markers scale up for larger frames and stay at the base size for small ones."""
    small_marker, small_gap, small_pair_gap = native_label_marker_sizes(400, 400)
    reference_marker, _, _ = native_label_marker_sizes(800, 800)
    large_marker, large_gap, large_pair_gap = native_label_marker_sizes(1920, 1080)

    assert small_marker == LABEL_MARKER_SIZE, "never smaller than the on-screen size"
    assert small_pair_gap == LABEL_MARKER_PAIR_GAP
    assert reference_marker == LABEL_MARKER_SIZE
    assert large_marker > reference_marker
    assert large_gap > small_gap
    assert large_pair_gap > small_pair_gap


def test_the_pair_gap_stays_tighter_than_the_gap_to_the_centroid() -> None:
    """A label and a prediction read as one group only if they sit closer together."""
    _marker, gap, pair_gap = native_label_marker_sizes(1920, 1080)

    assert pair_gap < gap


def test_draw_label_marker_fills_the_requested_square() -> None:
    """The marker lands where it was asked to, in the color it was given."""
    image = QtGui.QImage(60, 60, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(0, 0, 0))

    painter = QtGui.QPainter(image)
    try:
        draw_label_marker(painter, 20, 20, 10, QtGui.QColor(255, 0, 0))
    finally:
        painter.end()

    # Inside the square is the fill color, outside it is untouched.
    assert QtGui.QColor(image.pixel(25, 25)).getRgb()[:3] == (255, 0, 0)
    assert QtGui.QColor(image.pixel(5, 5)).getRgb()[:3] == (0, 0, 0)
    # The outline keeps the marker readable over the animal it sits beside.
    assert QtGui.QColor(image.pixel(20, 20)).getRgb()[:3] == (255, 255, 255)


def test_draw_label_marker_restores_the_antialiasing_setting() -> None:
    """Markers are drawn aliased for sharp edges, without disturbing later drawing."""
    image = QtGui.QImage(60, 60, QtGui.QImage.Format.Format_RGB888)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    try:
        draw_label_marker(painter, 10, 10, 10, QtGui.QColor(255, 0, 0))
        assert painter.testRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    finally:
        painter.end()
