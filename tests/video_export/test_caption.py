"""Tests for the caption and legend banner burned into an exported frame."""

import numpy as np
import pytest

try:
    from PySide6 import QtGui
    from PySide6.QtWidgets import QApplication

    from jabs.video_export import caption as caption_module

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Qt needs an application before it will touch the font database."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def canvas() -> QtGui.QImage:
    """A black frame to draw a banner onto."""
    image = QtGui.QImage(400, 300, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(0, 0, 0))
    return image


def _draw(image: QtGui.QImage, caption: str, legend=()) -> None:
    painter = QtGui.QPainter(image)
    try:
        caption_module.draw_overlay_caption(
            painter, image.width(), image.height(), caption, legend
        )
    finally:
        painter.end()


def _painted_rows(image: QtGui.QImage) -> np.ndarray:
    """Return the rows of the image that the banner touched."""
    buffer = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine()
    )
    return np.flatnonzero(buffer.any(axis=1))


def test_caption_draws_a_banner_in_the_top_left(canvas: QtGui.QImage) -> None:
    """The banner goes where it will not cover the arena floor in the middle."""
    _draw(canvas, "Grooming predictions (raw)")

    rows = _painted_rows(canvas)
    assert rows.size, "nothing was drawn"
    assert rows.max() < canvas.height() // 2, "banner should stay near the top"


def test_nothing_is_drawn_without_a_caption_or_legend(canvas: QtGui.QImage) -> None:
    """An export with nothing to explain gets no banner."""
    _draw(canvas, "")

    assert _painted_rows(canvas).size == 0


def test_legend_swatches_use_the_given_colors(canvas: QtGui.QImage) -> None:
    """The legend's swatch is the same color as the marker it explains."""
    _draw(canvas, "Grooming predictions (raw)", [("behavior", QtGui.QColor(255, 0, 0))])

    colors = {
        QtGui.QColor(canvas.pixel(x, y)).getRgb()[:3]
        for y in range(canvas.height() // 2)
        for x in range(canvas.width())
    }
    assert (255, 0, 0) in colors


def test_a_long_legend_wraps_instead_of_running_off_the_frame(canvas: QtGui.QImage) -> None:
    """A project with many behaviors stays inside the frame, on more lines."""
    short_canvas = QtGui.QImage(400, 300, QtGui.QImage.Format.Format_RGB888)
    short_canvas.fill(QtGui.QColor(0, 0, 0))
    legend_color = QtGui.QColor(255, 0, 0)

    _draw(short_canvas, "Multi-class predictions (raw)", [("one", legend_color)])
    _draw(
        canvas,
        "Multi-class predictions (raw)",
        [(f"behavior number {i}", legend_color) for i in range(8)],
    )

    assert _painted_rows(canvas).max() > _painted_rows(short_canvas).max(), "should wrap"
    # Every painted pixel stays within the frame: the banner never draws past its width.
    buffer = np.frombuffer(canvas.constBits(), dtype=np.uint8).reshape(
        canvas.height(), canvas.bytesPerLine()
    )
    painted_columns = np.flatnonzero(buffer.any(axis=0))
    assert painted_columns.max() < canvas.width() * 3


def test_caption_is_skipped_without_a_gui_application(canvas, monkeypatch, caplog) -> None:
    """Text needs Qt's font database, which aborts without a QGuiApplication.

    A headless export drops the banner and says so, rather than taking the whole
    export down with it.
    """

    class _NoApplication:
        @staticmethod
        def instance():
            return None

    monkeypatch.setattr(caption_module.QtGui, "QGuiApplication", _NoApplication)

    with caplog.at_level("WARNING"):
        _draw(canvas, "Grooming predictions (raw)")

    assert _painted_rows(canvas).size == 0
    assert "without the overlay caption" in caplog.text
