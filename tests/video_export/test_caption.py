"""Tests for the caption and legend banner burned into an exported frame."""

# Annotations below name Qt types, which do not exist when the guarded import of
# PySide6 fails. Deferring their evaluation keeps this module importable, and so
# skippable, on a machine without Qt.
from __future__ import annotations

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


RED = QtGui.QColor(255, 0, 0) if not SKIP_UI_TESTS else None
BRIGHT = QtGui.QColor(0, 255, 0) if not SKIP_UI_TESTS else None


@pytest.fixture
def canvas() -> QtGui.QImage:
    """A black frame to draw a banner onto."""
    image = QtGui.QImage(400, 300, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(0, 0, 0))
    return image


@pytest.fixture
def bright_canvas() -> QtGui.QImage:
    """A light frame, so drawing that escapes the dark banner shows up."""
    image = QtGui.QImage(400, 300, QtGui.QImage.Format.Format_RGB888)
    image.fill(BRIGHT)
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
    return np.flatnonzero(_buffer(image).any(axis=1))


def _painted_columns(image: QtGui.QImage, background: QtGui.QColor | None = None) -> np.ndarray:
    """Return the pixel columns that differ from the frame's background color."""
    pixels = _buffer(image)[:, : image.width() * 3].reshape(image.height(), image.width(), 3)
    if background is None:
        changed = pixels.any(axis=2)
    else:
        changed = (pixels != np.array(background.getRgb()[:3], dtype=np.uint8)).any(axis=2)
    return np.flatnonzero(changed.any(axis=0))


def _buffer(image: QtGui.QImage) -> np.ndarray:
    """Return the image's raw bytes as ``(height, bytes_per_line)``."""
    return np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine()
    )


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
    assert _painted_columns(canvas).max() < canvas.width()


def test_caption_is_skipped_without_a_gui_application(canvas, monkeypatch, caplog) -> None:
    """Text needs Qt's font database, which aborts without a QGuiApplication.

    A headless export drops the banner and says so, rather than taking the whole
    export down with it.
    """
    _without_gui_application(monkeypatch)

    with caplog.at_level("WARNING"):
        _draw(canvas, "Grooming predictions (raw)")

    assert _painted_rows(canvas).size == 0
    assert "without the overlay caption" in caplog.text


def test_the_missing_application_warning_is_logged_once(canvas, monkeypatch, caplog) -> None:
    """The banner is drawn per frame, so an unlatched warning would flood the log."""
    _without_gui_application(monkeypatch)

    with caplog.at_level("WARNING"):
        for _ in range(5):
            _draw(canvas, "Grooming predictions (raw)")

    assert caplog.text.count("without the overlay caption") == 1


def test_an_entry_too_long_for_a_line_is_elided_to_fit(qapp) -> None:
    """A name that cannot fit a line of its own is shortened, not left over-wide.

    Wrapping alone cannot help an entry that is wider than the whole line, and a row
    wider than the limit is laid out past the banner regardless of how the banner is
    then sized.
    """
    font = QtGui.QFont()
    font.setPixelSize(14)
    metrics = QtGui.QFontMetrics(font)
    name = "a behavior with an unreasonably long name " * 3

    rows = caption_module._wrap_legend(
        [(name, RED)], metrics, swatch_size=11, swatch_gap=5, entry_gap=14, limit=200
    )

    assert len(rows) == 1
    assert rows[0].width <= 200
    (text, _color) = rows[0].entries[0]
    assert text != name, "the name should have been elided"
    assert text.rstrip("\u2026").strip() in name


def test_an_over_wide_entry_stays_inside_the_banner(bright_canvas) -> None:
    """Nothing is painted in the margin the banner is supposed to leave clear.

    Drawn on a light frame, because a stray swatch or glyph over the video is only
    visible where it does not land on the banner's own dark ground.
    """
    _draw(bright_canvas, "", [("a behavior with an unreasonably long name " * 3, RED)])

    painted = _painted_columns(bright_canvas, background=BRIGHT)
    assert painted.size, "the entry should still be drawn, elided"
    assert painted.max() < bright_canvas.width() - caption_module._BASE_MARGIN


def _without_gui_application(monkeypatch) -> None:
    """Make the caption code see no running application, with the warning re-armed."""

    class _NoApplication:
        @staticmethod
        def instance():
            return None

    monkeypatch.setattr(caption_module.QtGui, "QGuiApplication", _NoApplication)
    monkeypatch.setattr(caption_module, "_warned_without_application", False)
