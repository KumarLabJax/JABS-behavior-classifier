"""Burn a caption and color legend into the corner of an exported frame.

An exported video is watched outside JABS, where none of the menus that say what is
on screen are available and the marker colors mean nothing on their own. The banner
drawn here carries both: a line naming what is shown, and a swatch per color.

Text is the one part of the overlay that Qt cannot draw without a ``QGuiApplication``
(the font database needs one), so a caption is skipped, with a warning, when there is
no application instance. Everything else in an export still renders headlessly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from PySide6 import QtCore, QtGui

from jabs.overlay_drawing import native_overlay_scale

logger = logging.getLogger(__name__)

# Base sizes, in pixels, calibrated for an 800x800 frame and scaled from there.
_BASE_FONT_SIZE = 14
_BASE_MARGIN = 8
_BASE_PADDING = 6
_BASE_SWATCH_GAP = 5
_BASE_ENTRY_GAP = 14

_BANNER_COLOR = QtGui.QColor(0, 0, 0, 160)
_TEXT_COLOR = QtGui.QColor(255, 255, 255, 255)
_SWATCH_OUTLINE_COLOR = QtGui.QColor(255, 255, 255, 255)


def draw_overlay_caption(
    painter: QtGui.QPainter,
    width: int,
    height: int,
    caption: str,
    legend: Sequence[tuple[str, QtGui.QColor]] = (),
) -> None:
    """Draw the caption and legend banner in the frame's top-left corner.

    The banner is sized to its contents and legend entries wrap onto further lines
    when they do not fit the frame's width, so a project with many behaviors stays
    readable instead of running off the edge.

    Args:
        painter: Painter for the frame being rendered.
        width: Frame width in pixels.
        height: Frame height in pixels.
        caption: Line of text naming what the overlay shows. An empty caption with
            an empty legend draws nothing.
        legend: ``(name, color)`` pairs to draw as swatches below the caption.
    """
    if not caption and not legend:
        return

    if QtGui.QGuiApplication.instance() is None:
        # Drawing text would abort on QFontDatabase. Markers are already on the
        # frame by this point, so the export continues without the banner.
        logger.warning("No QGuiApplication: exporting without the overlay caption")
        return

    scale = native_overlay_scale(width, height)
    font_size = max(_BASE_FONT_SIZE, round(_BASE_FONT_SIZE * scale))
    margin = max(_BASE_MARGIN, round(_BASE_MARGIN * scale))
    padding = max(_BASE_PADDING, round(_BASE_PADDING * scale))
    swatch_gap = max(_BASE_SWATCH_GAP, round(_BASE_SWATCH_GAP * scale))
    entry_gap = max(_BASE_ENTRY_GAP, round(_BASE_ENTRY_GAP * scale))

    font = QtGui.QFont(painter.font())
    font.setPixelSize(font_size)
    metrics = QtGui.QFontMetrics(font)
    line_height = metrics.height()
    swatch_size = metrics.ascent()

    # Width available to the banner's contents, before its own padding.
    content_limit = max(1, width - 2 * margin - 2 * padding)

    rows = _wrap_legend(legend, metrics, swatch_size, swatch_gap, entry_gap, content_limit)

    content_width = metrics.horizontalAdvance(caption) if caption else 0
    content_width = max(content_width, *(row.width for row in rows)) if rows else content_width
    content_width = min(content_width, content_limit)
    line_count = (1 if caption else 0) + len(rows)

    banner = QtCore.QRect(
        margin,
        margin,
        content_width + 2 * padding,
        line_count * line_height + 2 * padding,
    )

    antialiasing = painter.testRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(_BANNER_COLOR)
    painter.drawRect(banner)

    painter.setFont(font)
    text_x = banner.left() + padding
    text_y = banner.top() + padding

    if caption:
        painter.setPen(_TEXT_COLOR)
        painter.drawText(
            QtCore.QRect(text_x, text_y, content_width, line_height),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            caption,
        )
        text_y += line_height

    for row in rows:
        x = text_x
        for name, color in row.entries:
            swatch_y = text_y + (line_height - swatch_size) // 2
            painter.setBrush(color)
            painter.setPen(_SWATCH_OUTLINE_COLOR)
            painter.drawRect(x, swatch_y, swatch_size, swatch_size)
            x += swatch_size + swatch_gap

            painter.setPen(_TEXT_COLOR)
            painter.drawText(
                QtCore.QRect(x, text_y, metrics.horizontalAdvance(name), line_height),
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                name,
            )
            x += metrics.horizontalAdvance(name) + entry_gap
        text_y += line_height

    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, antialiasing)


class _LegendRow:
    """One line of legend entries, with the width they occupy."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, QtGui.QColor]] = []
        self.width = 0


def _wrap_legend(
    legend: Sequence[tuple[str, QtGui.QColor]],
    metrics: QtGui.QFontMetrics,
    swatch_size: int,
    swatch_gap: int,
    entry_gap: int,
    limit: int,
) -> list[_LegendRow]:
    """Split legend entries into lines that fit within ``limit`` pixels.

    An entry wider than ``limit`` on its own still gets a line to itself rather than
    being dropped: an over-wide banner is better than a missing class.
    """
    rows: list[_LegendRow] = []
    row = _LegendRow()
    for name, color in legend:
        entry_width = swatch_size + swatch_gap + metrics.horizontalAdvance(name)
        needed = entry_width if not row.entries else row.width + entry_gap + entry_width
        if row.entries and needed > limit:
            rows.append(row)
            row = _LegendRow()
            needed = entry_width
        row.entries.append((name, color))
        row.width = needed
    if row.entries:
        rows.append(row)
    return rows
