"""Tests for the player's segmentation overlay."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    from PySide6 import QtCore
    from PySide6.QtWidgets import QApplication

    from jabs.overlay_drawing import (
        BEHAVIOR_COLOR,
        NOT_BEHAVIOR_COLOR,
        SEGMENTATION_NO_PREDICTION_COLOR,
    )
    from jabs.ui.player_widget.frame_with_overlays import FrameWithOverlaysWidget
    from jabs.ui.player_widget.overlays import segmentation_overlay as segmentation_module
    from jabs.ui.player_widget.overlays.segmentation_overlay import SegmentationOverlay

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _overlay(
    enabled: bool,
    identities=(0, 1, 2),
    active_identity: int = 1,
    color_mode=None,
    predictions=None,
):
    """A SegmentationOverlay on a stand-in frame widget, with a stub pose."""
    modes = FrameWithOverlaysWidget.SegmentationColorMode
    parent = SimpleNamespace(
        SegmentationColorMode=modes,
        segmentation_color_mode=color_mode or modes.IDENTITY,
        segmentation_predictions=predictions,
        segmentation_color_lut=None,
        pixmap=lambda: SimpleNamespace(isNull=lambda: False),
        pose=SimpleNamespace(identities=list(identities)),
        current_frame=7,
        active_identity=active_identity,
        is_drawn_active=lambda identity: identity == active_identity,
        scaled_pix_width=800,
        image_to_widget_coords_cropped=lambda x, y, rect: (int(x), int(y)),
    )
    overlay = SegmentationOverlay.__new__(SegmentationOverlay)
    overlay._parent = parent
    overlay._priority = 0
    overlay._enabled = enabled
    return overlay


def test_segmentation_keeps_its_place_in_the_paint_order():
    """The overlays that used to be baked into the frame keep the order they had.

    Painted in this order, the track stays under the contours and the closest-animal
    markers and landmarks stay on top of them, which is how the decoder composited
    them before any of the four became an overlay.
    """
    widget = FrameWithOverlaysWidget()

    assert [type(overlay).__name__ for overlay in widget.overlays[:4]] == [
        "TrackOverlay",
        "SegmentationOverlay",
        "ClosestIdentityOverlay",
        "LandmarkOverlay",
    ]
    assert isinstance(widget.overlays[1], SegmentationOverlay)


def test_disabled_overlay_draws_nothing(monkeypatch):
    """A switched-off overlay does not touch the painter."""
    draw = MagicMock()
    monkeypatch.setattr(segmentation_module, "draw_identity_segmentation", draw)

    _overlay(enabled=False).paint(MagicMock(), QtCore.QRect(0, 0, 800, 800))

    draw.assert_not_called()


def test_every_identity_is_drawn_and_only_the_active_one_is_highlighted(monkeypatch):
    """Each animal gets its contours; the selected one is drawn in the active color."""
    draw = MagicMock()
    monkeypatch.setattr(segmentation_module, "draw_identity_segmentation", draw)

    _overlay(enabled=True, identities=(0, 1, 2), active_identity=1).paint(
        MagicMock(), QtCore.QRect(0, 0, 800, 800)
    )

    assert [call.args[3] for call in draw.call_args_list] == [0, 1, 2]
    assert [call.kwargs["active"] for call in draw.call_args_list] == [False, True, False]
    assert all(call.kwargs["line_width"] >= 1 for call in draw.call_args_list)


def test_contours_thicken_with_the_zoom(monkeypatch):
    """A cv2-drawn contour grew with the image; the pen scales so it still looks the same."""
    draw = MagicMock()
    monkeypatch.setattr(segmentation_module, "draw_identity_segmentation", draw)
    overlay = _overlay(enabled=True, identities=(0,))

    overlay.paint(MagicMock(), QtCore.QRect(0, 0, 800, 800))
    unzoomed = draw.call_args.kwargs["line_width"]

    # A crop a quarter of the width, shown at the same size on screen, is 4x zoom.
    overlay.paint(MagicMock(), QtCore.QRect(0, 0, 200, 200))
    zoomed = draw.call_args.kwargs["line_width"]

    assert unzoomed == 1
    assert zoomed > unzoomed


def test_identity_mode_passes_no_color_override(monkeypatch):
    """Coloring by identity leaves the active/inactive colors to the drawing code."""
    draw = MagicMock()
    monkeypatch.setattr(segmentation_module, "draw_identity_segmentation", draw)

    _overlay(enabled=True).paint(MagicMock(), QtCore.QRect(0, 0, 800, 800))

    assert all(call.kwargs["color"] is None for call in draw.call_args_list)


def test_prediction_mode_colors_each_identity_by_its_prediction(monkeypatch):
    """Each animal is colored by its prediction on the current frame (frame 7)."""
    draw = MagicMock()
    monkeypatch.setattr(segmentation_module, "draw_identity_segmentation", draw)
    predictions = [np.full(10, value, dtype=np.int8) for value in (1, 0, -1)]
    predictions.append(np.ones(5, dtype=np.int8))  # shorter than the current frame

    _overlay(
        enabled=True,
        identities=(0, 1, 2, 3, 4),
        color_mode=FrameWithOverlaysWidget.SegmentationColorMode.PREDICTION,
        predictions=predictions,
    ).paint(MagicMock(), QtCore.QRect(0, 0, 800, 800))

    assert [call.kwargs["color"] for call in draw.call_args_list] == [
        BEHAVIOR_COLOR,
        NOT_BEHAVIOR_COLOR,
        SEGMENTATION_NO_PREDICTION_COLOR,
        SEGMENTATION_NO_PREDICTION_COLOR,  # predictions end before the current frame
        SEGMENTATION_NO_PREDICTION_COLOR,  # no predictions for this identity at all
    ]


def test_prediction_mode_without_predictions_draws_white(monkeypatch):
    """With nothing to color by, every contour is drawn as having no prediction."""
    draw = MagicMock()
    monkeypatch.setattr(segmentation_module, "draw_identity_segmentation", draw)

    _overlay(
        enabled=True, color_mode=FrameWithOverlaysWidget.SegmentationColorMode.PREDICTION
    ).paint(MagicMock(), QtCore.QRect(0, 0, 800, 800))

    assert all(
        call.kwargs["color"] == SEGMENTATION_NO_PREDICTION_COLOR for call in draw.call_args_list
    )
