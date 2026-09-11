"""Tests for the player's segmentation overlay."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    from PySide6 import QtCore
    from PySide6.QtWidgets import QApplication

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


def _overlay(enabled: bool, identities=(0, 1, 2), active_identity: int = 1):
    """A SegmentationOverlay on a stand-in frame widget, with a stub pose."""
    parent = SimpleNamespace(
        pixmap=lambda: SimpleNamespace(isNull=lambda: False),
        pose=SimpleNamespace(identities=list(identities)),
        current_frame=7,
        active_identity=active_identity,
        scaled_pix_width=800,
        image_to_widget_coords_cropped=lambda x, y, rect: (int(x), int(y)),
    )
    overlay = SegmentationOverlay.__new__(SegmentationOverlay)
    overlay._parent = parent
    overlay._priority = 0
    overlay._enabled = enabled
    return overlay


def test_the_overlay_starts_switched_off():
    """Segmentation is opt-in: View > Overlay Segmentation turns it on."""
    widget = FrameWithOverlaysWidget()

    assert widget.segmentation_overlay_enabled is False


def test_toggling_the_overlay_repaints_the_frame(monkeypatch):
    """The contours are painted over the frame on screen, so no frame is re-decoded."""
    widget = FrameWithOverlaysWidget()
    update = MagicMock()
    monkeypatch.setattr(widget, "update", update)

    widget.segmentation_overlay_enabled = True

    assert widget.segmentation_overlay_enabled is True
    update.assert_called_once_with()

    # Setting the same value again is not a change and must not force a repaint.
    widget.segmentation_overlay_enabled = True
    update.assert_called_once_with()


def test_segmentation_is_painted_under_the_other_overlays():
    """Contours used to be drawn into the frame, so everything else sat on top."""
    widget = FrameWithOverlaysWidget()

    assert isinstance(widget.overlays[0], SegmentationOverlay)


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
