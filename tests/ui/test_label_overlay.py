"""Tests for the player's label overlay."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    from PySide6 import QtCore
    from PySide6.QtWidgets import QApplication

    from jabs.overlay_drawing import (
        LABEL_MARKER_GAP,
        LABEL_MARKER_PAIR_GAP,
        LABEL_MARKER_SIZE,
    )
    from jabs.ui.player_widget.frame_with_overlays import FrameWithOverlaysWidget
    from jabs.ui.player_widget.overlays import label_overlay as label_overlay_module
    from jabs.ui.player_widget.overlays.label_overlay import LabelOverlay

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)

_FRAME = 3
_CENTROID_X = 100
_CENTROID_Y = 200


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _labels(*values_per_identity: int) -> list[np.ndarray]:
    """One label array per identity, each holding ``value`` on every frame."""
    return [np.full(_FRAME + 1, value, dtype=np.int8) for value in values_per_identity]


def _overlay(
    manual_labels: list[np.ndarray] | None = None,
    predicted_labels: list[np.ndarray] | None = None,
    enabled: bool = True,
    identities: tuple[int, ...] = (0,),
    floating: bool = False,
):
    """A LabelOverlay on a stand-in frame widget, with a stub pose."""
    hull = SimpleNamespace(centroid=SimpleNamespace(x=_CENTROID_X, y=_CENTROID_Y))
    id_modes = FrameWithOverlaysWidget.IdentityOverlayMode
    parent = SimpleNamespace(
        pixmap=lambda: SimpleNamespace(isNull=lambda: False),
        pose=SimpleNamespace(
            identities=list(identities),
            get_identity_convex_hulls=lambda identity: {_FRAME: hull},
        ),
        current_frame=_FRAME,
        manual_labels=manual_labels,
        predicted_labels=predicted_labels,
        label_color_lut=None,
        identity_overlay_mode=id_modes.FLOATING if floating else id_modes.BBOX,
        IdentityOverlayMode=id_modes,
        image_to_widget_coords_cropped=lambda x, y, rect: (int(x), int(y)),
    )
    overlay = LabelOverlay.__new__(LabelOverlay)
    overlay._parent = parent
    overlay._priority = 0
    overlay._enabled = enabled
    return overlay


def _paint(overlay, monkeypatch) -> MagicMock:
    """Paint the overlay with the marker drawing stubbed out, and return the stub."""
    draw = MagicMock()
    monkeypatch.setattr(label_overlay_module, "draw_label_marker", draw)
    overlay.paint(MagicMock(), QtCore.QRect(0, 0, 800, 800))
    return draw


def test_disabled_overlay_draws_nothing(monkeypatch):
    """A switched-off overlay does not touch the painter."""
    draw = _paint(_overlay(manual_labels=_labels(1), enabled=False), monkeypatch)

    draw.assert_not_called()


def test_no_labels_draws_nothing(monkeypatch):
    """View > Label Overlay > No Overlay leaves both label sources unset."""
    draw = _paint(_overlay(), monkeypatch)

    draw.assert_not_called()


def test_empty_label_list_draws_nothing(monkeypatch):
    """No behavior or identity selected yet gives an empty list, not a marker."""
    draw = _paint(_overlay(manual_labels=[]), monkeypatch)

    draw.assert_not_called()


def test_one_marker_per_identity_for_a_single_label_source(monkeypatch):
    """Labels or predictions alone put one marker beside each animal."""
    draw = _paint(
        _overlay(manual_labels=_labels(1, 0, 1), identities=(0, 1, 2)),
        monkeypatch,
    )

    assert draw.call_count == 3


def test_a_single_marker_is_drawn_left_of_the_identity_label(monkeypatch):
    """Outside floating mode the marker leaves the centroid clear for the identity label."""
    draw = _paint(_overlay(predicted_labels=_labels(1)), monkeypatch)

    x, y = draw.call_args.args[1], draw.call_args.args[2]
    assert x == _CENTROID_X - LABEL_MARKER_SIZE - LABEL_MARKER_GAP
    assert y == _CENTROID_Y - LABEL_MARKER_SIZE


def test_a_single_marker_is_drawn_right_of_a_floating_identity_label(monkeypatch):
    """A floating identity label is connected to the centroid, so the marker moves right."""
    draw = _paint(_overlay(manual_labels=_labels(1), floating=True), monkeypatch)

    assert draw.call_args.args[1] == _CENTROID_X + LABEL_MARKER_GAP


def test_both_label_sources_are_drawn_side_by_side(monkeypatch):
    """The manual label comes first and the prediction sits next to it."""
    draw = _paint(
        _overlay(manual_labels=_labels(1), predicted_labels=_labels(0), floating=True),
        monkeypatch,
    )

    manual_call, prediction_call = draw.call_args_list
    manual_x = manual_call.args[1]
    assert manual_x == _CENTROID_X + LABEL_MARKER_GAP
    assert prediction_call.args[1] == manual_x + LABEL_MARKER_SIZE + LABEL_MARKER_PAIR_GAP
    # both markers sit on the same line, and each gets its own source's color
    assert manual_call.args[2] == prediction_call.args[2]
    assert manual_call.args[4] != prediction_call.args[4]


def test_the_pair_of_markers_stays_clear_of_the_identity_label(monkeypatch):
    """Drawn to the left, the whole pair is shifted, not just the first marker."""
    draw = _paint(_overlay(manual_labels=_labels(1), predicted_labels=_labels(0)), monkeypatch)

    pair_width = 2 * LABEL_MARKER_SIZE + LABEL_MARKER_PAIR_GAP
    assert draw.call_args_list[0].args[1] == _CENTROID_X - pair_width - LABEL_MARKER_GAP
    assert draw.call_args_list[1].args[1] == _CENTROID_X - LABEL_MARKER_SIZE - LABEL_MARKER_GAP


def test_both_sources_are_drawn_for_every_identity(monkeypatch):
    """Each animal gets its own pair of markers."""
    draw = _paint(
        _overlay(
            manual_labels=_labels(1, 0),
            predicted_labels=_labels(0, 1),
            identities=(0, 1),
        ),
        monkeypatch,
    )

    assert draw.call_count == 4


def test_a_source_missing_an_identity_keeps_the_other_marker_in_place(monkeypatch):
    """A label array built for a different pose file loses its marker, not the pair's layout."""
    draw = _paint(
        _overlay(
            manual_labels=_labels(1),  # only identity 0
            predicted_labels=_labels(0, 1),
            identities=(0, 1),
            floating=True,
        ),
        monkeypatch,
    )

    # identity 0 gets both markers, identity 1 only its prediction, drawn in the
    # position it would have had with the manual label present
    assert draw.call_count == 3
    prediction_x = _CENTROID_X + LABEL_MARKER_GAP + LABEL_MARKER_SIZE + LABEL_MARKER_PAIR_GAP
    assert draw.call_args_list[2].args[1] == prediction_x


def test_the_marker_is_skipped_outside_a_crop(monkeypatch):
    """An animal cropped out of the displayed region gets no marker."""
    overlay = _overlay(manual_labels=_labels(1))
    overlay.parent.image_to_widget_coords_cropped = lambda x, y, rect: None

    draw = _paint(overlay, monkeypatch)

    draw.assert_not_called()


def test_an_identity_without_a_convex_hull_is_skipped(monkeypatch):
    """An animal with no pose on this frame has nowhere to put a marker."""
    overlay = _overlay(manual_labels=_labels(1))
    overlay.parent.pose.get_identity_convex_hulls = lambda identity: {_FRAME: None}

    draw = _paint(overlay, monkeypatch)

    draw.assert_not_called()
