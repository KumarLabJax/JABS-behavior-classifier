"""Tests for what a label edit repaints in the player."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.main_window.central_widget import CentralWidget
    from jabs.ui.player_widget import PlayerWidget

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


def _widget() -> SimpleNamespace:
    """Stand-in exposing what _label_button_common() touches."""
    return SimpleNamespace(
        _project=SimpleNamespace(save_annotations=MagicMock()),
        _labels=object(),
        _pose_est=object(),
        _controls=SimpleNamespace(disable_label_buttons=MagicMock()),
        _jabs_timeline=SimpleNamespace(clear_selection=MagicMock()),
        _update_label_counts=MagicMock(),
        set_train_button_enabled_state=MagicMock(),
        _set_label_track=MagicMock(),
        _player_widget=SimpleNamespace(reload_frame=MagicMock(), update=MagicMock()),
    )


def test_a_label_edit_does_not_reload_the_frame_itself() -> None:
    """Nothing decoded into the frame comes from the labels.

    The label overlay is painted on top of the frame and repaints itself when the new
    values are pushed to it, so re-decoding the frame here only duplicated that (or did
    nothing at all, with the overlay switched off).
    """
    widget = _widget()

    CentralWidget._label_button_common(widget)

    widget._player_widget.reload_frame.assert_not_called()
    widget._player_widget.update.assert_not_called()


def test_a_label_edit_still_refreshes_the_label_track() -> None:
    """The timeline and the label overlay are both refreshed from here."""
    widget = _widget()

    CentralWidget._label_button_common(widget)

    widget._set_label_track.assert_called_once_with()
    widget._project.save_annotations.assert_called_once_with(widget._labels, widget._pose_est)


def test_pushing_overlay_values_is_what_repaints_the_frame() -> None:
    """The overlay's own repaint is what makes an edit visible with the overlay on."""
    player = PlayerWidget()
    player._frame_widget = MagicMock()
    player.reload_frame = MagicMock()

    player.set_labels([np.ones(3, dtype=np.int8)], None)

    player._frame_widget.set_label_overlay.assert_called_once()
    player.reload_frame.assert_called_once_with()
