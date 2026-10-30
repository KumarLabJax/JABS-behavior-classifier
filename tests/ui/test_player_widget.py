from pathlib import Path
from unittest import mock

import numpy as np
import pytest

try:
    from PySide6 import QtGui
    from PySide6.QtWidgets import QApplication

    from jabs.ui.player_widget import PlayerWidget
    from jabs.ui.player_widget.overlays.bounding_box import BoundingBoxOverlay

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_reset_clears_current_video_path():
    """Reset restores the documented no-video-loaded state."""
    widget = PlayerWidget()
    widget._video_path = Path("example.mp4")

    widget.reset()

    assert widget.current_video_path is None


def test_displayed_frame_number_comes_from_the_player_thread():
    """The image and its frame number have to agree, or every overlay is off by a frame.

    The position slider that ``current_frame`` reads is updated by a separate signal,
    which arrives after the image during playback. Taking the number from there left
    the pose, label and segmentation overlays drawn against the previous frame, while
    stepping through with the arrow keys looked correct because that path updates the
    position first.
    """
    widget = PlayerWidget()
    image = QtGui.QImage(8, 8, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(60, 60, 60))

    widget._display_image(image, 42)

    assert widget._frame_widget.current_frame == 42
    assert widget.current_frame == 0, "the slider is still where the other signal left it"


def test_the_display_slot_is_declared_with_both_arguments():
    """The Slot declaration has to match the method, which takes the frame number too.

    A stale one-argument declaration still works today - PySide6 dispatches to the
    Python callable, and both arguments arrive whether the connection is direct or
    queued - but it describes a signature that no longer exists.
    """
    widget = PlayerWidget()

    meta = widget.metaObject()
    assert meta.indexOfSlot("_display_image(QImage,int)") != -1
    assert meta.indexOfSlot("_display_image(QImage)") == -1


def test_frame_paints_without_a_pose_file():
    """A video opened without pose must paint; the bbox overlay used to assume a pose."""
    widget = PlayerWidget()
    frame_widget = widget._frame_widget
    frame_widget.set_pose(None)
    frame_widget.identity_overlay_mode = PlayerWidget.IdentityOverlayMode.BBOX
    image = QtGui.QImage(8, 8, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(60, 60, 60))
    widget._display_image(image, 0)
    frame_widget.resize(64, 64)

    frame_widget.grab()


def test_only_the_active_identity_is_emphasized_by_default():
    """The labeling GUI highlights the selected animal and fades the rest."""
    widget = PlayerWidget()
    frame_widget = widget._frame_widget
    frame_widget.set_active_identity(1)

    assert widget.emphasize_active_identity
    assert frame_widget.is_drawn_active(1)
    assert not frame_widget.is_drawn_active(0)

    widget.emphasize_active_identity = False
    assert frame_widget.is_drawn_active(0)


@pytest.mark.parametrize("emphasize", [True, False], ids=["emphasized", "not_emphasized"])
def test_bounding_boxes_follow_active_identity_emphasis(emphasize: bool) -> None:
    """Bounding boxes use the active color only for identities drawn as active.

    With emphasis off, as in the viewer, every box gets the active color instead of
    singling out identity 0.
    """
    widget = PlayerWidget()
    frame_widget = widget._frame_widget
    pose = mock.Mock(format_major_version=8, has_bounding_boxes=True, identities=[0, 1])
    pose.get_bounding_boxes.side_effect = lambda i: np.array(
        [[[10 + 40 * i, 10], [40 + 40 * i, 40]]], dtype=np.float64
    )
    pose.identity_index_to_display.side_effect = str
    frame_widget.set_pose(pose)
    frame_widget.set_active_identity(0)
    frame_widget.identity_overlay_mode = PlayerWidget.IdentityOverlayMode.BBOX
    widget.emphasize_active_identity = emphasize
    image = QtGui.QImage(100, 100, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(60, 60, 60))
    widget._display_image(image, 0)
    frame_widget.resize(100, 100)

    (overlay,) = frame_widget.findChildren(BoundingBoxOverlay)
    overlay._get_accent_color = mock.Mock(return_value=QtGui.QColor("red"))
    overlay._get_highlight_color = mock.Mock(return_value=QtGui.QColor("blue"))
    frame_widget.grab()

    overlay._get_accent_color.assert_called()
    assert overlay._get_highlight_color.called is emphasize
