from pathlib import Path

import pytest

try:
    from PySide6 import QtGui
    from PySide6.QtWidgets import QApplication

    from jabs.ui.player_widget import PlayerWidget

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
