from pathlib import Path
from unittest import mock

import numpy as np
import pytest

try:
    from PySide6 import QtCore, QtGui
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
    pose = mock.Mock(has_bounding_boxes=True, identities=[0, 1])
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


@pytest.mark.parametrize("has_boxes", [True, False], ids=["boxes", "no-boxes"])
def test_bounding_box_overlay_follows_the_capability(has_boxes: bool) -> None:
    """Boxes are drawn exactly when the pose has BOUNDING_BOXES; no version is consulted."""
    widget = PlayerWidget()
    frame_widget = widget._frame_widget
    pose = mock.Mock(has_bounding_boxes=has_boxes, identities=[0])
    pose.get_bounding_boxes.return_value = np.array([[[10, 10], [40, 40]]], dtype=np.float32)
    pose.identity_index_to_display.side_effect = str
    frame_widget.set_pose(pose)
    frame_widget.identity_overlay_mode = PlayerWidget.IdentityOverlayMode.BBOX
    image = QtGui.QImage(100, 100, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(60, 60, 60))
    widget._display_image(image, 0)
    frame_widget.resize(100, 100)

    frame_widget.grab()

    assert pose.get_bounding_boxes.called is has_boxes


def test_reset_stops_the_player_thread_and_closes_the_stream():
    """Reset tears down playback, so a failed load can't leave a thread reading a cleared stream."""
    widget = PlayerWidget()
    thread = mock.Mock()
    stream = mock.Mock()
    widget._player_thread = thread
    widget._video_stream = stream

    widget.reset()

    thread.stop_playback.assert_called()
    thread.wait.assert_called()
    stream.close.assert_called_once()
    assert widget._player_thread is None
    assert widget._video_stream is None


def _widget_with_mock_thread(num_frames: int = 1000) -> tuple["PlayerWidget", mock.Mock]:
    """A player whose slider is enabled and whose player thread is a mock."""
    widget = PlayerWidget()
    thread = mock.Mock()
    widget._player_thread = thread
    widget._video_stream = mock.Mock(num_frames=num_frames)
    widget._video_stream.get_frame_time.return_value = "00:00:00"
    widget._position_slider.setMaximum(num_frames - 1)
    widget._position_slider.setEnabled(True)
    return widget, thread


def test_fast_slider_changes_seek_only_to_the_latest_position(qapp) -> None:
    """A burst of slider values, as from a fast drag, decodes one frame, not one per value."""
    widget, thread = _widget_with_mock_thread()

    for value in range(10, 500, 7):
        widget._position_slider.setValue(value)
    thread.seek.assert_not_called()

    qapp.processEvents()

    assert [c.args[0] for c in thread.seek.call_args_list] == [last := max(range(10, 500, 7))]
    assert widget.current_frame == last


def test_dragging_while_playing_seeks(qapp) -> None:
    """A drag pauses playback until release, and seeks as it goes."""
    widget, thread = _widget_with_mock_thread()
    widget._playing = True
    widget._position_slider.setSliderDown(True)

    widget._position_slider.setValue(300)
    qapp.processEvents()

    # pressing seeks to where the slider was, then the drag seeks to where it went
    assert [c.args[0] for c in thread.seek.call_args_list] == [0, 300]


def test_playback_moving_the_slider_does_not_seek(qapp) -> None:
    """During playback the player thread moves the slider, which must not seek back."""
    widget, thread = _widget_with_mock_thread()
    widget._playing = True

    widget._set_position(120)
    qapp.processEvents()

    thread.seek.assert_not_called()


def test_slider_release_seeks_immediately_and_drops_pending_seek(qapp) -> None:
    """Releasing the slider shows its final frame at once, without a stale seek after it."""
    widget, thread = _widget_with_mock_thread()
    widget._position_slider.setValue(200)

    widget._position_slider_release()
    qapp.processEvents()

    assert [c.args[0] for c in thread.seek.call_args_list] == [200]


def test_step_to_frame_seeks_once(qapp) -> None:
    """Stepping to a frame seeks once, not once for the step and again for the slider."""
    widget, thread = _widget_with_mock_thread()

    widget.seek_to_frame(42)
    qapp.processEvents()

    thread.seek.assert_called_once_with(42)


def test_is_playing_or_pending_is_false_when_idle(qapp) -> None:
    """A player with a loaded video that isn't playing reports nothing in progress."""
    widget, _ = _widget_with_mock_thread()

    assert not widget.is_playing_or_pending


def test_is_playing_or_pending_covers_the_delay_before_range_playback(qapp) -> None:
    """play_range starts the thread after a delay; callers must still see playback as active."""
    widget, thread = _widget_with_mock_thread()

    widget.play_range(10, 20)

    thread.start.assert_not_called()
    assert widget.is_playing_or_pending

    widget.stop()

    assert not widget.is_playing_or_pending


def test_is_playing_or_pending_is_true_while_playing(qapp) -> None:
    """Ordinary playback counts as playing."""
    widget, thread = _widget_with_mock_thread()

    widget.toggle_play()

    thread.start.assert_called_once()
    assert widget.is_playing_or_pending


def _seek_targets(thread: mock.Mock) -> list[int]:
    """Frames the mock player thread was asked to seek to, in order."""
    return [c.args[0] for c in thread.seek.call_args_list]


def test_play_range_shows_the_range_as_playing_before_it_starts(qapp) -> None:
    """During the start delay the controls already show playback, at the range start."""
    widget, _ = _widget_with_mock_thread()

    widget.play_range(10, 20)

    assert widget._play_button.isChecked()
    assert not widget._next_frame_button.isEnabled()
    assert not widget._previous_frame_button.isEnabled()
    assert widget.current_frame == 10


def test_play_range_drops_a_queued_slider_seek(qapp) -> None:
    """A seek queued just before play_range must not move playback off the range start.

    This is how play_bout_clip lost its lead-in frames: it seeks to the bout start,
    then plays a range starting a few frames earlier.
    """
    widget, thread = _widget_with_mock_thread()
    widget.seek_to_frame(100)

    widget.play_range(90, 150)
    qapp.processEvents()

    assert _seek_targets(thread)[-1] == 90


def test_toggle_play_during_range_delay_cancels_the_range(qapp) -> None:
    """Play/Pause shows Pause during the delay, so pressing it stops instead of playing."""
    widget, thread = _widget_with_mock_thread()
    widget.play_range(10, 20)

    widget.toggle_play()

    thread.start.assert_not_called()
    assert not widget.is_playing_or_pending
    assert not widget._play_button.isChecked()
    assert widget._next_frame_button.isEnabled()


def test_seeks_during_range_delay_are_ignored_like_during_playback(qapp) -> None:
    """Frame steps and wheel/keyboard slider changes don't move a pending range."""
    widget, thread = _widget_with_mock_thread()
    widget.play_range(10, 20)

    widget.seek_to_frame(500)
    widget._position_slider.setValue(600)
    qapp.processEvents()

    assert _seek_targets(thread) == [10]


def test_slider_click_during_range_delay_cancels_the_range(qapp) -> None:
    """Clicking the slider is a deliberate seek: it cancels the pending range and seeks there."""
    widget, thread = _widget_with_mock_thread()
    widget.play_range(10, 20)
    with QtCore.QSignalBlocker(widget._position_slider):
        widget._position_slider.setValue(300)

    widget._position_slider_clicked()
    widget._position_slider_release()

    assert not widget.is_playing_or_pending
    thread.start.assert_not_called()
    assert _seek_targets(thread)[-1] == 300


def test_failed_playback_start_resets_the_player(qapp) -> None:
    """If the range can't start, the player goes back to idle and forgets the range."""
    widget, _ = _widget_with_mock_thread()
    widget.play_range(10, 20)
    widget._video_stream = None

    widget._start_player_thread()

    assert not widget.is_playing_or_pending
    assert widget._playback_range is None
    assert not widget._play_button.isChecked()
