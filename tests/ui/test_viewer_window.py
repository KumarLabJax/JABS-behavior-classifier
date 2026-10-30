"""Tests for the jabs-viewer main window."""

from pathlib import Path
from unittest import mock

import pytest

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.player_widget import PlayerWidget
    from jabs.ui.viewer import viewer_window
    from jabs.ui.viewer.frame_size_dialog import FrameSizeDialog
    from jabs.ui.viewer.viewer_window import ViewerWindow

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)

VIDEO_PATH = Path("/data/video.mp4")
POSE_PATH = Path("/data/video_pose_est_v6.h5")


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def window() -> "ViewerWindow":
    """A viewer window whose player does not need a real video file.

    The player's ``load_video`` is replaced on the instance (patching attributes on the
    Qt wrapper class itself crashes PySide6), and it reports a 100 frame video.
    """
    window = ViewerWindow(app_name="test")
    window._player.load_video = mock.Mock()
    window._player._video_stream = mock.Mock(num_frames=100)
    return window


def _fake_pose(has_segmentation: bool = True) -> mock.Mock:
    """Build a stand-in pose file with 100 frames."""
    pose = mock.Mock()
    pose.identities = [0, 1, 2]
    pose.num_frames = 100
    pose.has_segmentation = has_segmentation
    return pose


def _patch_pose_lookup(monkeypatch, pose_path: Path | None, pose=None) -> mock.Mock:
    """Patch pose file discovery and opening in the viewer module."""

    def get_pose_path(_video_path: Path) -> Path:
        if pose_path is None:
            raise ValueError("Video does not have pose file")
        return pose_path

    open_pose = mock.Mock(return_value=pose)
    monkeypatch.setattr(viewer_window, "get_pose_path", get_pose_path)
    monkeypatch.setattr(viewer_window, "open_pose_file", open_pose)
    return open_pose


def test_overlay_controls_disabled_before_a_video_is_opened() -> None:
    """With nothing open there is no pose to overlay."""
    window = ViewerWindow(app_name="test")

    assert not window._pose_action.isEnabled()
    assert not window._segmentation_action.isEnabled()


def test_video_without_pose_file_opens_without_overlays(monkeypatch, window) -> None:
    """A video with no pose file beside it plays with the overlay controls disabled."""
    open_pose = _patch_pose_lookup(monkeypatch, pose_path=None)

    window.open_video(VIDEO_PATH)

    window._player.load_video.assert_called_once_with(VIDEO_PATH, None)
    open_pose.assert_not_called()
    assert window.pose_est is None
    assert not window._pose_action.isEnabled()
    assert not window._segmentation_action.isEnabled()


def test_video_with_pose_file_enables_overlays(monkeypatch, window) -> None:
    """The pose file beside the video is opened and passed to the player."""
    pose = _fake_pose()
    open_pose = _patch_pose_lookup(monkeypatch, POSE_PATH, pose)

    window.open_video(VIDEO_PATH)

    open_pose.assert_called_once_with(POSE_PATH)
    window._player.load_video.assert_called_once_with(VIDEO_PATH, pose)
    assert window.pose_est is pose
    assert window._pose_action.isEnabled()
    assert window._segmentation_action.isEnabled()


def test_segmentation_control_disabled_when_pose_has_no_segmentation(monkeypatch, window) -> None:
    """Pose files without segmentation data leave only the pose toggle enabled."""
    _patch_pose_lookup(monkeypatch, POSE_PATH, _fake_pose(has_segmentation=False))

    window.open_video(VIDEO_PATH)

    assert window._pose_action.isEnabled()
    assert not window._segmentation_action.isEnabled()


def test_unreadable_pose_file_warns_and_opens_video_anyway(monkeypatch, window) -> None:
    """A pose file that fails to open is reported, and the video still plays."""
    open_pose = _patch_pose_lookup(monkeypatch, POSE_PATH)
    open_pose.side_effect = OSError("corrupt file")
    message_dialog = mock.Mock()
    monkeypatch.setattr(viewer_window, "MessageDialog", message_dialog)

    window.open_video(VIDEO_PATH)

    message_dialog.warning.assert_called_once()
    window._player.load_video.assert_called_once_with(VIDEO_PATH, None)
    assert window.pose_est is None


def test_unreadable_video_reports_error_and_keeps_previous_state(monkeypatch, window) -> None:
    """A video that fails to open is reported and the window state is not updated."""
    _patch_pose_lookup(monkeypatch, POSE_PATH, _fake_pose())
    window._player.load_video.side_effect = OSError("bad video")
    message_dialog = mock.Mock()
    monkeypatch.setattr(viewer_window, "MessageDialog", message_dialog)

    window.open_video(VIDEO_PATH)

    message_dialog.error.assert_called_once()
    assert window.pose_est is None
    assert window.windowTitle() == "test"


def test_pose_toggle_sets_pose_overlay_mode() -> None:
    """The pose toggle switches between drawing every identity's pose and none."""
    window = ViewerWindow(app_name="test")
    assert window._player.pose_overlay_mode == PlayerWidget.PoseOverlayMode.ALL

    window._pose_action.setChecked(False)
    assert window._player.pose_overlay_mode == PlayerWidget.PoseOverlayMode.NONE

    window._pose_action.setChecked(True)
    assert window._player.pose_overlay_mode == PlayerWidget.PoseOverlayMode.ALL


def test_segmentation_toggle_sets_segmentation_overlay() -> None:
    """The segmentation toggle drives the player's segmentation overlay."""
    window = ViewerWindow(app_name="test")
    frame_widget = window._player._frame_widget
    assert frame_widget.segmentation_overlay_enabled

    window._segmentation_action.setChecked(False)
    assert not frame_widget.segmentation_overlay_enabled


def test_every_identity_is_drawn_without_active_emphasis() -> None:
    """The viewer has no selected animal, so no identity is drawn faded."""
    window = ViewerWindow(app_name="test")
    frame_widget = window._player._frame_widget

    assert not window._player.emphasize_active_identity
    assert all(frame_widget.is_drawn_active(identity) for identity in range(4))


def test_pose_file_opens_on_a_blank_background(monkeypatch, window) -> None:
    """A pose file opened on its own plays over solid frames sized to fit the pose."""
    pose = _fake_pose()
    pose.fps = 25
    open_pose = _patch_pose_lookup(monkeypatch, None, pose)
    monkeypatch.setattr(viewer_window, "estimate_frame_size", lambda _pose: (800, 700))
    window._player.load_blank_frames = mock.Mock()

    window.open_pose_file(POSE_PATH)

    open_pose.assert_called_once_with(POSE_PATH)
    frames, loaded_pose = window._player.load_blank_frames.call_args.args
    assert loaded_pose is pose
    assert frames.num_frames == 100
    assert frames.fps == 25
    assert frames.dimensions == (800, 700)
    assert window.pose_est is pose
    assert window._pose_action.isEnabled()
    assert window._segmentation_action.isEnabled()
    assert window.windowTitle() == f"test - {POSE_PATH.name}"


def test_unreadable_pose_file_opened_on_its_own_reports_error(monkeypatch, window) -> None:
    """With no video to fall back on, a pose file that fails to open is an error."""
    open_pose = _patch_pose_lookup(monkeypatch, None)
    open_pose.side_effect = ValueError("not a pose file")
    message_dialog = mock.Mock()
    monkeypatch.setattr(viewer_window, "MessageDialog", message_dialog)
    window._player.load_blank_frames = mock.Mock()

    window.open_pose_file(POSE_PATH)

    message_dialog.error.assert_called_once()
    window._player.load_blank_frames.assert_not_called()
    assert window.pose_est is None


@pytest.mark.parametrize(
    ("path", "opener"),
    [
        (Path("/data/video.mp4"), "open_video"),
        (Path("/data/video.AVI"), "open_video"),
        (Path("/data/video_pose_est_v6.h5"), "open_pose_file"),
    ],
    ids=["mp4", "avi", "pose"],
)
def test_open_file_picks_the_opener_by_extension(window, path: Path, opener: str) -> None:
    """The command-line file argument may be either a video or a pose file."""
    window.open_video = mock.Mock()
    window.open_pose_file = mock.Mock()

    window.open_file(path)

    getattr(window, opener).assert_called_once_with(path)


def _open_pose_only(monkeypatch, window, estimate: tuple[int, int] = (800, 700)) -> mock.Mock:
    """Open a fake pose file on its own, returning the player's load_blank_frames spy."""
    pose = _fake_pose()
    pose.fps = 30
    _patch_pose_lookup(monkeypatch, None, pose)
    monkeypatch.setattr(viewer_window, "estimate_frame_size", lambda _pose: estimate)
    window._player.load_blank_frames = mock.Mock()
    window.open_pose_file(POSE_PATH)
    return window._player.load_blank_frames


def test_frame_size_can_only_be_set_for_a_pose_file_without_video(monkeypatch, window) -> None:
    """With a video the frame size comes from the video, so the action is disabled."""
    assert not window._frame_size_action.isEnabled()

    _open_pose_only(monkeypatch, window)
    assert window._frame_size_action.isEnabled()

    window.open_video(VIDEO_PATH)
    assert not window._frame_size_action.isEnabled()


def test_set_frame_size_reloads_frames_and_keeps_position(monkeypatch, window) -> None:
    """Resizing replaces the blank frames and returns to the frame being viewed."""
    load_blank_frames = _open_pose_only(monkeypatch, window)
    window._player.seek_to_frame = mock.Mock()
    # current_frame is the slider position; no player thread is running to react to it
    window._player._position_slider.setMaximum(100)
    window._player._position_slider.setValue(42)

    window.set_frame_size(1024, 768)

    frames, _pose = load_blank_frames.call_args.args
    assert frames.dimensions == (1024, 768)
    window._player.seek_to_frame.assert_called_once_with(42)
    assert "1024x768 set frame size" in window.statusBar().currentMessage()


def test_set_frame_size_does_nothing_for_a_video(monkeypatch, window) -> None:
    """A video's frames are the video's own size."""
    _patch_pose_lookup(monkeypatch, POSE_PATH, _fake_pose())
    window.open_video(VIDEO_PATH)
    window._player.load_blank_frames = mock.Mock()

    window.set_frame_size(1024, 768)

    window._player.load_blank_frames.assert_not_called()


def test_frame_size_dialog_starts_at_current_size_and_can_restore_estimate() -> None:
    """The dialog is seeded with the size in use and offers the estimate back."""
    dialog = FrameSizeDialog(current_size=(1024, 768), estimated_size=(800, 700))
    assert dialog.frame_size == (1024, 768)

    dialog._use_estimate()

    assert dialog.frame_size == (800, 700)
