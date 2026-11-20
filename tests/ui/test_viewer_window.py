"""Tests for the jabs-viewer main window."""

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

try:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication

    from jabs.ui.player_widget import PlayerWidget
    from jabs.ui.viewer import viewer_window
    from jabs.ui.viewer.frame_size_dialog import FrameSizeDialog
    from jabs.ui.viewer.overlay_export import OverlayChoice
    from jabs.ui.viewer.pose_loader_thread import PoseLoaderThread
    from jabs.ui.viewer.viewer_window import ViewerWindow

    class _SynchronousPoseLoader(QObject):
        """PoseLoaderThread stand-in that opens the pose file as soon as it is started.

        Lets tests assert on the window right after ``open_video`` returns, as if the
        background load had already finished.
        """

        pose_loaded = Signal(object)
        load_error = Signal(Exception)
        finished = Signal()

        def __init__(
            self,
            pose_path: Path,
            loader: Callable[[Path], object],
            parent: QObject | None = None,
        ) -> None:
            super().__init__(parent)
            self.pose_path = pose_path
            self._loader = loader

        def start(self) -> None:
            try:
                pose_est = self._loader(self.pose_path)
            except Exception as e:
                self.load_error.emit(e)
            else:
                self.pose_loaded.emit(pose_est)
            self.finished.emit()

        def wait(self) -> bool:
            return True

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


@pytest.fixture(autouse=True)
def synchronous_pose_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open pose files synchronously; tests of the background load restore the thread."""
    monkeypatch.setattr(viewer_window, "PoseLoaderThread", _SynchronousPoseLoader)


def _process_events_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    """Run the Qt event loop until ``condition()`` is true, failing after ``timeout``."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the event loop"
        QApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture
def window() -> "ViewerWindow":
    """A viewer window whose player does not need a real video file.

    The player's ``load_video`` is replaced on the instance (patching attributes on the
    Qt wrapper class itself crashes PySide6), and it reports a 100 frame video.
    """
    window = ViewerWindow(app_name="test")
    window._player.load_video = mock.Mock()
    window._player._video_stream = mock.Mock(num_frames=100)
    window._video_frame_count = mock.Mock(return_value=100)
    return window


def _fake_pose(
    has_segmentation: bool = True, has_bounding_boxes: bool = False, has_landmarks: bool = False
) -> mock.Mock:
    """Build a stand-in pose file with 100 frames.

    It is a v8 pose file when it has bounding boxes, and v6 otherwise. With landmarks it
    records the arena corners.
    """
    pose = mock.Mock()
    pose.identities = [0, 1, 2]
    pose.num_frames = 100
    pose.has_segmentation = has_segmentation
    pose.has_bounding_boxes = has_bounding_boxes
    pose.format_major_version = 8 if has_bounding_boxes else 6
    pose.static_objects = {"corners": np.zeros((4, 2))} if has_landmarks else {}
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


@pytest.mark.parametrize("pose_frames", [90, 110], ids=["shorter", "longer"])
def test_pose_file_with_different_frame_count_warns_and_opens_video_anyway(
    monkeypatch, window, pose_frames: int
) -> None:
    """A pose file that doesn't have one frame per video frame is not drawn over the video.

    The overlays index the pose by video frame, so a shorter pose file would fail past its
    last frame.
    """
    pose = _fake_pose()
    pose.num_frames = pose_frames
    _patch_pose_lookup(monkeypatch, POSE_PATH, pose)
    message_dialog = mock.Mock()
    monkeypatch.setattr(viewer_window, "MessageDialog", message_dialog)

    window.open_video(VIDEO_PATH)

    message_dialog.warning.assert_called_once()
    window._player.load_video.assert_called_once_with(VIDEO_PATH, None)
    assert window.pose_est is None
    assert not window._pose_action.isEnabled()
    assert "does not match the video" in window.statusBar().currentMessage()


def test_unreadable_video_reports_error_and_leaves_nothing_open(monkeypatch, window) -> None:
    """A video that fails to open is reported and the window shows nothing open."""
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


def test_loading_pose_file_shows_status_without_override_cursor(
    monkeypatch: pytest.MonkeyPatch, window: "ViewerWindow"
) -> None:
    """Opening a pose file reports progress in the status bar, not with a busy cursor.

    On macOS, Qt's wait cursor is built from an image, and that conversion crashes the
    app under Qt 6.11.
    """
    seen: dict[str, object] = {}

    def open_pose(path: Path) -> mock.Mock:
        seen["message"] = window.statusBar().currentMessage()
        seen["cursor"] = QApplication.overrideCursor()
        return _fake_pose()

    monkeypatch.setattr(viewer_window, "get_pose_path", lambda _video_path: POSE_PATH)
    monkeypatch.setattr(viewer_window, "open_pose_file", open_pose)

    window.open_video(VIDEO_PATH)

    assert seen["message"] == f"Loading pose file {POSE_PATH.name}..."
    assert seen["cursor"] is None
    assert "Loading pose file" not in window.statusBar().currentMessage()


@pytest.mark.parametrize("fail", ["pose", "video"])
def test_failed_load_does_not_leave_loading_message(
    monkeypatch: pytest.MonkeyPatch, window: "ViewerWindow", fail: str
) -> None:
    """When opening stops on an error, the status bar no longer says it is loading."""
    window.statusBar().showMessage("previous status")
    monkeypatch.setattr(viewer_window, "MessageDialog", mock.Mock())
    if fail == "pose":
        open_pose = _patch_pose_lookup(monkeypatch, POSE_PATH)
        open_pose.side_effect = OSError("unreadable")
        window.open_pose_file(POSE_PATH)
    else:
        _patch_pose_lookup(monkeypatch, POSE_PATH, _fake_pose())
        window._player.load_video.side_effect = OSError("unreadable")
        window.open_video(VIDEO_PATH)

    # whatever was open was released before the load, so its status no longer applies
    assert window.statusBar().currentMessage() == viewer_window.OPEN_PROMPT


def test_bounding_box_control_follows_the_pose_file(monkeypatch, window) -> None:
    """Only a pose file with bounding boxes (v8+) enables the bounding box toggle."""
    assert not window._bbox_action.isEnabled()

    _patch_pose_lookup(monkeypatch, POSE_PATH, _fake_pose(has_bounding_boxes=False))
    window.open_video(VIDEO_PATH)
    assert not window._bbox_action.isEnabled()

    _patch_pose_lookup(monkeypatch, POSE_PATH, _fake_pose(has_bounding_boxes=True))
    window.open_video(VIDEO_PATH)
    assert window._bbox_action.isEnabled()


def test_bounding_box_toggle_sets_bounding_box_overlay() -> None:
    """The bounding box toggle drives the player's bounding box overlay."""
    window = ViewerWindow(app_name="test")
    frame_widget = window._player._frame_widget
    assert frame_widget.bounding_box_overlay_enabled
    assert frame_widget.identity_overlay_mode == PlayerWidget.IdentityOverlayMode.BBOX

    window._bbox_action.setChecked(False)
    assert not frame_widget.bounding_box_overlay_enabled

    window._bbox_action.setChecked(True)
    assert frame_widget.bounding_box_overlay_enabled


def test_landmark_control_follows_the_pose_file(monkeypatch, window) -> None:
    """Only a pose file that records arena landmarks enables the landmark toggle."""
    assert not window._landmarks_action.isEnabled()

    _patch_pose_lookup(monkeypatch, POSE_PATH, _fake_pose(has_landmarks=False))
    window.open_video(VIDEO_PATH)
    assert not window._landmarks_action.isEnabled()

    _patch_pose_lookup(monkeypatch, POSE_PATH, _fake_pose(has_landmarks=True))
    window.open_video(VIDEO_PATH)
    assert window._landmarks_action.isEnabled()


def test_landmark_toggle_sets_landmark_overlay() -> None:
    """The landmark toggle drives the player's landmark overlay, which starts on."""
    window = ViewerWindow(app_name="test")
    frame_widget = window._player._frame_widget
    assert frame_widget.landmark_overlay_enabled

    window._landmarks_action.setChecked(False)
    assert not frame_widget.landmark_overlay_enabled

    window._landmarks_action.setChecked(True)
    assert frame_widget.landmark_overlay_enabled


def test_export_actions_follow_what_is_open(monkeypatch, window) -> None:
    """Frames can be exported from any open file; a video export needs a pose to draw."""
    assert not window._export_frame_action.isEnabled()
    assert not window._export_video_action.isEnabled()

    _patch_pose_lookup(monkeypatch, None)
    window.open_video(VIDEO_PATH)
    assert window._export_frame_action.isEnabled()
    assert not window._export_video_action.isEnabled()

    _patch_pose_lookup(monkeypatch, POSE_PATH, _fake_pose())
    window.open_video(VIDEO_PATH)
    assert window._export_video_action.isEnabled()


def test_export_passes_the_video_and_displayed_overlays(monkeypatch, window) -> None:
    """Exports start from the open video and the overlays the View menu has on."""
    pose = _fake_pose(has_bounding_boxes=True, has_landmarks=True)
    _patch_pose_lookup(monkeypatch, POSE_PATH, pose)
    window.open_video(VIDEO_PATH)
    # load_video is mocked, so set the path it would have set
    window._player._video_path = VIDEO_PATH
    window._segmentation_action.setChecked(False)
    window._exporter = mock.Mock()

    window.export_frame()
    window.export_video()

    source, overlays = window._exporter.export_frame.call_args.args
    assert source.path == VIDEO_PATH
    assert source.pose_est is pose
    assert not source.pose_only
    assert overlays == OverlayChoice(
        pose=True, segmentation=False, bounding_boxes=True, landmarks=True
    )
    assert window._exporter.export_video.call_args.args == (source, overlays)


def test_pose_only_export_draws_over_fresh_blank_frames(monkeypatch, window) -> None:
    """A pose file on its own exports over new frames at the current frame size."""
    _open_pose_only(monkeypatch, window)
    window.set_frame_size(640, 480)
    window._exporter = mock.Mock()

    window.export_video()

    source, _overlays = window._exporter.export_video.call_args.args
    assert source.path == POSE_PATH
    assert source.pose_only
    assert source.frames.dimensions == (640, 480)
    assert source.frames.num_frames == 100
    assert source.frames is not window._player._video_stream


def test_export_does_nothing_with_nothing_open() -> None:
    """With no file open there is nothing to export."""
    window = ViewerWindow(app_name="test")
    window._exporter = mock.Mock()

    window.export_frame()
    window.export_video()

    window._exporter.export_frame.assert_not_called()
    window._exporter.export_video.assert_not_called()


@pytest.fixture
def blocking_pose_load(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[threading.Event, mock.Mock]]:
    """Load pose files on the real background thread, held until the test releases it."""
    monkeypatch.setattr(viewer_window, "PoseLoaderThread", PoseLoaderThread)
    release = threading.Event()
    pose = _fake_pose()

    def open_pose(_path: Path) -> mock.Mock:
        assert release.wait(5), "test never released the pose load"
        return pose

    monkeypatch.setattr(viewer_window, "get_pose_path", lambda _video_path: POSE_PATH)
    monkeypatch.setattr(viewer_window, "open_pose_file", open_pose)
    yield release, pose
    release.set()


def test_pose_file_opens_in_the_background(window, blocking_pose_load) -> None:
    """The window stays usable while a pose file opens, then shows the video and pose."""
    release, pose = blocking_pose_load

    window.open_video(VIDEO_PATH)

    # open_video returned while the pose file is still loading
    loader = window._pose_loader
    assert loader is not None and loader.isRunning()
    window._player.load_video.assert_not_called()
    assert window.statusBar().currentMessage() == f"Loading pose file {POSE_PATH.name}..."

    release.set()
    _process_events_until(lambda: window._pose_loader is None)

    window._player.load_video.assert_called_once_with(VIDEO_PATH, pose)
    assert window.pose_est is pose
    assert window._pose_action.isEnabled()


def test_open_requests_are_ignored_while_a_pose_file_loads(window, blocking_pose_load) -> None:
    """A second file can't be opened until the first one's pose file has loaded."""
    release, pose = blocking_pose_load
    window._player.load_blank_frames = mock.Mock()

    window.open_video(VIDEO_PATH)
    loader = window._pose_loader
    window.open_video(Path("/data/other.mp4"))
    window.open_pose_file(POSE_PATH)

    assert window._pose_loader is loader
    release.set()
    _process_events_until(lambda: window._pose_loader is None)
    window._player.load_video.assert_called_once_with(VIDEO_PATH, pose)
    window._player.load_blank_frames.assert_not_called()


def test_busy_dialog_shows_for_a_slow_load_and_closes_after(
    monkeypatch, window, blocking_pose_load
) -> None:
    """A pose file that takes a while to open shows a busy dialog until it has loaded."""
    release, _pose = blocking_pose_load
    dialog = mock.Mock()
    monkeypatch.setattr(viewer_window, "create_progress_dialog", mock.Mock(return_value=dialog))
    monkeypatch.setattr(viewer_window, "BUSY_DIALOG_DELAY_MS", 0)

    window.open_video(VIDEO_PATH)
    _process_events_until(lambda: dialog.show.called)
    dialog.hide.assert_not_called()

    release.set()
    _process_events_until(lambda: window._pose_loader is None)
    # hide(), since the progress dialog ignores close events
    dialog.hide.assert_called_once()


def test_quick_load_does_not_flash_the_busy_dialog(monkeypatch, window) -> None:
    """A pose file that opens before the delay never builds the busy dialog."""
    create_dialog = mock.Mock()
    monkeypatch.setattr(viewer_window, "create_progress_dialog", create_dialog)
    _patch_pose_lookup(monkeypatch, POSE_PATH, _fake_pose())

    window.open_video(VIDEO_PATH)
    deadline = time.monotonic() + 2 * viewer_window.BUSY_DIALOG_DELAY_MS / 1000
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)

    create_dialog.assert_not_called()


def _open_pose_only_then_block(monkeypatch: pytest.MonkeyPatch, window: "ViewerWindow") -> None:
    """Open a pose file on its own at once, then restore the blocking background load."""
    blocking_open = viewer_window.open_pose_file
    monkeypatch.setattr(viewer_window, "PoseLoaderThread", _SynchronousPoseLoader)
    _open_pose_only(monkeypatch, window)
    monkeypatch.setattr(viewer_window, "PoseLoaderThread", PoseLoaderThread)
    monkeypatch.setattr(viewer_window, "open_pose_file", blocking_open)
    monkeypatch.setattr(viewer_window, "get_pose_path", lambda _video_path: POSE_PATH)


def test_closing_mid_load_waits_without_blocking(window: "ViewerWindow") -> None:
    """Closing mid-load is put off rather than blocking on the thread.

    Destroying a running QThread aborts, so the window can't close yet, but waiting
    on it would freeze the window for the rest of the load.
    """
    loader = mock.Mock(pose_path=POSE_PATH)
    window._pose_loader = loader
    window._exporter = mock.Mock()

    assert window.close() is False

    loader.wait.assert_not_called()
    window._exporter.shutdown.assert_not_called()
    assert "Closing once" in window.statusBar().currentMessage()


def test_close_requested_mid_load_discards_the_pose_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    window: "ViewerWindow",
    blocking_pose_load: tuple[threading.Event, mock.Mock],
) -> None:
    """Once the load finishes, the window closes without showing the file or any dialog."""
    release, _pose = blocking_pose_load
    message_dialog = mock.Mock()
    monkeypatch.setattr(viewer_window, "MessageDialog", message_dialog)
    window._exporter = mock.Mock()

    window.open_video(VIDEO_PATH)
    window.close()
    release.set()
    _process_events_until(lambda: window._exporter.shutdown.called)

    window._player.load_video.assert_not_called()
    message_dialog.warning.assert_not_called()
    assert window.pose_est is None


def test_open_and_export_actions_are_disabled_while_a_pose_file_loads(
    monkeypatch: pytest.MonkeyPatch,
    window: "ViewerWindow",
    blocking_pose_load: tuple[threading.Event, mock.Mock],
) -> None:
    """Nothing can be opened or exported mid-load, so no request is silently dropped."""
    release, _pose = blocking_pose_load
    _open_pose_only_then_block(monkeypatch, window)
    assert window._export_video_action.isEnabled()

    window.open_video(VIDEO_PATH)

    assert not window._open_action.isEnabled()
    assert not window._open_pose_action.isEnabled()
    assert not window._export_frame_action.isEnabled()
    assert not window._export_video_action.isEnabled()

    release.set()
    _process_events_until(lambda: window._pose_loader is None)

    assert window._open_action.isEnabled()
    assert window._open_pose_action.isEnabled()
    assert window._export_video_action.isEnabled()


def test_the_open_file_is_released_before_the_next_pose_file_loads(
    monkeypatch: pytest.MonkeyPatch,
    window: "ViewerWindow",
    blocking_pose_load: tuple[threading.Event, mock.Mock],
) -> None:
    """The old pose is not kept in memory alongside the one that is loading."""
    release, pose = blocking_pose_load
    _open_pose_only_then_block(monkeypatch, window)
    assert window.pose_est is not None

    window.open_video(VIDEO_PATH)

    assert window.pose_est is None
    assert window._player.pose_est is None
    assert window.windowTitle() == "test"

    release.set()
    _process_events_until(lambda: window._pose_loader is None)
    assert window.pose_est is pose


def test_video_frame_count_is_read_off_the_gui_thread(
    window: "ViewerWindow", blocking_pose_load: tuple[threading.Event, mock.Mock]
) -> None:
    """The frame count check, slow for a video on a network share, runs on the loader."""
    release, _pose = blocking_pose_load
    threads: list[threading.Thread] = []

    def frame_count(_path: Path) -> int:
        threads.append(threading.current_thread())
        return 100

    window._video_frame_count = frame_count

    window.open_video(VIDEO_PATH)
    release.set()
    _process_events_until(lambda: window._pose_loader is None)

    assert threads and threads[0] is not threading.main_thread()


def test_unreadable_video_frame_count_still_opens_with_the_pose(
    monkeypatch: pytest.MonkeyPatch, window: "ViewerWindow"
) -> None:
    """A frame count that can't be read is left for opening the video to report."""
    pose = _fake_pose()
    _patch_pose_lookup(monkeypatch, POSE_PATH, pose)
    window._video_frame_count = mock.Mock(side_effect=OSError("unreadable"))

    window.open_video(VIDEO_PATH)

    window._player.load_video.assert_called_once_with(VIDEO_PATH, pose)
