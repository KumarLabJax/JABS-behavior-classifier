from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    from PySide6.QtWidgets import QApplication

    import jabs.ui.main_window.menu_handlers as menu_handlers_module
    from jabs.ui.main_window.menu_handlers import (
        _SETTINGS_EXPORT_FRAME_DIR,
        _SETTINGS_EXPORT_OVERLAY,
        MenuHandlers,
    )

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
    """Ensure a QApplication exists for UI-related tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def handler_setup():
    """Create a MenuHandlers instance with a lightweight fake window."""
    player = SimpleNamespace(
        get_raw_frame=MagicMock(),
        get_overlay_frame=MagicMock(),
        current_frame=42,
        current_video_path=Path("video.mp4"),
    )
    window = SimpleNamespace(
        _central_widget=SimpleNamespace(_player_widget=player),
        _settings=MagicMock(),
        display_status_message=MagicMock(),
    )
    window._settings.value.return_value = ""
    return MenuHandlers(window), window, player


def _patch_export_dialog(
    monkeypatch,
    *,
    accepted: bool = True,
    selected: tuple[str, ...] = ("/tmp/video_frame000042.png",),
    overlay_checked: bool = False,
):
    """Replace the QFileDialog and QCheckBox used by export_frame with controllable mocks.

    Returns the mock dialog instance so tests can assert on dialog interactions
    (e.g. ``selectFile`` / ``setDirectory``).
    """
    fake_qfiledialog = MagicMock()
    dialog = fake_qfiledialog.return_value
    dialog.exec.return_value = (
        fake_qfiledialog.DialogCode.Accepted if accepted else fake_qfiledialog.DialogCode.Rejected
    )
    dialog.selectedFiles.return_value = list(selected)
    dialog.layout.return_value = None  # not a QGridLayout -> checkbox is not inserted
    monkeypatch.setattr(menu_handlers_module.QtWidgets, "QFileDialog", fake_qfiledialog)

    fake_checkbox = MagicMock()
    fake_checkbox.isChecked.return_value = overlay_checked
    monkeypatch.setattr(
        menu_handlers_module.QtWidgets, "QCheckBox", MagicMock(return_value=fake_checkbox)
    )
    return dialog


def test_export_frame_success(monkeypatch, handler_setup):
    """Export uses the suggested filename, persists the directory, and reports success."""
    handler, window, player = handler_setup
    pixmap = MagicMock()
    pixmap.save.return_value = True
    player.get_raw_frame.return_value = pixmap

    dialog = _patch_export_dialog(monkeypatch, selected=("/tmp/video_frame000042.png",))

    handler.export_frame()

    dialog.selectFile.assert_called_once_with("video_frame000042.png")
    player.get_raw_frame.assert_called_once_with(42)
    pixmap.save.assert_called_once_with("/tmp/video_frame000042.png", "PNG")
    player.get_overlay_frame.assert_not_called()
    window._settings.setValue.assert_any_call(_SETTINGS_EXPORT_FRAME_DIR, "/tmp")
    window.display_status_message.assert_called_once_with(
        "Frame exported: /tmp/video_frame000042.png",
        5000,
    )


def test_export_frame_cancelled_does_not_save(monkeypatch, handler_setup):
    """Cancelling the dialog leaves the filesystem and settings unchanged."""
    handler, window, player = handler_setup
    pixmap = MagicMock()
    player.get_raw_frame.return_value = pixmap

    _patch_export_dialog(monkeypatch, accepted=False)

    handler.export_frame()

    player.get_raw_frame.assert_not_called()
    pixmap.save.assert_not_called()
    window._settings.setValue.assert_not_called()
    window.display_status_message.assert_not_called()


def test_export_frame_no_frame_uses_warning_message(monkeypatch, handler_setup):
    """Missing frames report the warning text as the dialog message, not the title."""
    handler, _, player = handler_setup
    player.get_raw_frame.return_value = None
    warning = MagicMock()

    _patch_export_dialog(monkeypatch)
    monkeypatch.setattr(menu_handlers_module.MessageDialog, "warning", warning)

    handler.export_frame()

    player.get_raw_frame.assert_called_once_with(42)
    warning.assert_called_once_with(handler.window, message="No frame available to export.")


def test_export_frame_appends_extension(monkeypatch, handler_setup):
    """A missing .png suffix is appended before exporting."""
    handler, _, player = handler_setup
    pixmap = MagicMock()
    pixmap.save.return_value = True
    player.get_raw_frame.return_value = pixmap

    _patch_export_dialog(monkeypatch, selected=("/tmp/custom_name",))

    handler.export_frame()

    pixmap.save.assert_called_once_with("/tmp/custom_name.png", "PNG")


def test_export_frame_write_failure_uses_error_message(monkeypatch, handler_setup):
    """Export failures report the error text as the dialog message and do not persist state."""
    handler, window, player = handler_setup
    pixmap = MagicMock()
    pixmap.save.return_value = False
    player.get_raw_frame.return_value = pixmap
    error = MagicMock()

    _patch_export_dialog(monkeypatch, selected=("/tmp/video_frame000042.png",))
    monkeypatch.setattr(menu_handlers_module.MessageDialog, "error", error)

    handler.export_frame()

    error.assert_called_once_with(
        handler.window,
        message="Failed to export frame to:\n/tmp/video_frame000042.png",
    )
    window._settings.setValue.assert_not_called()
    window.display_status_message.assert_not_called()


def test_export_frame_saves_overlay_copy(monkeypatch, handler_setup):
    """With the checkbox enabled, a second -overlay.png copy is written and reported."""
    handler, window, player = handler_setup
    pixmap = MagicMock()
    pixmap.save.return_value = True
    player.get_raw_frame.return_value = pixmap
    overlay_pixmap = MagicMock()
    overlay_pixmap.save.return_value = True
    player.get_overlay_frame.return_value = overlay_pixmap

    _patch_export_dialog(
        monkeypatch, selected=("/tmp/video_frame000042.png",), overlay_checked=True
    )

    handler.export_frame()

    pixmap.save.assert_called_once_with("/tmp/video_frame000042.png", "PNG")
    player.get_overlay_frame.assert_called_once_with(42)
    overlay_pixmap.save.assert_called_once_with("/tmp/video_frame000042-overlay.png", "PNG")
    window._settings.setValue.assert_any_call(_SETTINGS_EXPORT_OVERLAY, True)
    window.display_status_message.assert_called_once_with(
        "Frame exported: /tmp/video_frame000042.png (+ overlay copy)",
        5000,
    )


def test_export_frame_restores_existing_directory(monkeypatch, tmp_path, handler_setup):
    """The last successful export directory is reused when it still exists."""
    handler, window, player = handler_setup
    pixmap = MagicMock()
    pixmap.save.return_value = True
    player.get_raw_frame.return_value = pixmap
    window._settings.value.return_value = str(tmp_path)

    dialog = _patch_export_dialog(monkeypatch, selected=(str(tmp_path / "chosen_frame.png"),))

    handler.export_frame()

    dialog.setDirectory.assert_called_once_with(str(tmp_path))
    dialog.selectFile.assert_called_once_with("video_frame000042.png")


def test_export_frame_missing_directory_falls_back(monkeypatch, handler_setup):
    """A missing saved directory is not applied to the dialog."""
    handler, window, player = handler_setup
    pixmap = MagicMock()
    pixmap.save.return_value = True
    player.get_raw_frame.return_value = pixmap
    window._settings.value.return_value = "/path/that/does/not/exist"

    dialog = _patch_export_dialog(monkeypatch, selected=("/tmp/video_frame000042.png",))

    handler.export_frame()

    dialog.setDirectory.assert_not_called()
    dialog.selectFile.assert_called_once_with("video_frame000042.png")


def test_handle_select_all_delegates_to_central_widget():
    """handle_select_all() calls select_all() on the central widget."""
    central = SimpleNamespace(select_all=MagicMock())
    window = SimpleNamespace(_central_widget=central)
    handler = MenuHandlers(window)
    handler.handle_select_all()
    central.select_all.assert_called_once_with()


def test_handle_select_current_bout_delegates_to_central_widget():
    """handle_select_current_bout() calls select_current_bout() on the central widget."""
    central = SimpleNamespace(select_current_bout=MagicMock())
    window = SimpleNamespace(_central_widget=central)
    handler = MenuHandlers(window)
    handler.handle_select_current_bout()
    central.select_current_bout.assert_called_once_with()


# --- export_overlay_video -------------------------------------------------------


class _FakeV6Pose:
    """Stands in for a v6+ pose object, with segmentation optionally present."""

    def __init__(self, has_segmentation: bool) -> None:
        self.has_segmentation = has_segmentation


@pytest.fixture
def video_export_setup(handler_setup, monkeypatch):
    """Handler wired for export_overlay_video, with the thread class replaced.

    Returns ``(handlers, window, player, thread_cls, thread)`` where ``thread`` is
    the instance the handler constructed.
    """
    handlers, window, player = handler_setup
    player.pose_est = _FakeV6Pose(has_segmentation=True)
    player.num_frames = 100
    player.current_video_path = Path("/videos/clip.avi")

    thread = MagicMock()
    thread_cls = MagicMock(return_value=thread)
    monkeypatch.setattr(menu_handlers_module, "VideoExportThread", thread_cls)
    monkeypatch.setattr(menu_handlers_module.QtWidgets, "QProgressDialog", MagicMock())
    return handlers, window, player, thread_cls, thread


def test_export_overlay_video_starts_thread_with_chosen_path(video_export_setup, monkeypatch):
    """The selected path, pose object and segmentation choice reach the thread."""
    handlers, _window, player, thread_cls, thread = video_export_setup
    _patch_export_dialog(monkeypatch, selected=("/tmp/out.mp4",), overlay_checked=True)

    handlers.export_overlay_video()

    args = thread_cls.call_args.args
    assert args[0] == player.current_video_path
    assert args[1] == Path("/tmp/out.mp4")
    assert args[2] is player.pose_est
    assert args[3] is True  # draw_segmentation
    thread.start.assert_called_once()


def test_export_overlay_video_appends_extension(video_export_setup, monkeypatch):
    """A filename without .mp4 still produces an mp4 path."""
    handlers, _window, _player, thread_cls, _thread = video_export_setup
    _patch_export_dialog(monkeypatch, selected=("/tmp/no_extension",))

    handlers.export_overlay_video()

    assert thread_cls.call_args.args[1] == Path("/tmp/no_extension.mp4")


def test_export_overlay_video_cancelled_dialog_starts_nothing(video_export_setup, monkeypatch):
    """Dismissing the save dialog must not start an export."""
    handlers, _window, _player, thread_cls, _thread = video_export_setup
    _patch_export_dialog(monkeypatch, accepted=False)

    handlers.export_overlay_video()

    thread_cls.assert_not_called()


def test_export_overlay_video_without_video_warns(handler_setup, monkeypatch):
    """No loaded video is a warning, not a crash."""
    handlers, _window, player = handler_setup
    player.pose_est = None
    player.current_video_path = None
    warning = MagicMock()
    monkeypatch.setattr(menu_handlers_module.MessageDialog, "warning", warning)

    handlers.export_overlay_video()

    warning.assert_called_once()


@pytest.mark.parametrize(
    ("has_segmentation", "expect_enabled"),
    [(True, True), (False, False)],
    ids=["v6-with-segmentation", "v6-without-segmentation"],
)
def test_export_overlay_video_checkbox_reflects_segmentation_availability(
    video_export_setup, monkeypatch, has_segmentation: bool, expect_enabled: bool
):
    """Segmentation is optional even in v6+, so the box tracks the data, not the version."""
    handlers, _window, player, _thread_cls, _thread = video_export_setup
    player.pose_est = _FakeV6Pose(has_segmentation=has_segmentation)

    fake_qfiledialog = MagicMock()
    dialog = fake_qfiledialog.return_value
    dialog.exec.return_value = fake_qfiledialog.DialogCode.Accepted
    dialog.selectedFiles.return_value = ["/tmp/out.mp4"]
    dialog.layout.return_value = None
    monkeypatch.setattr(menu_handlers_module.QtWidgets, "QFileDialog", fake_qfiledialog)
    checkbox = MagicMock()
    checkbox.isChecked.return_value = has_segmentation
    monkeypatch.setattr(
        menu_handlers_module.QtWidgets, "QCheckBox", MagicMock(return_value=checkbox)
    )

    handlers.export_overlay_video()

    if expect_enabled:
        checkbox.setEnabled.assert_not_called()
    else:
        checkbox.setEnabled.assert_called_once_with(False)
        assert "segmentation" in checkbox.setToolTip.call_args.args[0].lower()


def test_export_overlay_video_thread_deletes_itself_on_finished(video_export_setup, monkeypatch):
    """Deletion is driven by `finished`, not by the result callbacks.

    The result callbacks run while ``run()`` is still on the stack; destroying a
    QThread that has not finished aborts the process.
    """
    handlers, _window, _player, _thread_cls, thread = video_export_setup
    _patch_export_dialog(monkeypatch, selected=("/tmp/out.mp4",))

    handlers.export_overlay_video()

    assert thread.finished.connect.call_count == 2
    connected = [c.args[0] for c in thread.finished.connect.call_args_list]
    assert thread.deleteLater in connected, "deleteLater must be driven by finished"


def test_export_overlay_video_preserves_dots_in_filenames(video_export_setup, monkeypatch):
    """A dotted filename must not be rewritten.

    `Path.with_suffix()` replaces everything after the last dot, so a name like
    `session_2024.09.01` would silently become `session_2024.09.mp4` - writing to a
    different file than the user asked for.
    """
    handlers, _window, _player, thread_cls, _thread = video_export_setup
    _patch_export_dialog(monkeypatch, selected=("/tmp/session_2024.09.01",))

    handlers.export_overlay_video()

    assert thread_cls.call_args.args[1] == Path("/tmp/session_2024.09.01.mp4")


def test_export_overlay_video_leaves_an_mp4_name_alone(video_export_setup, monkeypatch):
    """A name that already ends in .mp4 is used as-is, not doubled up."""
    handlers, _window, _player, thread_cls, _thread = video_export_setup
    _patch_export_dialog(monkeypatch, selected=("/tmp/already.mp4",))

    handlers.export_overlay_video()

    assert thread_cls.call_args.args[1] == Path("/tmp/already.mp4")
