from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    from PySide6.QtWidgets import QApplication

    import jabs.ui.main_window.menu_handlers as menu_handlers_module
    from jabs.ui.main_window.menu_handlers import _SETTINGS_EXPORT_FRAME_DIR, MenuHandlers

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


def test_export_frame_success(monkeypatch, handler_setup):
    """Export uses the suggested filename, persists the directory, and reports success."""
    handler, window, player = handler_setup
    pixmap = MagicMock()
    pixmap.save.return_value = True
    player.get_raw_frame.return_value = pixmap

    captured = {}

    def fake_get_save_file_name(parent, title, initial_path, file_filter, options=None):
        captured["title"] = title
        captured["initial_path"] = initial_path
        return "/tmp/video_frame000042.png", "PNG Images (*.png)"

    monkeypatch.setattr(
        menu_handlers_module.QtWidgets.QFileDialog,
        "getSaveFileName",
        fake_get_save_file_name,
    )
    monkeypatch.setattr(menu_handlers_module, "USE_NATIVE_FILE_DIALOG", True)

    handler.export_frame()

    assert captured["title"] == "Export Frame"
    assert captured["initial_path"] == "video_frame000042.png"
    pixmap.save.assert_called_once_with("/tmp/video_frame000042.png", "PNG")
    window._settings.setValue.assert_called_once_with(_SETTINGS_EXPORT_FRAME_DIR, "/tmp")
    window.display_status_message.assert_called_once_with(
        "Frame exported: /tmp/video_frame000042.png",
        5000,
    )


def test_export_frame_cancelled_does_not_save(monkeypatch, handler_setup):
    """Cancelling the dialog leaves the filesystem and settings unchanged."""
    handler, window, player = handler_setup
    pixmap = MagicMock()
    player.get_raw_frame.return_value = pixmap

    monkeypatch.setattr(
        menu_handlers_module.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("", ""),
    )
    monkeypatch.setattr(menu_handlers_module, "USE_NATIVE_FILE_DIALOG", True)

    handler.export_frame()

    pixmap.save.assert_not_called()
    window._settings.setValue.assert_not_called()
    window.display_status_message.assert_not_called()


def test_export_frame_no_frame_uses_warning_message(monkeypatch, handler_setup):
    """Missing frames report the warning text as the dialog message, not the title."""
    handler, _, player = handler_setup
    player.get_raw_frame.return_value = None
    warning = MagicMock()

    monkeypatch.setattr(menu_handlers_module.MessageDialog, "warning", warning)

    handler.export_frame()

    warning.assert_called_once_with(handler.window, message="No frame available to export.")


def test_export_frame_appends_extension(monkeypatch, handler_setup):
    """A missing .png suffix is appended before exporting."""
    handler, _, player = handler_setup
    pixmap = MagicMock()
    pixmap.save.return_value = True
    player.get_raw_frame.return_value = pixmap

    monkeypatch.setattr(
        menu_handlers_module.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("/tmp/custom_name", "PNG Images (*.png)"),
    )
    monkeypatch.setattr(menu_handlers_module, "USE_NATIVE_FILE_DIALOG", True)

    handler.export_frame()

    pixmap.save.assert_called_once_with("/tmp/custom_name.png", "PNG")


def test_export_frame_write_failure_uses_error_message(monkeypatch, handler_setup):
    """Export failures report the error text as the dialog message and do not persist state."""
    handler, window, player = handler_setup
    pixmap = MagicMock()
    pixmap.save.return_value = False
    player.get_raw_frame.return_value = pixmap
    error = MagicMock()

    monkeypatch.setattr(
        menu_handlers_module.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("/tmp/video_frame000042.png", "PNG Images (*.png)"),
    )
    monkeypatch.setattr(menu_handlers_module.MessageDialog, "error", error)
    monkeypatch.setattr(menu_handlers_module, "USE_NATIVE_FILE_DIALOG", True)

    handler.export_frame()

    error.assert_called_once_with(
        handler.window,
        message="Failed to export frame to:\n/tmp/video_frame000042.png",
    )
    window._settings.setValue.assert_not_called()
    window.display_status_message.assert_not_called()


def test_export_frame_restores_existing_directory(monkeypatch, tmp_path, handler_setup):
    """The last successful export directory is reused when it still exists."""
    handler, window, player = handler_setup
    pixmap = MagicMock()
    pixmap.save.return_value = True
    player.get_raw_frame.return_value = pixmap
    window._settings.value.return_value = str(tmp_path)

    captured = {}

    def fake_get_save_file_name(parent, title, initial_path, file_filter, options=None):
        captured["initial_path"] = initial_path
        return str(tmp_path / "chosen_frame.png"), "PNG Images (*.png)"

    monkeypatch.setattr(
        menu_handlers_module.QtWidgets.QFileDialog,
        "getSaveFileName",
        fake_get_save_file_name,
    )
    monkeypatch.setattr(menu_handlers_module, "USE_NATIVE_FILE_DIALOG", True)

    handler.export_frame()

    assert captured["initial_path"] == str(tmp_path / "video_frame000042.png")


def test_export_frame_missing_directory_falls_back(monkeypatch, handler_setup):
    """A missing saved directory falls back to the bare suggested filename."""
    handler, window, player = handler_setup
    pixmap = MagicMock()
    pixmap.save.return_value = True
    player.get_raw_frame.return_value = pixmap
    window._settings.value.return_value = "/path/that/does/not/exist"

    captured = {}

    def fake_get_save_file_name(parent, title, initial_path, file_filter, options=None):
        captured["initial_path"] = initial_path
        return "/tmp/video_frame000042.png", "PNG Images (*.png)"

    monkeypatch.setattr(
        menu_handlers_module.QtWidgets.QFileDialog,
        "getSaveFileName",
        fake_get_save_file_name,
    )
    monkeypatch.setattr(menu_handlers_module, "USE_NATIVE_FILE_DIALOG", True)

    handler.export_frame()

    assert captured["initial_path"] == "video_frame000042.png"
