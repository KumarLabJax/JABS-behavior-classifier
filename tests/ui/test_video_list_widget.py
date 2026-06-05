from unittest.mock import MagicMock

import pytest

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from jabs.ui.main_window.video_list_widget import _EXCLUDED_ROLE, VideoListDockWidget

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


def _mock_project(videos, excluded):
    """Build a mock project exposing videos and per-video exclusion state."""
    project = MagicMock()
    project.video_manager.videos = videos
    project.settings_manager.is_video_excluded.side_effect = lambda v: v in excluded
    return project


def _item_for(widget, video):
    """Return the list item whose UserRole matches the given video name."""
    file_list = widget._file_list
    for i in range(file_list.count()):
        item = file_list.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == video:
            return item
    raise AssertionError(f"no list item for {video!r}")


def test_set_project_tags_excluded_rows():
    """set_project marks each row's excluded role from project settings."""
    widget = VideoListDockWidget()
    widget.set_project(_mock_project(["a.avi", "b.avi"], excluded={"b.avi"}))

    assert _item_for(widget, "a.avi").data(_EXCLUDED_ROLE) is False
    assert _item_for(widget, "b.avi").data(_EXCLUDED_ROLE) is True


def test_set_video_excluded_persists_and_updates_row():
    """Toggling exclusion persists via settings_manager and updates the row role."""
    project = _mock_project(["a.avi"], excluded=set())
    widget = VideoListDockWidget()
    widget.set_project(project)
    item = _item_for(widget, "a.avi")

    widget._set_video_excluded(item, "a.avi", True)

    project.settings_manager.set_video_excluded.assert_called_once_with("a.avi", True)
    assert item.data(_EXCLUDED_ROLE) is True
