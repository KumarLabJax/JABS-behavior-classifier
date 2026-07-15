from unittest.mock import MagicMock

import pytest

try:
    from PySide6 import QtWidgets
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication

    from jabs.ui.main_window.video_list_widget import (
        _EXCLUDED_ROLE,
        VideoListDockWidget,
        _VideoListWidget,
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


def test_text_pen_color_dims_excluded_rows_in_all_states():
    """Excluded rows use the disabled palette color whether selected or not."""
    palette = QPalette()
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("red"))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor("green")
    )
    palette.setColor(
        QPalette.ColorGroup.Active, QPalette.ColorRole.HighlightedText, QColor("blue")
    )
    pen = _VideoListWidget.HighlightTextDelegate._text_pen_color

    # unselected excluded -> dimmed normal text
    assert pen(palette, selected=False, excluded=True).name() == QColor("red").name()
    # selected excluded -> dimmed highlighted text (still readable on highlight bg)
    assert pen(palette, selected=True, excluded=True).name() == QColor("green").name()
    # selected included -> normal highlighted text
    assert pen(palette, selected=True, excluded=False).name() == QColor("blue").name()


def _patch_menu(monkeypatch, chooser):
    """Replace QMenu with a non-modal subclass whose exec() delegates to chooser.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        chooser: callable taking the menu's action list and returning the action
            to treat as "chosen" (or None for no selection).
    """

    class _NonModalMenu(QtWidgets.QMenu):
        def exec(self, *_args, **_kwargs):
            return chooser(self.actions())

    monkeypatch.setattr(QtWidgets, "QMenu", _NonModalMenu)


def _classify_action(actions):
    """Return the "Classify Video" action from a list of menu actions."""
    for action in actions:
        if action.text() == "Classify Video":
            return action
    raise AssertionError("no 'Classify Video' action in menu")


@pytest.mark.parametrize("available", [True, False], ids=["ready", "not-ready"])
def test_classify_action_enabled_reflects_availability(monkeypatch, available):
    """The Classify Video action is enabled only when a classifier is available."""
    widget = VideoListDockWidget()
    widget.set_project(_mock_project(["a.avi"], excluded=set()))
    widget.set_classify_available(available)
    item = _item_for(widget, "a.avi")
    monkeypatch.setattr(widget._file_list, "itemAt", lambda _pos: item)

    captured = {}

    def chooser(actions):
        captured["enabled"] = _classify_action(actions).isEnabled()
        return None  # choose nothing

    _patch_menu(monkeypatch, chooser)
    widget._show_context_menu(QPoint(0, 0))

    assert captured["enabled"] is available


def test_choosing_classify_action_emits_request(monkeypatch):
    """Choosing Classify Video emits classify_video_requested with the video name."""
    widget = VideoListDockWidget()
    widget.set_project(_mock_project(["a.avi"], excluded=set()))
    widget.set_classify_available(True)
    item = _item_for(widget, "a.avi")
    monkeypatch.setattr(widget._file_list, "itemAt", lambda _pos: item)
    _patch_menu(monkeypatch, _classify_action)

    requested = []
    widget.classify_video_requested.connect(requested.append)
    widget._show_context_menu(QPoint(0, 0))

    assert requested == ["a.avi"]
