from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jabs.core.enums import CacheFormat
from jabs.io.feature_cache import IdentityCacheInfo
from jabs.project import VideoFeatureCacheStatus

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


def _cache_status(video, window_sizes=frozenset({5})) -> VideoFeatureCacheStatus:
    """Build a feature cache status with one cached identity."""
    return VideoFeatureCacheStatus(
        video=video,
        cache_dir=Path("/features") / Path(video).stem,
        identity_caches=(
            IdentityCacheInfo(
                directory=Path("/features") / Path(video).stem / "0",
                identity=0,
                cache_format=CacheFormat.PARQUET,
                feature_version=17,
                pose_hash="hash",
                num_frames=100,
                distance_scale_factor=None,
                window_sizes=window_sizes,
                per_frame_present=True,
                size_bytes=10,
            ),
        ),
        current_feature_version=17,
        expected_identity_count=1,
    )


def test_feature_cache_status_rescans_the_video():
    """Get Info re-scans the video so the status it shows is current."""
    project = _mock_project(["a.avi"], excluded=set())
    refreshed = _cache_status("a.avi")
    project.refresh_feature_cache_status.return_value = refreshed
    widget = VideoListDockWidget()
    widget.set_project(project)

    assert widget._feature_cache_status("a.avi") is refreshed
    project.refresh_feature_cache_status.assert_called_once_with("a.avi")


def test_feature_cache_status_falls_back_to_stored_status_on_error():
    """A failed re-scan falls back to the status already stored on the project."""
    project = _mock_project(["a.avi"], excluded=set())
    stored = _cache_status("a.avi")
    project.refresh_feature_cache_status.side_effect = OSError("unreadable")
    project.feature_cache_status = {"a.avi": stored}
    widget = VideoListDockWidget()
    widget.set_project(project)

    assert widget._feature_cache_status("a.avi") is stored


def test_feature_cache_status_without_project():
    """With no project loaded there is no status to report."""
    assert VideoListDockWidget()._feature_cache_status("a.avi") is None


def _visible_videos(widget):
    """Return the video names of the rows currently visible in the list."""
    file_list = widget._file_list
    return [
        file_list.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(file_list.count())
        if not file_list.item(i).isHidden()
    ]


def test_filter_keeps_selection_when_current_video_still_matches():
    """Filtering must not cost the user their place when the selection still matches.

    Regression: the deselect guard checked only whether the item *was* the
    current one, so every keystroke in the filter box cleared the selection even
    when the selected video was still visible.
    """
    widget = VideoListDockWidget()
    widget.set_project(_mock_project(["walk_01.avi", "walk_02.avi", "rear_01.avi"], set()))
    widget.select_video("walk_01.avi", suppress_event=True)

    widget._filter_list("walk")

    assert _visible_videos(widget) == ["walk_01.avi", "walk_02.avi"]
    current = widget._file_list.currentItem()
    assert current is not None
    assert current.data(Qt.ItemDataRole.UserRole) == "walk_01.avi"


def test_filter_deselects_current_video_when_it_is_filtered_out():
    """A current item that no longer matches is deselected, so no hidden row stays current."""
    widget = VideoListDockWidget()
    widget.set_project(_mock_project(["walk_01.avi", "rear_01.avi"], set()))
    widget.select_video("rear_01.avi", suppress_event=True)

    widget._filter_list("walk")

    assert _visible_videos(widget) == ["walk_01.avi"]
    assert widget._file_list.currentItem() is None


def test_filter_does_not_emit_selection_change():
    """Deselecting a filtered-out item must not be reported as a user selection."""
    widget = VideoListDockWidget()
    widget.set_project(_mock_project(["walk_01.avi", "rear_01.avi"], set()))
    widget.select_video("rear_01.avi", suppress_event=True)

    emitted = []
    widget.selectionChanged.connect(emitted.append)
    widget._filter_list("walk")

    assert emitted == []


def test_clearing_filter_restores_all_rows():
    """Clearing the filter text un-hides every row."""
    widget = VideoListDockWidget()
    videos = ["walk_01.avi", "walk_02.avi", "rear_01.avi"]
    widget.set_project(_mock_project(videos, set()))

    widget._filter_list("walk")
    widget._filter_list("")

    assert _visible_videos(widget) == sorted(videos)


def test_filter_with_no_current_item_is_safe():
    """Filtering with nothing selected must not raise."""
    widget = VideoListDockWidget()
    widget.set_project(_mock_project(["walk_01.avi", "rear_01.avi"], set()))
    widget._file_list.setCurrentItem(None)

    widget._filter_list("walk")

    assert _visible_videos(widget) == ["walk_01.avi"]
    assert widget._file_list.currentItem() is None
