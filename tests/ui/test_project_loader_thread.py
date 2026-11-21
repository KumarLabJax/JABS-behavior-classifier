"""Tests for the background project loader thread."""

from __future__ import annotations

from unittest import mock

import pytest

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.project_loader_thread import ProjectLoaderThread

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
    """Ensure a QApplication exists, matching the other UI tests in this suite."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _loader(monkeypatch: pytest.MonkeyPatch, project: mock.MagicMock) -> ProjectLoaderThread:
    """A loader whose Project constructor returns ``project``."""
    monkeypatch.setattr("jabs.ui.project_loader_thread.Project", lambda *a, **k: project)
    return ProjectLoaderThread("/tmp/project")


def test_poses_are_prepared_before_the_project_is_loaded(monkeypatch):
    """Conversion runs in the loader and reports progress before project_loaded fires."""
    events: list[object] = []
    project = mock.MagicMock()

    def prepare(progress):
        events.append("prepare")
        progress(1, 2)
        progress(2, 2)

    project.prepare_poses.side_effect = prepare
    thread = _loader(monkeypatch, project)
    thread.conversion_progress.connect(lambda done, total: events.append((done, total)))
    thread.project_loaded.connect(lambda: events.append("loaded"))

    # run() directly rather than start(): the work is synchronous and this keeps
    # the test free of thread scheduling and event loop timing
    thread.run()

    assert events == ["prepare", (1, 2), (2, 2), "loaded"]
    assert thread.project is project


def test_a_failed_conversion_step_still_loads_the_project(monkeypatch):
    """Converting ahead is an optimization: if it fails, the project still opens."""
    project = mock.MagicMock()
    project.prepare_poses.side_effect = RuntimeError("pool broke")
    thread = _loader(monkeypatch, project)
    loaded: list[bool] = []
    errors: list[Exception] = []
    thread.project_loaded.connect(lambda: loaded.append(True))
    thread.load_error.connect(errors.append)

    thread.run()

    assert loaded == [True]
    assert errors == []
