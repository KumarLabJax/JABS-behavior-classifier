"""Tests for MainWindow shutdown and feature cache scan handling.

The methods under test are called unbound with lightweight stand-ins for ``self``,
so a full MainWindow (and its child widgets) never has to be constructed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

try:
    import jabs.ui.main_window.main_window as main_window_module
    from jabs.ui.main_window.main_window import MainWindow

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


def _scan_complete_stub(project, scan_thread, sender=None) -> SimpleNamespace:
    """Build a stub self for _feature_cache_scan_complete.

    Args:
        project: The project the window currently has open.
        scan_thread: The scan thread the window is tracking.
        sender: The thread whose signal is being delivered; defaults to
            ``scan_thread`` (the current scan).
    """
    return SimpleNamespace(
        _project=project,
        _feature_cache_scan_thread=scan_thread,
        sender=lambda: scan_thread if sender is None else sender,
    )


def test_quit_application_closes_window_before_quitting(monkeypatch):
    """Quit closes the window first so closeEvent's cleanup runs, then quits.

    Closing delivers the close event synchronously; the explicit quit is what
    ends the application even when another top-level window is still open.
    """
    calls = MagicMock()
    stub = SimpleNamespace(close=calls.close)
    monkeypatch.setattr(
        main_window_module,
        "QtWidgets",
        SimpleNamespace(QApplication=SimpleNamespace(quit=calls.quit)),
    )

    MainWindow.quit_application(stub)

    assert calls.mock_calls == [call.close(), call.quit()]


def test_feature_cache_scan_results_stored_on_project():
    """Scan results are handed to the project that was scanned."""
    project = MagicMock()
    stub = _scan_complete_stub(project, MagicMock())
    statuses = {"a.avi": MagicMock()}

    MainWindow._feature_cache_scan_complete(stub, project, statuses)

    project.set_feature_cache_status.assert_called_once_with(statuses)


def test_feature_cache_scan_results_discarded_for_closed_project():
    """Results arriving after another project was opened are dropped."""
    scanned_project = MagicMock()
    current_project = MagicMock()
    stub = _scan_complete_stub(current_project, MagicMock())

    MainWindow._feature_cache_scan_complete(stub, scanned_project, {"a.avi": MagicMock()})

    scanned_project.set_feature_cache_status.assert_not_called()
    current_project.set_feature_cache_status.assert_not_called()


def test_feature_cache_scan_results_discarded_when_superseded():
    """An older scan finishing after a newer one started must not write back.

    Its results describe the cache before the newer scan's starting point (for
    example before a training run computed features).
    """
    project = MagicMock()
    current_scan = MagicMock()
    older_scan = MagicMock()
    stub = _scan_complete_stub(project, current_scan, sender=older_scan)

    MainWindow._feature_cache_scan_complete(stub, project, {"a.avi": MagicMock()})

    project.set_feature_cache_status.assert_not_called()


def test_feature_cache_scan_thread_released_when_finished():
    """A finished scan thread is dropped and scheduled for deletion."""
    thread = MagicMock()
    stub = SimpleNamespace(_feature_cache_scan_thread=thread)

    MainWindow._feature_cache_scan_finished(stub, thread)

    assert stub._feature_cache_scan_thread is None
    thread.deleteLater.assert_called_once()


def test_finished_scan_thread_does_not_clear_a_newer_one():
    """An older thread finishing must not drop the reference to a running scan."""
    older = MagicMock()
    newer = MagicMock()
    stub = SimpleNamespace(_feature_cache_scan_thread=newer)

    MainWindow._feature_cache_scan_finished(stub, older)

    assert stub._feature_cache_scan_thread is newer
    older.deleteLater.assert_called_once()
    newer.deleteLater.assert_not_called()


def test_refresh_feature_cache_status_without_project_does_nothing(monkeypatch):
    """With no project open there is nothing to scan."""
    scan_thread = MagicMock()
    monkeypatch.setattr(main_window_module, "FeatureCacheScanThread", scan_thread)
    stub = SimpleNamespace(_project=None, _feature_cache_scan_thread=None)

    MainWindow.refresh_feature_cache_status(stub)

    scan_thread.assert_not_called()
