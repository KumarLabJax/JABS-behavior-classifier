"""Tests for MainWindow shutdown and feature cache scan handling.

The methods under test are called unbound with lightweight stand-ins for ``self``,
so a full MainWindow (and its child widgets) never has to be constructed.
"""

import logging
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


def _refresh_stub(scan_thread=None, pending=False, has_project=True) -> SimpleNamespace:
    """Build a stub self for the scan start/finish bookkeeping.

    Args:
        scan_thread: The scan thread the window is tracking, if any.
        pending: Whether a rescan is already queued.
        has_project: Whether a project is open.
    """
    return SimpleNamespace(
        _project=MagicMock() if has_project else None,
        _feature_cache_scan_thread=scan_thread,
        _feature_cache_scan_pending=pending,
        _start_feature_cache_scan=MagicMock(),
    )


def test_feature_cache_scan_thread_released_when_finished():
    """A finished scan thread is dropped and scheduled for deletion."""
    thread = MagicMock()
    stub = _refresh_stub(scan_thread=thread)

    MainWindow._feature_cache_scan_finished(stub, thread)

    assert stub._feature_cache_scan_thread is None
    thread.deleteLater.assert_called_once()


def test_finished_scan_thread_does_not_clear_a_newer_one():
    """An older thread finishing must not drop the reference to a running scan."""
    older = MagicMock()
    newer = MagicMock()
    stub = _refresh_stub(scan_thread=newer)

    MainWindow._feature_cache_scan_finished(stub, older)

    assert stub._feature_cache_scan_thread is newer
    older.deleteLater.assert_called_once()
    newer.deleteLater.assert_not_called()


def test_refresh_feature_cache_status_without_project_does_nothing():
    """With no project open there is nothing to scan."""
    stub = _refresh_stub(has_project=False)

    MainWindow.refresh_feature_cache_status(stub)

    stub._start_feature_cache_scan.assert_not_called()


def test_start_feature_cache_scan_tracks_and_starts_the_thread(monkeypatch):
    """The scan thread is built for the open project, tracked, and started."""
    scan_thread_class = MagicMock()
    monkeypatch.setattr(main_window_module, "FeatureCacheScanThread", scan_thread_class)
    project = MagicMock()
    stub = SimpleNamespace(
        _project=project,
        _feature_cache_scan_thread=None,
        _feature_cache_scan_complete=MagicMock(),
        _feature_cache_scan_finished=MagicMock(),
    )

    MainWindow._start_feature_cache_scan(stub)

    scan_thread_class.assert_called_once_with(project, parent=stub)
    thread = scan_thread_class.return_value
    thread.start.assert_called_once()
    assert stub._feature_cache_scan_thread is thread


def test_refresh_starts_a_scan_when_none_is_running():
    """With no scan in flight, a refresh starts one immediately."""
    stub = _refresh_stub()

    MainWindow.refresh_feature_cache_status(stub)

    stub._start_feature_cache_scan.assert_called_once()
    assert stub._feature_cache_scan_pending is False


def test_refresh_during_a_scan_queues_instead_of_starting_a_second():
    """A refresh while scanning stops that scan and queues a replacement.

    Keeps a single scan in flight, so repeated refreshes cannot pile up concurrent
    passes over the feature directory or leave untracked threads at shutdown.
    """
    running = MagicMock()
    stub = _refresh_stub(scan_thread=running)

    MainWindow.refresh_feature_cache_status(stub)

    stub._start_feature_cache_scan.assert_not_called()
    running.request_termination.assert_called_once()
    assert stub._feature_cache_scan_pending is True


def test_queued_rescan_starts_when_the_running_scan_finishes():
    """The queued refresh runs once the scan it superseded has finished."""
    thread = MagicMock()
    stub = _refresh_stub(scan_thread=thread, pending=True)

    MainWindow._feature_cache_scan_finished(stub, thread)

    assert stub._feature_cache_scan_thread is None
    thread.deleteLater.assert_called_once()
    stub._start_feature_cache_scan.assert_called_once()
    assert stub._feature_cache_scan_pending is False


def test_no_rescan_started_when_none_was_queued():
    """A scan finishing with nothing queued does not start another."""
    thread = MagicMock()
    stub = _refresh_stub(scan_thread=thread, pending=False)

    MainWindow._feature_cache_scan_finished(stub, thread)

    stub._start_feature_cache_scan.assert_not_called()


def test_queued_rescan_dropped_when_no_project_is_open():
    """A queued refresh is abandoned if the project was closed meanwhile."""
    thread = MagicMock()
    stub = _refresh_stub(scan_thread=thread, pending=True, has_project=False)

    MainWindow._feature_cache_scan_finished(stub, thread)

    stub._start_feature_cache_scan.assert_not_called()


def test_stop_feature_cache_scan_without_a_scan_is_a_noop():
    """Nothing to stop when no scan is running."""
    stub = _refresh_stub()

    MainWindow._stop_feature_cache_scan(stub)  # must not raise


def test_stop_feature_cache_scan_asks_the_thread_to_stop():
    """A responsive scan is asked to stop and is not forced."""
    thread = MagicMock()
    thread.wait.return_value = True
    stub = _refresh_stub(scan_thread=thread)

    MainWindow._stop_feature_cache_scan(stub)

    thread.request_termination.assert_called_once()
    thread.wait.assert_called_once()
    thread.terminate.assert_not_called()


def test_stop_feature_cache_scan_forces_an_unresponsive_thread():
    """A scan stuck in a filesystem call is terminated rather than waited on forever.

    Leaving it running would let the window be destroyed with a live child thread,
    which aborts the process; blocking indefinitely would make JABS unquittable on
    a wedged mount.
    """
    thread = MagicMock()
    # the cooperative stop times out, the forced stop works
    thread.wait.side_effect = [False, True]
    stub = _refresh_stub(scan_thread=thread)

    MainWindow._stop_feature_cache_scan(stub)

    thread.request_termination.assert_called_once()
    thread.terminate.assert_called_once()
    assert thread.wait.call_count == 2


def test_stop_feature_cache_scan_logs_when_it_cannot_stop_the_thread(caplog):
    """A thread that survives even termination is reported rather than hidden."""
    thread = MagicMock()
    thread.wait.return_value = False
    stub = _refresh_stub(scan_thread=thread)

    with caplog.at_level(logging.ERROR, logger=main_window_module.__name__):
        MainWindow._stop_feature_cache_scan(stub)

    thread.terminate.assert_called_once()
    assert "could not be stopped" in caplog.text
