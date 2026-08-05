"""Tests for the background feature cache scan thread."""

from unittest import mock

import pytest

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.feature_cache_scan_thread import FeatureCacheScanThread

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
    """Ensure a QApplication exists, matching the other UI tests in this suite.

    A QCoreApplication would be enough for QThread, but creating one here would
    prevent widget tests later in the session from creating a QApplication.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_scan_emits_project_and_statuses(monkeypatch):
    """A successful scan emits the scanned project alongside its results."""
    project = mock.MagicMock()
    statuses = {"a.avi": mock.MagicMock()}
    monkeypatch.setattr(
        "jabs.ui.feature_cache_scan_thread.scan_project_feature_cache",
        lambda p, should_continue=None: statuses if p is project else None,
    )
    thread = FeatureCacheScanThread(project)
    received = []
    thread.scan_complete.connect(lambda p, s: received.append((p, s)))

    # run() directly rather than start(): the work is synchronous and this keeps
    # the test free of thread scheduling and event loop timing
    thread.run()

    assert received == [(project, statuses)]


def test_failed_scan_emits_nothing(monkeypatch):
    """A scan that raises is logged and emits no results."""

    def _raise(_project, should_continue=None):
        raise OSError("feature dir unreadable")

    monkeypatch.setattr("jabs.ui.feature_cache_scan_thread.scan_project_feature_cache", _raise)
    thread = FeatureCacheScanThread(mock.MagicMock())
    received = []
    thread.scan_complete.connect(lambda p, s: received.append((p, s)))

    thread.run()

    assert received == []


def test_termination_request_stops_scan_and_discards_results(monkeypatch):
    """Requesting termination stops the scan and suppresses partial results."""
    scanned = []

    def _scan(_project, should_continue=None):
        # mimic scan_project_feature_cache: check the predicate per video
        for video in ("a.avi", "b.avi", "c.avi"):
            if should_continue is not None and not should_continue():
                break
            scanned.append(video)
        return dict.fromkeys(scanned, mock.MagicMock())

    monkeypatch.setattr("jabs.ui.feature_cache_scan_thread.scan_project_feature_cache", _scan)
    thread = FeatureCacheScanThread(mock.MagicMock())
    received = []
    thread.scan_complete.connect(lambda p, s: received.append((p, s)))

    thread.request_termination()
    thread.run()

    assert scanned == []
    assert received == []
