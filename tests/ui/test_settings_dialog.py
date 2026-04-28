from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.settings_dialog.settings_dialog import _OverlapCheckThread

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


def test_overlap_check_thread_emits_complete() -> None:
    """Successful overlap checks emit the conflicting video list."""
    project = SimpleNamespace(
        get_overlapping_behavior_label_videos=MagicMock(return_value=["video1.avi"])
    )
    emitted_results = []
    emitted_errors = []

    thread = _OverlapCheckThread(project)
    thread.check_complete.connect(emitted_results.append)
    thread.check_failed.connect(emitted_errors.append)

    thread.run()

    assert emitted_results == [["video1.avi"]]
    assert emitted_errors == []


def test_overlap_check_thread_emits_error_on_exception() -> None:
    """Failed overlap checks emit an error instead of silently ending the thread."""
    expected_error = RuntimeError("pose load failed")
    project = SimpleNamespace(
        get_overlapping_behavior_label_videos=MagicMock(side_effect=expected_error)
    )
    emitted_results = []
    emitted_errors = []

    thread = _OverlapCheckThread(project)
    thread.check_complete.connect(emitted_results.append)
    thread.check_failed.connect(emitted_errors.append)

    thread.run()

    assert emitted_results == []
    assert emitted_errors == [expected_error]
