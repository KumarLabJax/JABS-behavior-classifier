"""Tests for jabs-viewer's background pose file loader."""

from pathlib import Path
from unittest import mock

import pytest

try:
    from jabs.ui.viewer.pose_loader_thread import PoseLoaderThread

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)

POSE_PATH = Path("/data/video_pose_est_v6.h5")


def test_emits_the_opened_pose() -> None:
    """A pose file that opens is emitted to the GUI thread."""
    pose = object()
    loader = mock.Mock(return_value=pose)
    thread = PoseLoaderThread(POSE_PATH, loader)
    loaded, errors = mock.Mock(), mock.Mock()
    thread.pose_loaded.connect(loaded)
    thread.load_error.connect(errors)

    thread.run()

    loader.assert_called_once_with(POSE_PATH)
    loaded.assert_called_once_with(pose)
    errors.assert_not_called()
    assert thread.pose_path == POSE_PATH


def test_errors_are_reported_not_raised() -> None:
    """A pose file that fails to open is reported as a signal rather than raised."""
    error = OSError("corrupt file")
    thread = PoseLoaderThread(POSE_PATH, mock.Mock(side_effect=error))
    loaded, errors = mock.Mock(), mock.Mock()
    thread.pose_loaded.connect(loaded)
    thread.load_error.connect(errors)

    thread.run()

    errors.assert_called_once_with(error)
    loaded.assert_not_called()
