"""Tests for CentralWidget helpers that don't require instantiating the widget."""

from types import SimpleNamespace

import pytest

try:
    from jabs.ui.main_window.central_widget import CentralWidget

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


def _stub_widget(excluded: set[str]) -> SimpleNamespace:
    """Minimal stand-in exposing what _included_counts reads from self."""
    return SimpleNamespace(
        _project=SimpleNamespace(
            settings_manager=SimpleNamespace(is_video_excluded=lambda v: v in excluded)
        )
    )


def test_included_counts_drops_excluded_videos():
    """Excluded videos are removed from the counts used for train-button thresholds."""
    counts = {
        "a.avi": {0: {"fragmented_frame_counts": (30, 30)}},
        "b.avi": {0: {"fragmented_frame_counts": (30, 30)}},
        "c.avi": {0: {"fragmented_frame_counts": (30, 30)}},
    }
    result = CentralWidget._included_counts(_stub_widget({"b.avi"}), counts)

    assert set(result.keys()) == {"a.avi", "c.avi"}
    # surviving entries are passed through unchanged
    assert result["a.avi"] == counts["a.avi"]


def test_included_counts_no_exclusions_returns_all():
    """With nothing excluded, all videos are retained."""
    counts = {"a.avi": {0: {}}, "b.avi": {0: {}}}
    result = CentralWidget._included_counts(_stub_widget(set()), counts)
    assert set(result.keys()) == {"a.avi", "b.avi"}


def test_included_counts_none_returns_empty():
    """None counts (not yet computed) return an empty dict instead of raising."""
    assert CentralWidget._included_counts(_stub_widget(set()), None) == {}


def _bout_stub_widget(counts: dict, excluded: set[str]) -> SimpleNamespace:
    """Stand-in exposing what _included_project_bout_totals reads from self."""
    stub = _stub_widget(excluded)
    stub._counts = counts
    return stub


def test_included_project_bout_totals_excludes_excluded_videos():
    """Bout totals for the report sum only non-excluded videos."""
    counts = {
        "a.avi": {0: {"unfragmented_bout_counts": (3, 2)}},
        "b.avi": {0: {"unfragmented_bout_counts": (10, 10)}},  # excluded
    }
    stub = _bout_stub_widget(counts, {"b.avi"})
    assert CentralWidget._included_project_bout_totals(stub) == (3, 2)


def test_included_project_bout_totals_handles_none_counts():
    """No counts yet -> zero totals (no crash)."""
    stub = _bout_stub_widget(None, set())
    assert CentralWidget._included_project_bout_totals(stub) == (0, 0)


def test_frame_count_mismatch_message_none_when_equal():
    """Matching video/pose frame counts produce no warning message."""
    assert CentralWidget._frame_count_mismatch_message("v.avi", 100, 100) is None


def test_frame_count_mismatch_message_reports_counts():
    """A mismatch yields a message naming the video and both frame counts."""
    msg = CentralWidget._frame_count_mismatch_message("v.avi", 100, 90)
    assert msg is not None
    assert "v.avi" in msg
    assert "100" in msg
    assert "90" in msg
