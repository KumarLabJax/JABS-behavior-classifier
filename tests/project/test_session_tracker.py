"""Tests for the labeling session activity log written by :class:`SessionTracker`."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jabs.project.session_tracker import ActivityTypes, SessionTracker


def _counts(frames: tuple[int, int], bouts: tuple[int, int]) -> dict:
    """Build one identity's entry of the dict ``Project.load_counts`` returns."""
    return {
        "fragmented_frame_counts": frames,
        "fragmented_bout_counts": bouts,
        "unfragmented_frame_counts": frames,
        "unfragmented_bout_counts": bouts,
    }


def _make_project(tmp_path: Path, counts: dict[str, dict] | None = None) -> MagicMock:
    """Build a Project-like mock with a real session directory.

    Args:
        tmp_path: Directory the session log is written to.
        counts: Mapping of behavior name to the dict ``Project.counts`` returns
            for it. Behaviors not listed report no counts.

    Returns:
        A mock with the attributes ``SessionTracker`` uses.
    """
    project = MagicMock()
    project.project_paths = SimpleNamespace(session_dir=tmp_path)
    project.counts.side_effect = lambda behavior: (counts or {}).get(behavior, {})
    return project


@pytest.fixture
def tracker(tmp_path: Path) -> SessionTracker:
    """A started tracker writing to a temporary session directory."""
    tracker = SessionTracker(_make_project(tmp_path))
    tracker.start_session()
    return tracker


def _activity_log(tracker: SessionTracker) -> list[dict]:
    """Read back the activity log the tracker flushed to disk."""
    return json.loads(tracker._session_file.read_text())["activity_log"]


def test_video_activities_name_the_video_under_the_same_key(tracker: SessionTracker) -> None:
    """Opening and closing a video must record the filename under one key."""
    tracker.video_opened(Path("/videos/one.mp4"))
    tracker.video_closed(Path("/videos/one.mp4"))

    opened, closed = _activity_log(tracker)
    assert opened["activity"] == ActivityTypes.VIDEO_OPENED.name
    assert closed["activity"] == ActivityTypes.VIDEO_CLOSED.name
    assert opened["video"] == "one.mp4"
    assert closed["video"] == "one.mp4"


@pytest.mark.parametrize(
    ("method", "activity"),
    [
        ("video_opened", ActivityTypes.VIDEO_OPENED),
        ("video_closed", ActivityTypes.VIDEO_CLOSED),
    ],
    ids=["opened", "closed"],
)
def test_video_activities_are_not_logged_when_tracking_disabled(
    tmp_path: Path, method: str, activity: ActivityTypes
) -> None:
    """A disabled tracker records nothing and never creates a session file."""
    tracker = SessionTracker(_make_project(tmp_path), tracking_enabled=False)
    tracker.start_session()

    getattr(tracker, method)(Path("/videos/one.mp4"))

    assert tracker._session is None
    assert list(tmp_path.iterdir()) == []


def test_label_counts_are_totaled_over_videos_and_identities(tmp_path: Path) -> None:
    """Selecting a behavior snapshots the project-wide totals for it."""
    project = _make_project(
        tmp_path,
        counts={
            "groom": {
                "video_a.mp4": {0: _counts((10, 5), (2, 1)), 1: _counts((3, 0), (1, 0))},
                "video_b.mp4": {0: _counts((7, 2), (1, 1))},
            }
        },
    )
    tracker = SessionTracker(project)
    tracker.start_session()

    tracker.behavior_selected("groom")

    assert tracker._session["starting_label_counts"]["groom"] == {
        "frames_behavior": 20,
        "frames_not_behavior": 7,
        "bouts_behavior": 4,
        "bouts_not_behavior": 2,
    }


def test_starting_counts_are_captured_only_on_first_selection(tmp_path: Path) -> None:
    """Re-selecting a behavior keeps the totals from when it was first selected."""
    counts = {"groom": {"video_a.mp4": {0: _counts((10, 5), (2, 1))}}}
    project = _make_project(tmp_path, counts=counts)
    tracker = SessionTracker(project)
    tracker.start_session()

    tracker.behavior_selected("groom")
    counts["groom"] = {"video_a.mp4": {0: _counts((99, 99), (9, 9))}}
    tracker.behavior_selected("groom")

    assert tracker._session["starting_label_counts"]["groom"]["frames_behavior"] == 10
    assert [entry["activity"] for entry in _activity_log(tracker)] == [
        ActivityTypes.BEHAVIOR_SELECTED.name
    ] * 2


def test_ending_counts_use_the_same_totals_as_starting_counts(tmp_path: Path) -> None:
    """Both snapshots are totaled the same way, so they can be compared."""
    counts = {"groom": {"video_a.mp4": {0: _counts((10, 5), (2, 1))}}}
    project = _make_project(tmp_path, counts=counts)
    tracker = SessionTracker(project)
    tracker.start_session()
    tracker.behavior_selected("groom")
    session_file = tracker._session_file

    counts["groom"] = {"video_a.mp4": {0: _counts((18, 5), (3, 1))}}
    tracker.end_session()

    session = json.loads(session_file.read_text())
    assert session["session_end"] is not None
    assert session["starting_label_counts"]["groom"]["frames_behavior"] == 10
    assert session["ending_label_counts"]["groom"] == {
        "frames_behavior": 18,
        "frames_not_behavior": 5,
        "bouts_behavior": 3,
        "bouts_not_behavior": 1,
    }
