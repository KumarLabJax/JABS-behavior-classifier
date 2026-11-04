"""Tests for the labeling session activity log written by :class:`SessionTracker`."""

import datetime
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


@pytest.mark.parametrize(
    ("present", "activity"),
    [
        (True, ActivityTypes.BEHAVIOR_LABEL_CREATED),
        (False, ActivityTypes.NOT_BEHAVIOR_LABEL_CREATED),
    ],
    ids=["behavior", "not-behavior"],
)
def test_label_created_records_the_frame_range(
    tracker: SessionTracker, present: bool, activity: ActivityTypes
) -> None:
    """A created label records its activity type and the range it covers."""
    tracker.label_created(Path("/videos/one.mp4"), 2, "groom", present, 10, 20)

    (entry,) = _activity_log(tracker)
    assert entry["activity"] == activity.name
    assert entry["video"] == "one.mp4"
    assert entry["behavior"] == "groom"
    assert entry["identity"] == 2
    assert (entry["start_frame"], entry["end_frame"]) == (10, 20)


def test_label_deleted_records_the_frame_range(tracker: SessionTracker) -> None:
    """A deleted label records the same fields as a created one."""
    tracker.label_deleted(Path("/videos/one.mp4"), 2, "groom", 10, 20)

    (entry,) = _activity_log(tracker)
    assert entry["activity"] == ActivityTypes.LABEL_DELETED.name
    assert entry["video"] == "one.mp4"
    assert entry["behavior"] == "groom"
    assert entry["identity"] == 2
    assert (entry["start_frame"], entry["end_frame"]) == (10, 20)


def test_classifier_trained_omits_scores_that_were_not_given(tracker: SessionTracker) -> None:
    """Optional metrics are absent rather than null when not supplied."""
    tracker.classifier_trained("groom", "XGBOOST", 5)

    (entry,) = _activity_log(tracker)
    assert entry["activity"] == ActivityTypes.CLASSIFIER_TRAINED.name
    assert entry["behavior"] == "groom"
    assert entry["classifier_type"] == "XGBOOST"
    assert entry["k-fold"] == 5
    assert "mean accuracy" not in entry
    assert "mean fbeta (behavior)" not in entry


def test_classifier_trained_includes_scores_that_were_given(tracker: SessionTracker) -> None:
    """Supplied metrics are logged as 3-significant-digit strings."""
    tracker.classifier_trained("groom", "XGBOOST", 5, accuracy=0.912345, fbeta_behavior=0.8)

    (entry,) = _activity_log(tracker)
    assert entry["mean accuracy"] == "0.912"
    assert entry["mean fbeta (behavior)"] == "0.8"


def test_pause_and_resume_are_logged_in_order(tracker: SessionTracker) -> None:
    """Pausing and resuming a session each append one entry."""
    tracker.pause_session()
    tracker.resume_session()

    assert [entry["activity"] for entry in _activity_log(tracker)] == [
        ActivityTypes.SESSION_PAUSED.name,
        ActivityTypes.SESSION_RESUMED.name,
    ]


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("label_created", (Path("/videos/one.mp4"), 0, "groom", True, 1, 2)),
        ("label_deleted", (Path("/videos/one.mp4"), 0, "groom", 1, 2)),
        ("classifier_trained", ("groom", "XGBOOST", 5)),
        ("pause_session", ()),
        ("resume_session", ()),
        ("behavior_selected", ("groom",)),
    ],
    ids=["created", "deleted", "trained", "paused", "resumed", "selected"],
)
def test_activities_are_not_logged_when_tracking_disabled(
    tmp_path: Path, method: str, args: tuple
) -> None:
    """A disabled tracker records nothing and never creates a session file."""
    tracker = SessionTracker(_make_project(tmp_path), tracking_enabled=False)
    tracker.start_session()

    getattr(tracker, method)(*args)

    assert tracker._session is None
    assert list(tmp_path.iterdir()) == []


def test_every_activity_entry_starts_with_a_timestamp_and_activity(
    tracker: SessionTracker,
) -> None:
    """All entries share one envelope, so the log can be parsed uniformly."""
    tracker.video_opened(Path("/videos/one.mp4"))
    tracker.label_created(Path("/videos/one.mp4"), 0, "groom", True, 1, 2)
    tracker.classifier_trained("groom", "XGBOOST", 5)
    tracker.pause_session()

    for entry in _activity_log(tracker):
        assert list(entry)[:2] == ["timestamp", "activity"]
        datetime.datetime.fromisoformat(entry["timestamp"])
