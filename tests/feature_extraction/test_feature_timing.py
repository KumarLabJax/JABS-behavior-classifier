"""Tests for the feature timing collector."""

import pickle
from unittest import mock

import pytest

from jabs.feature_extraction import feature_timing
from jabs.feature_extraction.feature_timing import (
    FeatureTimings,
    TimingKey,
    TimingStage,
)


def test_record_accumulates_seconds_and_calls() -> None:
    """Repeated records for one key sum their seconds and calls."""
    timings = FeatureTimings()
    timings.record("base", "angles", TimingStage.PER_FRAME, 1.5)
    timings.record("base", "angles", TimingStage.PER_FRAME, 0.5)

    stats = timings.stats()[TimingKey("base", "angles", TimingStage.PER_FRAME)]
    assert stats.total_seconds == pytest.approx(2.0)
    assert stats.calls == 2
    assert len(timings) == 1


def test_record_accepts_stage_string() -> None:
    """A stage given as its string value is normalized to TimingStage."""
    timings = FeatureTimings()
    timings.record("base", "angles", "window", 1.0)
    assert TimingKey("base", "angles", TimingStage.WINDOW) in timings.stats()


def test_measure_records_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """measure() records the perf_counter delta around its body."""
    clock = mock.Mock(side_effect=[10.0, 12.5])
    monkeypatch.setattr(feature_timing.time, "perf_counter", clock)
    timings = FeatureTimings()

    with timings.measure("base", "angles", TimingStage.WINDOW):
        pass

    stats = timings.stats()[TimingKey("base", "angles", TimingStage.WINDOW)]
    assert stats.total_seconds == pytest.approx(2.5)
    assert stats.calls == 1


def test_measure_records_when_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """measure() still records when the timed body raises."""
    monkeypatch.setattr(feature_timing.time, "perf_counter", mock.Mock(side_effect=[0.0, 1.0]))
    timings = FeatureTimings()

    with pytest.raises(RuntimeError), timings.measure("base", "angles", TimingStage.PER_FRAME):
        raise RuntimeError("boom")

    assert len(timings) == 1


def test_merge_combines_collectors() -> None:
    """merge() adds another collector's stats without aliasing them."""
    a = FeatureTimings()
    a.record("base", "angles", TimingStage.PER_FRAME, 1.0)
    b = FeatureTimings()
    b.record("base", "angles", TimingStage.PER_FRAME, 2.0)
    b.record("social", "closest_distances", TimingStage.WINDOW, 3.0)

    a.merge(b)

    stats = a.stats()
    assert stats[TimingKey("base", "angles", TimingStage.PER_FRAME)].total_seconds == 3.0
    assert stats[TimingKey("base", "angles", TimingStage.PER_FRAME)].calls == 2
    assert stats[TimingKey("social", "closest_distances", TimingStage.WINDOW)].calls == 1
    # merging must not alias the other collector's stats
    a.record("social", "closest_distances", TimingStage.WINDOW, 1.0)
    assert b.stats()[TimingKey("social", "closest_distances", TimingStage.WINDOW)].calls == 1


def test_stats_returns_copy() -> None:
    """Mutating the result of stats() does not change the collector."""
    timings = FeatureTimings()
    timings.record("base", "angles", TimingStage.PER_FRAME, 1.0)
    timings.stats()[TimingKey("base", "angles", TimingStage.PER_FRAME)].add(5.0)
    assert timings.stats()[TimingKey("base", "angles", TimingStage.PER_FRAME)].calls == 1


def test_round_trips_through_pickle() -> None:
    """Collectors survive pickling, as needed to return them from workers."""
    timings = FeatureTimings()
    timings.record("base", "angles", TimingStage.PER_FRAME, 1.0)
    restored = pickle.loads(pickle.dumps(timings))
    assert restored.stats() == timings.stats()


def test_summarize_sums_stages_and_sorts_descending() -> None:
    """summarize() totals stages per feature and sorts longest first."""
    timings = FeatureTimings()
    timings.record("base", "angles", TimingStage.PER_FRAME, 1.0)
    timings.record("base", "angles", TimingStage.WINDOW, 4.0)
    timings.record("base", "point_speeds", TimingStage.PER_FRAME, 2.0)
    timings.record("social", "(group setup)", TimingStage.SETUP, 3.0)

    summary = timings.summarize()

    assert [(s.group, s.feature) for s in summary] == [
        ("base", "angles"),
        ("social", "(group setup)"),
        ("base", "point_speeds"),
    ]
    assert summary[0].per_frame_seconds == pytest.approx(1.0)
    assert summary[0].window_seconds == pytest.approx(4.0)
    assert summary[0].total_seconds == pytest.approx(5.0)
    assert summary[1].setup_seconds == pytest.approx(3.0)


def test_summarize_breaks_ties_by_name() -> None:
    """Features with equal totals are ordered by name."""
    timings = FeatureTimings()
    timings.record("base", "b", TimingStage.PER_FRAME, 1.0)
    timings.record("base", "a", TimingStage.PER_FRAME, 1.0)
    assert [s.feature for s in timings.summarize()] == ["a", "b"]


@pytest.mark.parametrize(("top_n", "expected"), [(None, 3), (2, 2), (0, 0), (10, 3)])
def test_summarize_top_n(top_n: int | None, expected: int) -> None:
    """top_n limits the number of summaries returned."""
    timings = FeatureTimings()
    for i, name in enumerate(["a", "b", "c"]):
        timings.record("base", name, TimingStage.PER_FRAME, float(i))
    assert len(timings.summarize(top_n)) == expected


def test_summarize_negative_top_n_raises() -> None:
    """A negative top_n is rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        FeatureTimings().summarize(-1)
