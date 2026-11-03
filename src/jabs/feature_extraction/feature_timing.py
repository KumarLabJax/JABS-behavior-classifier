"""Accumulate wall-clock compute time for individual feature modules.

A `FeatureTimings` instance is attached to one or more `FeatureGroup` objects, which then
record how long each feature module spends computing per-frame and window features. The
collector is a plain, picklable container so timings gathered in worker processes can be
returned to the parent process and combined with `FeatureTimings.merge`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum


class TimingStage(str, Enum):
    """Phase of feature computation that a timing measurement covers."""

    SETUP = "setup"
    PER_FRAME = "per_frame"
    WINDOW = "window"


# feature name used for a group's shared setup work (e.g. building ClosestIdentityInfo
# in `_init_feature_mods`), which is not attributable to any single feature module
GROUP_SETUP_FEATURE = "(group setup)"


@dataclass(frozen=True, order=True)
class TimingKey:
    """Identifies one timed unit of work.

    Attributes:
        group: Name of the feature group, e.g. "base".
        feature: Name of the feature module, or `GROUP_SETUP_FEATURE` for shared setup.
        stage: Phase of computation being timed.
    """

    group: str
    feature: str
    stage: TimingStage


@dataclass
class TimingStats:
    """Accumulated timing for a single `TimingKey`.

    Attributes:
        total_seconds: Sum of elapsed wall-clock seconds across all calls.
        calls: Number of measurements accumulated.
    """

    total_seconds: float = 0.0
    calls: int = 0

    def add(self, seconds: float, calls: int = 1) -> None:
        """Accumulate one or more measurements.

        Args:
            seconds: Elapsed seconds to add.
            calls: Number of calls the elapsed time represents.
        """
        self.total_seconds += seconds
        self.calls += calls


@dataclass(frozen=True)
class FeatureTimingSummary:
    """Total compute time for one feature module, summed across stages.

    Attributes:
        group: Name of the feature group.
        feature: Name of the feature module.
        per_frame_seconds: Total seconds spent computing per-frame features.
        window_seconds: Total seconds spent computing window features.
        setup_seconds: Total seconds spent in shared group setup (only nonzero for
            `GROUP_SETUP_FEATURE` entries).
    """

    group: str
    feature: str
    per_frame_seconds: float
    window_seconds: float
    setup_seconds: float

    @property
    def total_seconds(self) -> float:
        """Total seconds across all stages."""
        return self.per_frame_seconds + self.window_seconds + self.setup_seconds


class FeatureTimings:
    """Collector for per-feature compute times.

    Measurements are keyed by (group, feature, stage) and accumulate across calls, so a
    single collector can be shared by every identity and every video processed in a run.

    Note:
        Some groups compute shared data lazily, on first use by a feature module. That time
        is attributed to whichever feature module triggers it first.
    """

    def __init__(self) -> None:
        self._stats: dict[TimingKey, TimingStats] = {}

    def __len__(self) -> int:
        """Return the number of distinct (group, feature, stage) entries recorded."""
        return len(self._stats)

    def record(self, group: str, feature: str, stage: TimingStage, seconds: float) -> None:
        """Record one measurement.

        Args:
            group: Name of the feature group.
            feature: Name of the feature module.
            stage: Phase of computation the measurement covers.
            seconds: Elapsed wall-clock seconds.
        """
        key = TimingKey(group, feature, TimingStage(stage))
        self._stats.setdefault(key, TimingStats()).add(seconds)

    @contextmanager
    def measure(self, group: str, feature: str, stage: TimingStage) -> Iterator[None]:
        """Context manager that records the elapsed time of its body.

        The measurement is recorded even if the body raises.

        Args:
            group: Name of the feature group.
            feature: Name of the feature module.
            stage: Phase of computation being timed.

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(group, feature, stage, time.perf_counter() - start)

    def merge(self, other: FeatureTimings) -> None:
        """Add all measurements from another collector into this one.

        Args:
            other: Collector to merge, e.g. one returned from a worker process.
        """
        for key, stats in other._stats.items():
            self._stats.setdefault(key, TimingStats()).add(stats.total_seconds, stats.calls)

    def stats(self) -> dict[TimingKey, TimingStats]:
        """Return a copy of the raw accumulated measurements.

        Returns:
            Mapping of timing key to accumulated stats.
        """
        return {
            key: TimingStats(s.total_seconds, s.calls) for key, s in sorted(self._stats.items())
        }

    def summarize(self, top_n: int | None = None) -> list[FeatureTimingSummary]:
        """Summarize total compute time per feature module, longest first.

        Args:
            top_n: If given, return only the `top_n` slowest entries.

        Returns:
            Summaries sorted by total time descending (ties broken by group, feature name).

        Raises:
            ValueError: If `top_n` is negative.
        """
        if top_n is not None and top_n < 0:
            raise ValueError("top_n must be non-negative")

        totals: dict[tuple[str, str], dict[TimingStage, float]] = {}
        for key, stats in self._stats.items():
            stage_totals = totals.setdefault((key.group, key.feature), {})
            stage_totals[key.stage] = stage_totals.get(key.stage, 0.0) + stats.total_seconds

        summaries = [
            FeatureTimingSummary(
                group=group,
                feature=feature,
                per_frame_seconds=stages.get(TimingStage.PER_FRAME, 0.0),
                window_seconds=stages.get(TimingStage.WINDOW, 0.0),
                setup_seconds=stages.get(TimingStage.SETUP, 0.0),
            )
            for (group, feature), stages in totals.items()
        ]
        summaries.sort(key=lambda s: (-s.total_seconds, s.group, s.feature))
        return summaries if top_n is None else summaries[:top_n]
