"""Tests for rendering feature timings as Markdown."""

import pytest

from jabs.feature_extraction.feature_timing import (
    GROUP_SETUP_FEATURE,
    FeatureTimings,
    TimingStage,
)
from jabs.feature_extraction.feature_timing_report import render_timing_report


@pytest.fixture
def timings() -> FeatureTimings:
    """Timings for three entries totaling 10 seconds."""
    t = FeatureTimings()
    t.record("base", "angles", TimingStage.PER_FRAME, 1.0)
    t.record("base", "angles", TimingStage.WINDOW, 5.0)
    t.record("social", GROUP_SETUP_FEATURE, TimingStage.SETUP, 3.0)
    t.record("base", "point_speeds", TimingStage.PER_FRAME, 1.0)
    return t


def _table_rows(report: str) -> list[list[str]]:
    """Return the cells of each table body row."""
    rows = [line for line in report.splitlines() if line.startswith("| ") and "Rank" not in line]
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]


def test_rows_sorted_slowest_first(timings: FeatureTimings) -> None:
    """Rows are ranked by total time, descending, with percentages of the grand total."""
    rows = _table_rows(render_timing_report(timings))

    assert [row[1] for row in rows] == ["`angles`", f"`{GROUP_SETUP_FEATURE}`", "`point_speeds`"]
    assert rows[0] == ["1", "`angles`", "base", "6.000", "60.0%", "1.000", "5.000"]
    # setup entries have no per-frame or window time
    assert rows[1][5:] == ["-", "-"]


def test_top_n_limits_rows_but_not_total(timings: FeatureTimings) -> None:
    """top_n drops rows but percentages and the total still cover every entry."""
    report = render_timing_report(timings, top_n=1)
    rows = _table_rows(report)

    assert len(rows) == 1
    assert rows[0][4] == "60.0%"
    assert "Showing the 1 slowest of 3 entries." in report
    assert "**10.000 s**" in report


def test_metadata_and_notes_rendered(timings: FeatureTimings) -> None:
    """Metadata and notes are rendered as bullet lists."""
    report = render_timing_report(timings, metadata={"Project": "/p"}, notes=["a caveat"])

    assert "- **Project:** /p" in report
    assert "## Notes\n\n- a caveat" in report
    assert report.endswith("\n")


def test_empty_timings() -> None:
    """An empty collector renders a message instead of a table."""
    report = render_timing_report(FeatureTimings())

    assert "No feature computation was timed." in report
    assert "| Rank |" not in report


def test_pipe_in_name_is_escaped() -> None:
    """A pipe in a feature name must not split the table cell."""
    t = FeatureTimings()
    t.record("base", "a|b", TimingStage.PER_FRAME, 1.0)

    assert "`a\\|b`" in render_timing_report(t)
