"""Render collected feature compute times as a Markdown report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .feature_timing import FeatureTimings, FeatureTimingSummary


def _escape_cell(text: str) -> str:
    """Escape characters that would break a Markdown table cell."""
    return text.replace("|", "\\|")


def _format_seconds(seconds: float) -> str:
    """Format a duration in seconds for a table cell, or "-" if nothing was recorded."""
    return f"{seconds:.3f}" if seconds > 0 else "-"


def _table_row(rank: int, summary: FeatureTimingSummary, grand_total: float) -> str:
    """Render one feature as a Markdown table row."""
    percent = 100.0 * summary.total_seconds / grand_total if grand_total > 0 else 0.0
    cells = [
        str(rank),
        f"`{_escape_cell(summary.feature)}`",
        _escape_cell(summary.group),
        f"{summary.total_seconds:.3f}",
        f"{percent:.1f}%",
        _format_seconds(summary.per_frame_seconds),
        _format_seconds(summary.window_seconds),
    ]
    return "| " + " | ".join(cells) + " |"


def render_timing_report(
    timings: FeatureTimings,
    top_n: int | None = None,
    metadata: Mapping[str, str] | None = None,
    notes: Sequence[str] = (),
) -> str:
    """Render feature compute times as a Markdown document.

    Features are listed from longest to shortest total compute time. Percentages are
    relative to the time of every recorded entry, including those omitted by `top_n`.

    Args:
        timings: Collected feature timings.
        top_n: If given, list only the `top_n` slowest entries.
        metadata: Key/value pairs describing the run, rendered as a bullet list.
        notes: Additional caveats rendered as a bullet list below the table.

    Returns:
        Markdown text, ending with a newline.

    Raises:
        ValueError: If `top_n` is negative.
    """
    all_summaries = timings.summarize()
    shown = timings.summarize(top_n)
    grand_total = sum(s.total_seconds for s in all_summaries)

    lines = ["# JABS Feature Compute Times", ""]

    if metadata:
        lines.extend(f"- **{key}:** {value}" for key, value in metadata.items())
        lines.append("")

    lines.append(f"Total feature compute time: **{grand_total:.3f} s**")
    lines.append("")

    if not all_summaries:
        lines.append("No feature computation was timed.")
    else:
        if len(shown) < len(all_summaries):
            lines.append(f"Showing the {len(shown)} slowest of {len(all_summaries)} entries.")
            lines.append("")
        lines.append(
            "| Rank | Feature | Group | Total (s) | % of Total | Per-Frame (s) | Window (s) |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---:|")
        lines.extend(
            _table_row(rank, summary, grand_total) for rank, summary in enumerate(shown, start=1)
        )

    if notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in notes)

    return "\n".join(lines) + "\n"
