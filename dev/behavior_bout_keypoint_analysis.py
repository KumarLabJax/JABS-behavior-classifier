r"""Report per-keypoint missing-value rates for behavior bouts in a JABS project.

For each positive bout of a behavior, computes the fraction of frames where each
pose keypoint is absent (below the confidence threshold). Produces:

  * A per-bout table showing the missing rate for each tracked keypoint.
  * A project-wide summary table with overall %, mean, median, and std
    of per-bout missing rates for each keypoint.

Pose files v3 and above are supported (v2 has an incompatible point-mask layout).

Usage::

    python dev/behavior_bout_keypoint_analysis.py PROJECT_DIR \\
        --behavior Seizure

    python dev/behavior_bout_keypoint_analysis.py PROJECT_DIR \\
        --behavior Seizure \\
        --report-file keypoint_report.csv
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from jabs.pose_estimation import open_pose_file
from jabs.project import Project

logger = logging.getLogger(__name__)

_KEYPOINT_FULL_NAMES: list[str] = [
    "Nose",
    "Left Ear",
    "Right Ear",
    "Base Neck",
    "Left Front Paw",
    "Right Front Paw",
    "Center Spine",
    "Left Rear Paw",
    "Right Rear Paw",
    "Base Tail",
    "Mid Tail",
    "Tip Tail",
]

# Abbreviated names used in the wide per-bout table.
_KEYPOINT_SHORT_NAMES: list[str] = [
    "Nose",
    "L.Ear",
    "R.Ear",
    "B.Nck",
    "LF.Pw",
    "RF.Pw",
    "C.Sp",
    "LR.Pw",
    "RR.Pw",
    "B.Til",
    "M.Til",
    "T.Til",
]

_N_KEYPOINTS: int = 12


@dataclass
class BoutRecord:
    """Missing-keypoint statistics for a single behavior bout.

    Attributes:
        video_name: Video filename relative to the project directory.
        identity: Zero-based identity index.
        start_frame: First frame of the bout (inclusive).
        end_frame: Last frame of the bout (inclusive).
        missing_pct: Per-keypoint missing fraction, shape (_N_KEYPOINTS,).
            Each element is the fraction of frames in the bout where that keypoint
            is absent (0.0 = never missing, 1.0 = always missing).
    """

    video_name: str
    identity: int
    start_frame: int
    end_frame: int
    missing_pct: npt.NDArray[np.float64]

    @property
    def duration(self) -> int:
        """Number of frames in the bout (inclusive on both ends)."""
        return self.end_frame - self.start_frame + 1

    @property
    def overall_missing_pct(self) -> float:
        """Mean missing fraction across all keypoints."""
        return float(self.missing_pct.mean())


def _collect_bouts(
    project: Project,
    behavior: str,
    console: Console,
) -> list[BoutRecord]:
    """Collect per-bout keypoint missing statistics for all positive bouts of *behavior*.

    For each video in the project with positive labels for *behavior*, opens the
    corresponding pose file and computes, for every labeled bout, the fraction of
    frames where each keypoint is absent.

    Args:
        project: Loaded JABS project.
        behavior: Behavior name to collect bouts for.
        console: Rich console for progress messages.

    Returns:
        List of BoutRecord objects, one per positive bout, in video-iteration order.

    Raises:
        click.ClickException: If a pose file is v2 (unsupported point-mask shape).
    """
    bouts: list[BoutRecord] = []

    for video_name in project.video_manager.videos:
        video_labels = project.video_manager.load_video_labels(video_name)
        if video_labels is None:
            logger.debug("No annotations for %s, skipping", video_name)
            continue

        bouts_by_identity: dict[int, list[tuple[int, int]]] = {}
        for identity, label_name, track_labels in video_labels.iter_identity_behavior_labels():
            if label_name != behavior:
                continue
            for block in track_labels.get_blocks():
                if block["present"]:
                    bouts_by_identity.setdefault(int(identity), []).append(
                        (int(block["start"]), int(block["end"]))
                    )

        if not bouts_by_identity:
            continue

        pose_path = project.video_manager.get_cached_pose_path(video_name)
        pose = open_pose_file(pose_path, cache_dir=project.project_paths.cache_dir)

        if pose.format_major_version <= 2:
            raise click.ClickException(
                f"Pose file for '{video_name}' is v{pose.format_major_version}. "
                "Only v3 and above are supported (v2 has an incompatible point-mask shape)."
            )

        # Stack masks: (n_identities, n_frames, 12) — 1 = valid, 0 = missing
        point_mask = np.stack(
            [pose.get_identity_point_mask(i) for i in range(pose.num_identities)]
        )

        console.print(f"  [dim]{video_name}[/dim]")

        for identity_idx, identity_bouts in sorted(bouts_by_identity.items()):
            for start, end in identity_bouts:
                bout_mask = point_mask[identity_idx, start : end + 1, :]
                # bout_mask: (bout_length, _N_KEYPOINTS), 1=valid 0=missing
                missing_pct: npt.NDArray[np.float64] = (1.0 - bout_mask.mean(axis=0)).astype(
                    np.float64
                )
                bouts.append(
                    BoutRecord(
                        video_name=video_name,
                        identity=identity_idx,
                        start_frame=start,
                        end_frame=end,
                        missing_pct=missing_pct,
                    )
                )
                logger.debug(
                    "Bout %s identity=%d [%d, %d] — overall missing %.1f%%",
                    video_name,
                    identity_idx,
                    start,
                    end,
                    bouts[-1].overall_missing_pct * 100,
                )

    return bouts


def _print_bout_table(
    bouts: list[BoutRecord],
    console: Console,
) -> None:
    """Print a Rich table with one row per bout showing per-keypoint missing rates.

    Args:
        bouts: Collected bout records.
        console: Rich console.
    """
    table = Table(title="Per-Bout Keypoint Missing Rates", show_lines=False)
    table.add_column("Video", style="dim", no_wrap=True)
    table.add_column("ID", justify="right")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Dur", justify="right")
    for name in _KEYPOINT_SHORT_NAMES:
        table.add_column(name, justify="right")
    table.add_column("Overall", justify="right", style="bold")

    for bout in bouts:
        row: list[str] = [
            bout.video_name,
            str(bout.identity),
            str(bout.start_frame),
            str(bout.end_frame),
            str(bout.duration),
        ]
        for pct in bout.missing_pct:
            row.append(f"{pct * 100:.1f}%")
        row.append(f"{bout.overall_missing_pct * 100:.1f}%")
        table.add_row(*row)

    console.print(table)


def _print_summary_table(
    bouts: list[BoutRecord],
    console: Console,
    behavior_name: str,
) -> None:
    """Print a Rich table of project-wide keypoint statistics aggregated over all bouts.

    Columns:
      * Overall %: frame-count-weighted missing rate across every frame of every bout.
      * Bout Mean, Bout Median, Bout Std: each bout is first reduced to a single
        missing % for that keypoint, then these statistics summarize the distribution
        of those per-bout values across all bouts.

    The distinction between Overall and Bout Mean matters when bouts have different
    lengths — Overall weights each bout by its frame count, Bout Mean treats every
    bout equally regardless of duration.

    Args:
        bouts: Collected bout records.
        console: Rich console.
        behavior_name: Behavior name.
    """
    durations = np.array([b.duration for b in bouts], dtype=np.float64)

    table = Table(
        title=f"Project-Wide Keypoint Summary (frames labeled {behavior_name})",
        caption="Overall: frame-weighted across all bouts.  Bout stats: each bout contributes one value.",
        show_lines=True,
    )
    table.add_column("Keypoint", no_wrap=True)
    table.add_column(
        "Overall %", justify="right", style="bold", header_style="bold", footer_style="bold"
    )
    table.add_column("Bout Mean %", justify="right")
    table.add_column("Bout Median %", justify="right")
    table.add_column("Bout Std %", justify="right")

    for kp_idx in range(_N_KEYPOINTS):
        per_bout: npt.NDArray[np.float64] = np.array(
            [b.missing_pct[kp_idx] for b in bouts], dtype=np.float64
        )
        overall = float(np.average(per_bout, weights=durations))
        table.add_row(
            _KEYPOINT_FULL_NAMES[kp_idx],
            f"{overall * 100:.1f}%",
            f"{per_bout.mean() * 100:.1f}%",
            f"{np.median(per_bout) * 100:.1f}%",
            f"{per_bout.std() * 100:.1f}%",
        )

    console.print(table)


def _write_report(
    bouts: list[BoutRecord],
    report_file: Path,
) -> None:
    """Write a CSV report with one row per bout.

    Columns: video, identity, start_frame, end_frame, duration, one column per
    keypoint named ``<keypoint>_missing_pct`` (values 0.0-1.0), and
    overall_missing_pct.

    Args:
        bouts: Collected bout records.
        report_file: Destination CSV path.
    """
    kp_col_names = [
        f"{name.lower().replace(' ', '_')}_missing_pct" for name in _KEYPOINT_FULL_NAMES
    ]
    fieldnames: list[str] = [
        "video",
        "identity",
        "start_frame",
        "end_frame",
        "duration",
        *kp_col_names,
        "overall_missing_pct",
    ]

    try:
        report_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise click.ClickException(f"Cannot create directory for report file: {e}") from e

    try:
        with report_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for bout in bouts:
                row: dict[str, str | int] = {
                    "video": bout.video_name,
                    "identity": bout.identity,
                    "start_frame": bout.start_frame,
                    "end_frame": bout.end_frame,
                    "duration": bout.duration,
                }
                for kp_idx, col_name in enumerate(kp_col_names):
                    row[col_name] = f"{bout.missing_pct[kp_idx]:.6f}"
                row["overall_missing_pct"] = f"{bout.overall_missing_pct:.6f}"
                writer.writerow(row)
    except OSError as e:
        raise click.ClickException(f"Failed to write report file: {e}") from e

    logger.info("Report written to %s", report_file)


@click.command()
@click.argument(
    "project_dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--behavior",
    required=True,
    type=str,
    help="Behavior label name to analyse (must exist in the project).",
)
@click.option(
    "--report-file",
    "report_file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help=(
        "Path to write a CSV report with one row per bout. "
        "Columns: video, identity, start_frame, end_frame, duration, "
        "one <keypoint>_missing_pct column per keypoint (0.0-1.0), "
        "and overall_missing_pct. If omitted, no CSV is written."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG-level) logging.",
)
def main(
    project_dir: Path,
    behavior: str,
    report_file: Path | None,
    verbose: bool,
) -> None:
    """Report per-keypoint missing rates for every behavior bout in PROJECT_DIR.

    For each positive bout of BEHAVIOR, computes the fraction of frames where
    each pose keypoint is absent. Prints a per-bout table and a project-wide
    summary with mean, median, and standard deviation per keypoint.

    Pose file versions v3 and above are supported.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    console = Console()

    with console.status("Loading project...", spinner="dots"):
        if not Project.is_valid_project_directory(project_dir):
            raise click.ClickException(f"Not a valid JABS project directory: {project_dir}")
        project = Project(project_dir, enable_session_tracker=False)

    if behavior not in project.settings["behavior"]:
        available = ", ".join(sorted(project.settings["behavior"])) or "(none)"
        raise click.ClickException(
            f"Behavior '{behavior}' not found in project. Available: {available}"
        )

    n_videos = len(project.video_manager.videos)
    console.print(f"Scanning {n_videos} video(s) for [bold]'{behavior}'[/bold] bouts...")
    bouts = _collect_bouts(project, behavior, console)

    if not bouts:
        raise click.ClickException(f"No positive bouts of '{behavior}' found in the project.")

    n_bouts = len(bouts)
    total_frames = sum(b.duration for b in bouts)
    console.print(
        f"\nFound [bold]{n_bouts}[/bold] bout(s) totalling [bold]{total_frames}[/bold] frame(s).\n"
    )

    console.print(Rule("Per-Bout Details"))
    _print_bout_table(bouts, console)

    console.print(Rule("Project-Wide Summary"))
    _print_summary_table(bouts, console, behavior)

    if report_file is not None:
        _write_report(bouts, report_file)
        console.print(f"\nCSV report written to [bold]{report_file}[/bold]")


if __name__ == "__main__":
    main()
