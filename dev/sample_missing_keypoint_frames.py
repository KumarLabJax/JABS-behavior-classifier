r"""Sample frames with high keypoint-missing rates from labeled behavior bouts.

Script for identifying frames within behavior-labeled bouts where
keypoint detection quality is poor, as indicated by missing (masked-out)
keypoints in the pose file. Intended for building targeted training datasets
for improved pose estimation models.

Pose file versions v3 and above are supported. Version v2 is not supported
because its point-mask array has a different shape (2D instead of 3D).

Usage::

    python dev/sample_missing_keypoint_frames.py PROJECT_DIR \\
        --behavior Seizure \\
        --threshold 0.5 \\
        --num-frames 200 \\
        --out-dir /data/missing_frames \\
        --report-file missing_frames.csv
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
from rich.console import Console
from rich.table import Table

from jabs.pose_estimation import open_pose_file
from jabs.project import Project

logger = logging.getLogger(__name__)


@dataclass
class FrameCandidate:
    """A single candidate frame with its missing-keypoint metrics.

    Attributes:
        video_name: Video filename relative to the project directory.
        frame_index: Zero-based frame index.
        labeled_identity_missing_pct: Fraction of missing keypoints for the
            labeled identity in this frame (0.0-1.0).
        other_identities_missing_pct: Total missing keypoints across all
            non-labeled identities divided by their total keypoint slots
            (0.0-1.0), or ``None`` if the pose file contains only one identity.
    """

    video_name: str
    frame_index: int
    labeled_identity_missing_pct: float
    other_identities_missing_pct: float | None


def _collect_candidates(
    project: Project,
    behavior: str,
    threshold: float,
    expected_keypoints: int,
    min_identities: int,
    console: Console,
) -> list[FrameCandidate]:
    """Collect candidate frames with missing-keypoint fractions above threshold.

    For each video that has positive labels for *behavior*, opens the pose
    file and computes two missing-keypoint fractions for every frame in every
    positive bout:

    - ``labeled_identity_missing_pct``: fraction of the *expected_keypoints*
      real keypoints that are missing for the specific labeled identity.
    - ``other_identities_missing_pct``: total missing keypoints across all
      non-labeled identities divided by their total keypoint slots.

    Using *expected_keypoints* < 12 accounts for pose models with fewer
    keypoints that pad the remaining slots with NaNs/zeros in the mask.  For
    example, a 5-keypoint model always has 7 structurally-absent keypoints;
    passing ``expected_keypoints=5`` ensures a frame with perfect detection
    reports 0% missing rather than 58%.

    When multiple identities are labeled positive in the same frame, the
    highest ``labeled_identity_missing_pct`` value is kept.

    Only frames where ``labeled_identity_missing_pct >= threshold`` are
    returned.

    Args:
        project: Loaded JABS project.
        behavior: Behavior name to scan labels for.
        threshold: Minimum labeled-identity missing fraction to include a frame.
        expected_keypoints: Number of real keypoints in the model (1-12).
            Fractions are computed out of this number, not 12.
        min_identities: Skip videos whose pose file contains fewer than this
            many identities.
        console: Rich console for progress messages.

    Returns:
        List of FrameCandidate objects, unsorted.

    Raises:
        click.ClickException: If a pose file is v2 (unsupported point-mask shape).
    """
    candidates: list[FrameCandidate] = []

    for video_name in project.video_manager.videos:
        video_labels = project.video_manager.load_video_labels(video_name)
        if video_labels is None:
            logger.debug("No annotations for %s, skipping", video_name)
            continue

        # Collect positive bouts per labeled identity for this behavior.
        # bouts_by_identity: {identity_str -> list of (start, end) frame pairs}
        bouts_by_identity: dict[str, list[tuple[int, int]]] = {}
        for identity, label_name, track_labels in video_labels.iter_identity_behavior_labels():
            if label_name != behavior:
                continue
            for block in track_labels.get_blocks():
                if block["present"]:
                    bouts_by_identity.setdefault(identity, []).append(
                        (int(block["start"]), int(block["end"]))
                    )

        if not bouts_by_identity:
            continue

        # Open and validate pose file.
        pose_path = project.video_manager.get_cached_pose_path(video_name)
        pose = open_pose_file(
            pose_path, cache_dir=project.project_paths.cache_dir
        )
        if pose.format_major_version <= 2:
            raise click.ClickException(
                f"Pose file for '{video_name}' is v{pose.format_major_version}. "
                "Only v3 and above are supported (v2 has an incompatible point-mask shape)."
            )

        if pose.num_identities < min_identities:
            logger.debug(
                "Skipping %s: %d identit(ies) < --min-identities %d",
                video_name,
                pose.num_identities,
                min_identities,
            )
            continue

        # _point_mask shape for v3+: (n_identities, n_frames, 12), 1=valid, 0=missing.
        point_mask = pose._point_mask

        console.print(f"  [dim]{video_name}[/dim]")

        # frame_results: {frame_index -> (labeled_missing_pct, other_missing_pct | None)}
        frame_results: dict[int, tuple[float, float | None]] = {}
        for identity_str, bouts in bouts_by_identity.items():
            identity_idx = int(identity_str)
            other_indices = [i for i in range(pose.num_identities) if i != identity_idx]
            for start, end in bouts:
                for frame in range(start, end + 1):
                    labeled_missing = (
                        expected_keypoints - float(point_mask[identity_idx, frame, :].sum())
                    ) / expected_keypoints
                    if frame not in frame_results or labeled_missing > frame_results[frame][0]:
                        if other_indices:
                            total_slots = expected_keypoints * len(other_indices)
                            total_present = float(point_mask[other_indices, frame, :].sum())
                            other_missing: float | None = (
                                total_slots - total_present
                            ) / total_slots
                        else:
                            other_missing = None
                        frame_results[frame] = (labeled_missing, other_missing)

        for frame_index, (labeled_missing, other_missing) in frame_results.items():
            if labeled_missing >= threshold:
                candidates.append(
                    FrameCandidate(
                        video_name=video_name,
                        frame_index=frame_index,
                        labeled_identity_missing_pct=labeled_missing,
                        other_identities_missing_pct=other_missing,
                    )
                )

    return candidates


def _select_top_frames(
    candidates: list[FrameCandidate],
    num_frames: int,
    min_frame_distance: int,
) -> list[FrameCandidate]:
    """Select top-N candidates by labeled-identity missing fraction.

    Candidates are ranked in descending order of ``labeled_identity_missing_pct``.
    Within each video, no two selected frames may be closer than
    *min_frame_distance* frames apart.

    Args:
        candidates: All candidate frames above the threshold.
        num_frames: Maximum number of frames to return.
        min_frame_distance: Minimum frame gap between selected frames per video.

    Returns:
        Selected FrameCandidate objects, sorted by video and frame index.
    """
    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.labeled_identity_missing_pct,
        reverse=True,
    )

    # Track all selected frame indices per video for distance enforcement.
    selected_per_video: dict[str, list[int]] = {}
    selected: list[FrameCandidate] = []

    for candidate in sorted_candidates:
        if len(selected) >= num_frames:
            break

        prior = selected_per_video.get(candidate.video_name, [])
        too_close = any(abs(candidate.frame_index - f) < min_frame_distance for f in prior)
        if too_close:
            logger.debug(
                "Skipping frame %d in %s (too close to a previously selected frame)",
                candidate.frame_index,
                candidate.video_name,
            )
            continue

        selected.append(candidate)
        selected_per_video.setdefault(candidate.video_name, []).append(candidate.frame_index)

    return sorted(selected, key=lambda c: (c.video_name, c.frame_index))


def _write_report(selected: list[FrameCandidate], report_file: Path) -> None:
    """Write a CSV report of the selected frames.

    Args:
        selected: Selected FrameCandidate objects.
        report_file: Path to write the CSV file.
    """
    with report_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "frame",
                "labeled_identity_missing_pct",
                "other_identities_missing_pct",
            ],
        )
        writer.writeheader()
        for c in selected:
            other = (
                f"{c.other_identities_missing_pct:.4f}"
                if c.other_identities_missing_pct is not None
                else "N/A"
            )
            writer.writerow(
                {
                    "video": c.video_name,
                    "frame": c.frame_index,
                    "labeled_identity_missing_pct": f"{c.labeled_identity_missing_pct:.4f}",
                    "other_identities_missing_pct": other,
                }
            )
    logger.info("Report written to %s", report_file)


def _write_frames(
    project_dir: Path,
    selected: list[FrameCandidate],
    out_dir: Path,
    console: Console,
) -> None:
    """Extract selected frames as PNG files.

    Output filenames follow the JABS GUI "Export Frame" convention:
    ``{video_stem}_frame{frame_number:06d}.png``

    Args:
        project_dir: Root directory of the JABS project.
        selected: Selected FrameCandidate objects, sorted by video and frame.
        out_dir: Directory to write PNG files.
        console: Rich console for progress messages.

    Raises:
        click.ClickException: If a video cannot be opened or a frame cannot be written.
    """
    by_video: dict[str, list[int]] = {}
    for c in selected:
        by_video.setdefault(c.video_name, []).append(c.frame_index)

    for video_name, frame_indices in by_video.items():
        video_path = project_dir / video_name
        console.print(f"  [dim]{video_path.name}[/dim] ({len(frame_indices)} frame(s))")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise click.ClickException(f"Unable to open video file: {video_path}")

        try:
            for frame_index in sorted(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                grabbed, frame = cap.read()
                if not grabbed or frame is None:
                    logger.warning(
                        "Could not read frame %d from %s; skipping",
                        frame_index,
                        video_name,
                    )
                    continue

                stem = Path(video_name).stem
                out_path = out_dir / f"{stem}_frame{frame_index:06d}.png"
                if not cv2.imwrite(str(out_path), frame):
                    raise click.ClickException(f"Failed to write PNG: {out_path}")
                logger.debug("Wrote %s", out_path.name)
        finally:
            cap.release()


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
    help="Behavior label name to scan (must exist in the project).",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    show_default=True,
    help=(
        "Minimum fraction of missing keypoints for the labeled identity "
        "required to include a frame as a candidate."
    ),
)
@click.option(
    "--expected-keypoints",
    "expected_keypoints",
    type=click.IntRange(1, 12),
    default=12,
    show_default=True,
    help=(
        "Number of real keypoints in the pose model (1-12). "
        "Use this when the model has fewer than 12 keypoints and pads the remaining "
        "slots with NaN/zero in the mask (e.g. pass 5 for a 5-keypoint model). "
        "Missing fractions are computed out of this number rather than 12, so that "
        "perfect detection always reports 0% missing."
    ),
)
@click.option(
    "--num-frames",
    "num_frames",
    required=True,
    type=click.IntRange(min=1),
    help=(
        "Number of frames to output. Candidates are ranked by descending "
        "labeled-identity missing-keypoint fraction and the top N are selected, "
        "subject to --min-frame-distance."
    ),
)
@click.option(
    "--min-identities",
    "min_identities",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help=(
        "Only include videos whose pose file contains at least this many identities. "
        "For example, pass 2 to exclude single-animal videos."
    ),
)
@click.option(
    "--min-frame-distance",
    "min_frame_distance",
    type=click.IntRange(min=1),
    default=30,
    show_default=True,
    help=(
        "Minimum number of frames that must separate any two selected frames within "
        "the same video. Enforces spatial diversity in the output."
    ),
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(
        "Directory to write PNG frames to. Created (including intermediate directories) "
        "if it does not exist. If omitted, no PNG files are written."
    ),
)
@click.option(
    "--report-file",
    "report_file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help=(
        "Path to write a CSV report listing each selected frame and its missing-keypoint "
        "fractions. Columns: video, frame, labeled_identity_missing_pct, other_identities_missing_pct. "
        "If omitted, no report is written."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging output.",
)
def main(
    project_dir: Path,
    behavior: str,
    threshold: float,
    expected_keypoints: int,
    num_frames: int,
    min_identities: int,
    min_frame_distance: int,
    out_dir: Path | None,
    report_file: Path | None,
    verbose: bool,
) -> None:
    """Sample frames with high keypoint-missing rates from labeled behavior bouts.

    Scans all bouts of BEHAVIOR in PROJECT_DIR, computes the fraction of
    missing keypoints per frame, and selects up to NUM_FRAMES candidates
    ranked by descending labeled-identity missing-keypoint fraction.

    Pose file versions v3 and above are supported.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if out_dir is None and report_file is None:
        raise click.UsageError(
            "At least one of --out-dir or --report-file must be provided; "
            "otherwise there is nothing to write."
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
    console.print(f"Scanning {n_videos} video(s) for '{behavior}' bouts...")
    candidates = _collect_candidates(
        project, behavior, threshold, expected_keypoints, min_identities, console
    )

    if not candidates:
        raise click.ClickException(
            f"No frames with labeled-identity missing fraction >= {threshold} found."
        )

    console.print(
        f"Found {len(candidates)} candidate frame(s) above threshold {threshold}. "
        f"Selecting top {num_frames} (min frame distance: {min_frame_distance})."
    )
    selected = _select_top_frames(candidates, num_frames, min_frame_distance)
    selected.sort(key=lambda c: c.labeled_identity_missing_pct, reverse=True)

    table = Table(title=f"Selected frames — '{behavior}'", show_lines=False)
    table.add_column("Video", style="dim", no_wrap=True)
    table.add_column("Frame", justify="right")
    table.add_column("Labeled missing %", justify="right")
    table.add_column("Other identities missing %", justify="right")
    for c in selected:
        other_str = (
            f"{c.other_identities_missing_pct * 100:.1f}%"
            if c.other_identities_missing_pct is not None
            else "[dim]N/A[/dim]"
        )
        table.add_row(
            c.video_name,
            str(c.frame_index),
            f"{c.labeled_identity_missing_pct * 100:.1f}%",
            other_str,
        )
    console.print(table)
    console.print(f"Selected {len(selected)} frame(s).")

    if report_file is not None:
        _write_report(selected, report_file)
        console.print(f"Report written to {report_file}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Writing {len(selected)} PNG frame(s) to {out_dir}...")
        _write_frames(project_dir, selected, out_dir, console)
        console.print("Done.")


if __name__ == "__main__":
    main()
