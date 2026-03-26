"""Sample frames from a JABS project filtered by a behavior label.

Extracts PNG frames from videos in a JABS project for frames annotated with a
given behavior label. Two sampling modes are available: a fixed number of
frames per individual or a fixed total number of frames across the whole project.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import cv2
import numpy as np
import numpy.random as npr
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from jabs.project import Project

logger = logging.getLogger(__name__)


def collect_behavior_bouts(project: Project, behavior: str) -> list[tuple[str, int, int]]:
    """Collect all labeled behavior bouts across every video and identity.

    Args:
        project: Loaded JABS project.
        behavior: Behavior label name to collect bouts for.

    Returns:
        List of ``(video_name, start_frame, end_frame)`` tuples, one entry per
        contiguous bout of the given behavior.  The same video frame may appear
        in multiple tuples if more than one identity was labeled in that frame.
    """
    bouts: list[tuple[str, int, int]] = []
    for video_name in project.video_manager.videos:
        video_labels = project.video_manager.load_video_labels(video_name)
        if video_labels is None:
            logger.debug("No annotations for %s, skipping", video_name)
            continue

        for identity, label_name, track_labels in video_labels.iter_identity_behavior_labels():
            if label_name != behavior:
                continue

            for block in track_labels.get_blocks():
                if block["present"]:
                    bouts.append((video_name, int(block["start"]), int(block["end"])))
                    logger.debug(
                        "Found bout in %s identity=%s frames %d-%d",
                        video_name,
                        identity,
                        block["start"],
                        block["end"],
                    )
    return bouts


def sample_frames_per_bout(
    bouts: list[tuple[str, int, int]],
    frames_per_bout: int,
    rng: npr.Generator,
) -> list[tuple[str, int]]:
    """Sample up to *frames_per_bout* frames from each behavior bout.

    If a bout contains fewer frames than *frames_per_bout*, all frames in that
    bout are included.  Duplicate ``(video_name, frame_index)`` pairs that arise
    when multiple identities share a bout are deduplicated.

    Args:
        bouts: List of ``(video_name, start_frame, end_frame)`` tuples.
        frames_per_bout: Maximum number of frames to sample per bout.
        rng: NumPy random number generator.

    Returns:
        Sorted list of unique ``(video_name, frame_index)`` pairs.
    """
    sampled: set[tuple[str, int]] = set()
    for video_name, start, end in bouts:
        bout_frames = np.arange(start, end + 1, dtype=np.int64)

        # sample the desired number of frames from each bout, if the bout is smaller than
        # frames_per_bout then sample the whole bout
        for f in rng.choice(
            bout_frames, size=min(frames_per_bout, len(bout_frames)), replace=False
        ):
            sampled.add((video_name, int(f)))

    return sorted(sampled)


def sample_num_frames_total(
    bouts: list[tuple[str, int, int]],
    num_frames: int,
    rng: npr.Generator,
) -> list[tuple[str, int]]:
    """Sample *num_frames* frames uniformly at random across all behavior bouts.

    All unique ``(video_name, frame_index)`` pairs within behavior bouts are
    collected first; then *num_frames* are drawn without replacement from that
    pool (or all frames if the pool is smaller than *num_frames*).

    Args:
        bouts: List of ``(video_name, start_frame, end_frame)`` tuples.
        num_frames: Total number of frames to sample.
        rng: NumPy random number generator.

    Returns:
        Sorted list of unique ``(video_name, frame_index)`` pairs.
    """
    # build a list of all frames from all labeled behavior bouts in all videos
    all_frames: set[tuple[str, int]] = set()
    for video_name, start, end in bouts:
        for f in range(start, end + 1):
            all_frames.add((video_name, f))

    # sample from all behavior frames
    frame_list = sorted(all_frames)
    indices = rng.choice(len(frame_list), size=min(num_frames, len(frame_list)), replace=False)
    return sorted(frame_list[i] for i in indices)


def write_frames(
    project_dir: Path,
    sampled: list[tuple[str, int]],
    out_dir: Path,
    console: Console | None = None,
) -> None:
    """Write sampled frames as PNG files to *out_dir*.

    Output filenames follow the JABS GUI "Export Frame" convention:
    ``{video_stem}_frame{frame_number:06d}.png``

    Args:
        project_dir: Root directory of the JABS project.
        sampled: Sorted list of ``(video_name, frame_index)`` pairs to extract.
        out_dir: Directory to write PNG files.  Must already exist.
        console: Rich Console to use for the progress bar.  A new one is created
            if not provided.

    Raises:
        OSError: If a video file cannot be opened.
        RuntimeError: If writing a PNG file fails.
    """
    # Group frames by video to avoid re-opening the same file repeatedly.
    by_video: dict[str, list[int]] = {}
    for video_name, frame_index in sampled:
        by_video.setdefault(video_name, []).append(frame_index)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    task = progress.add_task("Writing frames", total=len(sampled))

    with progress:
        for video_name, frame_indices in by_video.items():
            video_path = project_dir / video_name
            logger.info("Opening %s to extract %d frame(s)", video_path.name, len(frame_indices))
            progress.console.print(f"[dim]{video_path.name}[/dim]")

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise OSError(f"Unable to open video file: {video_path}")

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
                        progress.advance(task)
                        continue

                    stem = Path(video_name).stem
                    out_path = out_dir / f"{stem}_frame{frame_index:06d}.png"
                    success = cv2.imwrite(str(out_path), frame)
                    if not success:
                        raise RuntimeError(f"Failed to write PNG: {out_path}")
                    logger.info("Wrote %s", out_path.name)
                    progress.advance(task)
            finally:
                cap.release()


@click.command(name="sample-frames")
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
    help="Behavior label name to sample frames for.",
)
@click.option(
    "--num-frames",
    "num_frames",
    type=click.IntRange(min=1),
    default=None,
    help="Total number of frames to sample uniformly across all labeled bouts.",
)
@click.option(
    "--frames-per-bout",
    "frames_per_bout",
    type=click.IntRange(min=1),
    default=None,
    help="Number of frames to sample from each individual labeled bout.",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(
        "Directory to write PNG files to.  Defaults to the current working directory.  "
        "Created (including intermediate directories) if it does not exist."
    ),
)
@click.pass_context
def sample_frames_command(
    ctx: click.Context,
    project_dir: Path,
    behavior: str,
    num_frames: int | None,
    frames_per_bout: int | None,
    out_dir: Path | None,
) -> None:
    """Sample frames from a JABS project filtered by a behavior label.

    Extracts PNG images from the project videos for frames annotated with
    BEHAVIOR.  Use --num-frames to set a project-wide total, or
    --frames-per-bout to sample a fixed count per individual bout.
    Exactly one of these options must be provided.

    Output filenames follow the JABS GUI "Export Frame" convention:
    {video_stem}_frame{frame_number:06d}.png

    \b
    Examples:

        # Sample 200 frames total, distributed uniformly across all bouts
        jabs-cli sample-frames --behavior walking --num-frames 200 /path/to/project

        # Sample up to 10 frames from each individual bout
        jabs-cli sample-frames --behavior walking --frames-per-bout 10 /path/to/project

        # Write frames to a specific directory
        jabs-cli sample-frames --behavior walking --num-frames 100 \\
            --out-dir /data/frames /path/to/project
    """
    # Validate mutual exclusion.
    if num_frames is None and frames_per_bout is None:
        raise click.UsageError("Exactly one of --num-frames or --frames-per-bout is required.")
    if num_frames is not None and frames_per_bout is not None:
        raise click.UsageError("--num-frames and --frames-per-bout are mutually exclusive.")

    # Resolve output directory.
    resolved_out_dir = out_dir if out_dir is not None else Path.cwd()

    console = Console()

    if not Project.is_valid_project_directory(project_dir):
        raise click.ClickException(f"Not a valid JABS project directory: {project_dir}")

    with console.status("Loading project...", spinner="dots"):
        project = Project(project_dir, enable_session_tracker=False)

    if behavior not in project.settings["behavior"]:
        available = ", ".join(sorted(project.settings["behavior"])) or "(none)"
        raise click.ClickException(
            f"Behavior '{behavior}' not found in project.  Available behaviors: {available}"
        )

    if ctx.obj["VERBOSE"]:
        console.print(f"Project directory: {project_dir}")
        console.print(f"Behavior:          {behavior}")
        if num_frames is not None:
            console.print(f"Mode:              --num-frames {num_frames}")
        else:
            console.print(f"Mode:              --frames-per-bout {frames_per_bout}")
        console.print(f"Output directory:  {resolved_out_dir}")

    # Collect bouts.
    n_videos = len(project.video_manager.videos)
    with console.status(f"Scanning annotations ({n_videos} video(s))...", spinner="dots"):
        bouts = collect_behavior_bouts(project, behavior)

    if not bouts:
        raise click.ClickException(
            f"No frames labeled as '{behavior}' were found in this project."
        )

    # Sample frame indices.
    rng = npr.default_rng()
    if num_frames is not None:
        sampled = sample_num_frames_total(bouts, num_frames, rng)
    else:
        sampled = sample_frames_per_bout(bouts, frames_per_bout, rng)  # type: ignore[arg-type]

    if not sampled:
        raise click.ClickException("Sampling produced zero frames; nothing to write.")

    console.print(f"Sampling {len(sampled)} frame(s) for behavior '{behavior}'...")

    # Create output directory if needed.
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    write_frames(project_dir, sampled, resolved_out_dir, console=console)

    console.print(f"Done. {len(sampled)} frame(s) written to {resolved_out_dir}")
