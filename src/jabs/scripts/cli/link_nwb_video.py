"""Link JABS pose NWB files to their source videos so DANDI accepts the videos.

DANDI takes a video only when an NWB file points to it through an ``ImageSeries``.
This command matches each ``<name>.nwb`` file in one directory with the
``<name>.mp4`` video in another and adds that ``ImageSeries`` to the NWB file.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import click
import cv2
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from jabs.io.internal.pose import (
    NWBVideoLinkInfo,
    link_external_video,
    read_video_link_info,
    relative_video_path,
    require_video_link_support,
)

logger = logging.getLogger(__name__)

VIDEO_SUFFIX = ".mp4"

# Relative tolerance when comparing the frame rate of a video with that of the pose data.
# Allows for 29.97 fps video against pose data recorded as 30 fps.
_FPS_REL_TOL = 0.01


class LinkStatus(enum.Enum):
    """Outcome of linking one NWB file."""

    LINKED = "linked"
    WOULD_LINK = "would link"
    ALREADY_LINKED = "already linked"
    NO_VIDEO = "no video"
    MISMATCH = "mismatch"
    ERROR = "error"

    @property
    def is_problem(self) -> bool:
        """Whether this outcome means the file still has no usable link."""
        return self in {LinkStatus.NO_VIDEO, LinkStatus.MISMATCH, LinkStatus.ERROR}


_STATUS_STYLE = {
    LinkStatus.LINKED: "green",
    LinkStatus.WOULD_LINK: "cyan",
    LinkStatus.ALREADY_LINKED: "dim",
    LinkStatus.NO_VIDEO: "red",
    LinkStatus.MISMATCH: "red",
    LinkStatus.ERROR: "red",
}


@dataclass(frozen=True)
class LinkResult:
    """Result of trying to link one NWB file to its video.

    Attributes:
        nwb_path: The NWB file.
        status: What happened to it.
        message: Detail for the user, such as the stored path or the reason for a failure.
    """

    nwb_path: Path
    status: LinkStatus
    message: str = ""


class VideoProbe(NamedTuple):
    """Properties of a video file read from its header.

    Attributes:
        num_frames: Number of frames in the video.
        fps: Frame rate in frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    num_frames: int
    fps: float
    width: int
    height: int


def find_nwb_files(nwb_dir: Path) -> list[Path]:
    """List the NWB files directly inside a directory.

    Args:
        nwb_dir: Directory to look in. Subdirectories are not searched.

    Returns:
        Sorted paths of the ``.nwb`` files, leaving out hidden files such as the
        temporary copy written while a file is being modified.
    """
    return sorted(
        path for path in nwb_dir.glob("*.nwb") if path.is_file() and not path.name.startswith(".")
    )


def find_video(nwb_path: Path, video_dir: Path) -> Path | None:
    """Find the video that belongs to an NWB file.

    The video has the same name as the NWB file, with an ``.mp4`` extension.

    Args:
        nwb_path: The NWB file.
        video_dir: Directory holding the videos. Subdirectories are not searched.

    Returns:
        Path of the video, or None if there is no such file.
    """
    video_path = video_dir / f"{nwb_path.stem}{VIDEO_SUFFIX}"
    return video_path if video_path.is_file() else None


def probe_video(video_path: Path) -> VideoProbe:
    """Read the frame count, frame rate and frame size of a video.

    Args:
        video_path: Video file to inspect.

    Returns:
        The video's properties as reported by its header.

    Raises:
        OSError: If the video cannot be opened.
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Unable to open video file: {video_path}")
        return VideoProbe(
            num_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def describe_mismatch(probe: VideoProbe, info: NWBVideoLinkInfo) -> str | None:
    """Compare a video with the pose data of an NWB file.

    Args:
        probe: Properties of the candidate video.
        info: Frame count and rate of the pose data.

    Returns:
        A description of how they disagree, or None if they are consistent.
    """
    problems: list[str] = []
    if probe.num_frames != info.num_frames:
        problems.append(f"video has {probe.num_frames} frames, pose has {info.num_frames}")
    if not math.isclose(probe.fps, info.rate, rel_tol=_FPS_REL_TOL):
        problems.append(f"video is {probe.fps:g} fps, pose is {info.rate:g} fps")
    return "; ".join(problems) or None


def link_one(nwb_path: Path, video_dir: Path, *, dry_run: bool, verify: bool) -> LinkResult:
    """Link one NWB file to its video.

    Args:
        nwb_path: NWB file to modify.
        video_dir: Directory holding the videos.
        dry_run: Report what would happen without changing the file.
        verify: Check the video's frame count and rate against the pose data first.

    Returns:
        The outcome. Failures are reported in the result rather than raised, so one bad
        file does not stop a batch.
    """
    try:
        info = read_video_link_info(nwb_path)
    except (OSError, KeyError) as exc:
        return LinkResult(nwb_path, LinkStatus.ERROR, f"not a readable JABS pose NWB file: {exc}")

    video_path = find_video(nwb_path, video_dir)
    if video_path is None:
        return LinkResult(
            nwb_path, LinkStatus.NO_VIDEO, f"{nwb_path.stem}{VIDEO_SUFFIX} not found"
        )

    try:
        relative = relative_video_path(nwb_path, video_path)
    except ValueError as exc:
        return LinkResult(nwb_path, LinkStatus.ERROR, str(exc))

    if info.external_files:
        if relative in info.external_files:
            return LinkResult(nwb_path, LinkStatus.ALREADY_LINKED, relative)
        return LinkResult(
            nwb_path,
            LinkStatus.ERROR,
            f"already linked to {', '.join(info.external_files)}; leaving it unchanged",
        )

    dimension: tuple[int, int] | None = None
    if verify:
        try:
            probe = probe_video(video_path)
        except OSError as exc:
            return LinkResult(nwb_path, LinkStatus.ERROR, str(exc))
        if (mismatch := describe_mismatch(probe, info)) is not None:
            return LinkResult(nwb_path, LinkStatus.MISMATCH, mismatch)
        dimension = (probe.width, probe.height)

    if dry_run:
        return LinkResult(nwb_path, LinkStatus.WOULD_LINK, relative)

    try:
        stored = link_external_video(
            nwb_path,
            video_path,
            num_frames=info.num_frames,
            rate=info.rate,
            dimension=dimension,
        )
    except Exception as exc:
        # A batch should report a file it cannot modify and carry on with the rest.
        logger.error("Could not link %s to %s", nwb_path, video_path, exc_info=True)
        return LinkResult(nwb_path, LinkStatus.ERROR, str(exc))
    return LinkResult(nwb_path, LinkStatus.LINKED, stored)


def link_videos(
    nwb_files: list[Path],
    video_dir: Path,
    *,
    dry_run: bool,
    verify: bool,
    console: Console,
) -> list[LinkResult]:
    """Link every NWB file in a list to its video.

    Args:
        nwb_files: NWB files to process.
        video_dir: Directory holding the videos.
        dry_run: Report what would happen without changing any file.
        verify: Check each video's frame count and rate against the pose data first.
        console: Console to draw the progress bar on.

    Returns:
        One result per NWB file, in the order given.
    """
    results: list[LinkResult] = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    task = progress.add_task("Linking videos", total=len(nwb_files))
    with progress:
        for nwb_path in nwb_files:
            result = link_one(nwb_path, video_dir, dry_run=dry_run, verify=verify)
            logger.info("%s: %s %s", nwb_path.name, result.status.value, result.message)
            results.append(result)
            progress.advance(task)
    return results


def _print_summary(results: list[LinkResult], console: Console, *, verbose: bool) -> None:
    """Print the outcome counts and the files that need attention.

    Args:
        results: One result per NWB file.
        console: Console to print to.
        verbose: List every file instead of only those with a problem.
    """
    for result in results:
        if verbose or result.status.is_problem:
            style = _STATUS_STYLE[result.status]
            detail = f": {escape(result.message)}" if result.message else ""
            console.print(
                f"[{style}]{result.status.value:>14}[/{style}]  "
                f"{escape(result.nwb_path.name)}{detail}",
                highlight=False,
            )

    counts = {status: sum(r.status is status for r in results) for status in LinkStatus}
    parts = [
        f"[{_STATUS_STYLE[status]}]{count} {status.value}[/{_STATUS_STYLE[status]}]"
        for status, count in counts.items()
        if count
    ]
    console.print(f"{len(results)} NWB file(s): " + ", ".join(parts))


@click.command(name="link-nwb-video")
@click.argument(
    "nwb_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "video_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show which files would be linked without modifying any NWB file.",
)
@click.option(
    "--verify/--no-verify",
    default=True,
    show_default=True,
    help=(
        "Open each video and check that its frame count and frame rate match the pose "
        "data in the NWB file."
    ),
)
@click.pass_context
def link_nwb_video_command(
    ctx: click.Context,
    nwb_dir: Path,
    video_dir: Path,
    dry_run: bool,
    verify: bool,
) -> None:
    """Link JABS pose NWB files to their source videos.

    DANDI accepts a video only when an NWB file points to it, so each NWB file
    in NWB_DIR gets an ImageSeries in its acquisition that references the video with
    the same name and an .mp4 extension in VIDEO_DIR (for example, session1.nwb
    is linked to session1.mp4). NWB_DIR and VIDEO_DIR may be the same directory.

    The video is referenced, not copied. The reference is stored relative to the
    NWB file, which is how dandi organize --update-external-file-paths finds the
    video, so keep both directories where they are until dandi organize has run.

    Files that are already linked to their video are skipped, so the command can be
    run again. A file is left unchanged if its video is missing, cannot be opened, or
    has a different frame count or frame rate than the pose data. The exit status is
    1 if any file could not be linked.

    Examples:

    \b
    # NWB files and videos in different directories
    jabs-cli link-nwb-video /data/nwb /data/videos

    \b
    # NWB files and videos in the same directory
    jabs-cli link-nwb-video /data/session_files /data/session_files

    \b
    # See what would happen first
    jabs-cli link-nwb-video --dry-run /data/nwb /data/videos
    """
    nwb_files = find_nwb_files(nwb_dir)
    if not nwb_files:
        raise click.ClickException(f"No .nwb files found in {nwb_dir}")

    try:
        require_video_link_support()
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc

    console = Console()

    logger.info("Linking %d NWB file(s) in %s to videos in %s", len(nwb_files), nwb_dir, video_dir)
    results = link_videos(nwb_files, video_dir, dry_run=dry_run, verify=verify, console=console)
    _print_summary(results, console, verbose=ctx.obj["VERBOSE"] or dry_run)

    if any(result.status.is_problem for result in results):
        ctx.exit(1)
