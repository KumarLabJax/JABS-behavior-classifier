"""Export a copy of a video with the JABS pose overlay drawn on every frame.

Works on any video/pose pair, with or without a JABS project: the pose file is
found beside the video unless ``--pose-file`` says otherwise.

\b
Example:
  jabs-cli export-video /data/videos/clip.avi
  jabs-cli export-video clip.avi -o clip_overlay.mp4 --no-segmentation
"""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from jabs.pose_estimation import get_pose_path, open_pose_file


@click.command(
    name="export-video",
    help="Export a copy of a video with the JABS pose overlay drawn on every frame.",
)
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output video path. Defaults to <video>_overlay.mp4 beside the source video.",
)
@click.option(
    "--pose-file",
    "pose_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Pose file to use. Defaults to the highest-version pose file beside the video.",
)
@click.option(
    "--segmentation/--no-segmentation",
    "draw_segmentation",
    default=True,
    help="Include segmentation contours alongside the pose skeleton. Requires a v6 or "
    "newer pose file; ignored otherwise. Use --no-segmentation for pose only.",
)
@click.option(
    "--codec",
    default=None,
    help="FourCC codec for the output video  [default: mp4v]. 'avc1' (H.264) gives "
    "smaller files but is absent from some OpenCV builds.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite the output file if it already exists.",
)
def export_video_command(
    video_path: Path,
    output_path: Path | None,
    pose_path: Path | None,
    draw_segmentation: bool,
    codec: str | None,
    force: bool,
) -> None:
    """Render the pose overlay onto every frame of a video and write it out."""
    # Imported here, not at module scope: click imports every command module to build
    # the CLI, and jabs.video_export pulls in Qt. A module-scope import would make the
    # whole of jabs-cli require an importable Qt (libEGL and friends), breaking every
    # other subcommand on the headless machines this tool is mostly run on.
    try:
        from jabs.video_export import DEFAULT_CODEC, VideoExportError, export_overlay_video
    except ImportError as e:
        # Rendering the skeleton uses Qt's painter. Qt is installed as a dependency but
        # needs system graphics libraries that headless machines often lack, so say what
        # is missing rather than surfacing a bare ImportError traceback.
        raise click.ClickException(
            "export-video needs Qt, which could not be imported. On a headless Linux "
            "machine this usually means the system graphics libraries are missing "
            f"(install libegl1 and libgl1).\n\nUnderlying error: {e}"
        ) from e

    if codec is None:
        codec = DEFAULT_CODEC

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_overlay.mp4")

    if output_path.exists() and not force:
        raise click.ClickException(f"{output_path} already exists. Use --force to overwrite.")

    if pose_path is None:
        try:
            pose_path = get_pose_path(video_path)
        except ValueError as e:
            raise click.ClickException(str(e)) from e

    console = Console()
    try:
        pose_est = open_pose_file(pose_path)
    except Exception as e:
        # Pose loading surfaces OSError, KeyError and h5py errors depending on how
        # the file is malformed; none of them are useful as a raw traceback.
        raise click.ClickException(f"Could not read pose file {pose_path}: {e}") from e
    console.print(f"Video:  {video_path}")
    console.print(f"Pose:   {pose_path}")
    console.print(f"Output: {output_path}")

    # Segmentation is optional even in v6+ pose files. Say so up front rather than
    # letting the user wait out a full export and wonder where the contours went.
    if draw_segmentation and not getattr(pose_est, "has_segmentation", False):
        console.print(
            "[yellow]This pose file carries no segmentation data; "
            "exporting the pose overlay only.[/yellow]"
        )
        draw_segmentation = False

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Rendering overlay", total=None)

        def on_progress(written: int, total: int) -> None:
            progress.update(task_id, completed=written, total=total)

        try:
            frames_written = export_overlay_video(
                video_path,
                output_path,
                pose_est,
                draw_segmentation=draw_segmentation,
                codec=codec,
                progress_callback=on_progress,
            )
        except VideoExportError as e:
            raise click.ClickException(str(e)) from e

    console.print(f"Wrote {frames_written:,} frames to {output_path}")
