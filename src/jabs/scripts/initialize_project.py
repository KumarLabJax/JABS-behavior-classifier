#!/usr/bin/env python

"""Initialize a JABS project directory.

Computes features if they do not exist and can optionally regenerate and
overwrite existing feature H5 files.
"""

import json
import os
from multiprocessing import Pool
from pathlib import Path

import click
from jsonschema.exceptions import ValidationError
from rich.progress import Progress

import jabs.feature_extraction
import jabs.pose_estimation
import jabs.project
from jabs.core.enums import ProjectDistanceUnit
from jabs.project.video_manager import VideoManager
from jabs.schema.metadata import validate_metadata
from jabs.video_reader import VideoReader

DEFAULT_WINDOW_SIZE = 5
DEFAULT_PROCESSES = os.cpu_count() or 1


def generate_files_worker(params: dict):
    """worker function used for generating project feature and cache files"""
    project = params["project"]
    pose_est = project.load_pose_est(project.video_manager.video_path(params["video"]))

    features = jabs.feature_extraction.IdentityFeatures(
        params["video"],
        params["identity"],
        project.feature_dir,
        pose_est,
        force=params["force"],
        op_settings=project.get_project_defaults(),
    )

    # unlike per frame features, window features are not automatically
    # generated when opening the file. They are computed as needed based
    # on the requested window size. Force each window size to be
    # pre-computed by fetching it
    for w in params["window_sizes"]:
        # get the social features if they are supported, although this doesn't
        # matter with current implementation, as they are always computed if
        # the file supports them, they are just not included in the returned
        # features if this param is false
        use_social = pose_est.format_major_version > 2
        _ = features.get_window_features(w, use_social, force=params["force"])

    for identity in pose_est.identities:
        _ = pose_est.get_identity_convex_hulls(identity)


def validate_video_worker(params: dict):
    """worker function for validating project video"""
    vid_path = params["project_dir"] / params["video"]

    # make sure we can open the video
    try:
        vid_frames = VideoReader.get_nframes_from_file(vid_path)
    except OSError:
        return {
            "video": params["video"],
            "okay": False,
            "message": "Unable to open video",
        }

    # make sure the video and pose file have the same number of frames
    pose_path = jabs.pose_estimation.get_pose_path(vid_path)
    if jabs.pose_estimation.get_frames_from_file(pose_path) != vid_frames:
        return {
            "video": params["video"],
            "okay": False,
            "message": "Video and Pose File frame counts differ",
        }

    # make sure we can initialize a PoseEstimation object from this pose file
    try:
        _ = jabs.pose_estimation.open_pose_file(pose_path)
    except OSError:
        return {
            "video": params["video"],
            "okay": False,
            "message": "Unable to open pose file",
        }

    return {"video": params["video"], "okay": True}


def match_to_pose(video: str, project_dir: Path):
    """make sure a video has a corresponding h5 file"""
    path = project_dir / video

    try:
        _ = jabs.pose_estimation.get_pose_path(path)
    except ValueError:
        return {"video": video, "okay": False, "message": "Pose file not found"}
    return {"video": video, "okay": True}


def window_size_type(_ctx: click.Context, _param: click.Parameter, value: tuple[int, ...]):
    """Validate one or more window sizes for the Click CLI.

    Window size must be at least one frame on each side of the current frame.
    """
    for window_size in value:
        if window_size < 1:
            raise click.BadParameter("window size must be greater than or equal to 1")
    return value


def compute_project_features(
    project: jabs.project.Project,
    window_sizes: list[int],
    force: bool,
    pool: Pool,
) -> None:
    """Compute features for all identities in the project."""

    def feature_job_producer():
        for video in project.video_manager.videos:
            for identity in project.load_pose_est(
                project.video_manager.video_path(video)
            ).identities:
                yield {
                    "video": video,
                    "identity": identity,
                    "project": project,
                    "force": force,
                    "window_sizes": window_sizes,
                }

    total_identities = project.total_project_identities
    with Progress() as progress:
        task = progress.add_task(" Computing Features", total=total_identities)
        for _ in pool.imap_unordered(generate_files_worker, feature_job_producer()):
            progress.update(task, advance=1)


def _exit_with_message(message: str) -> None:
    """Print a user-facing error message and exit with status 1."""
    click.echo(message)
    raise click.exceptions.Exit(1)


def _load_metadata(metadata_path: Path | None) -> dict | None:
    """Load and validate project metadata if supplied."""
    if metadata_path is None:
        return None

    try:
        metadata = json.loads(metadata_path.read_text())
        validate_metadata(metadata)
    except json.JSONDecodeError as e:
        _exit_with_message(f"Error reading metadata file {metadata_path}: {e}")
    except OSError as e:
        _exit_with_message(f"Error opening metadata file {metadata_path}: {e}")
    except ValidationError as e:
        _exit_with_message(f"Metadata file {metadata_path} is not valid: {e.message}")

    return metadata


def _apply_project_metadata(project: jabs.project.Project, metadata: dict | None) -> None:
    """Merge or replace project metadata when requested."""
    if not metadata:
        return

    has_metadata = project.settings_manager.project_metadata != {}

    if not has_metadata:
        for video in project.video_manager.videos:
            if project.settings_manager.video_metadata(video) != {}:
                has_metadata = True
                break

    replace = True
    if has_metadata:
        response = (
            input(
                "Metadata already exists. Apply new metadata by [M]erge (default) or [R]eplace (clear existing)? [M/r]: "
            )
            .strip()
            .lower()
        )
        replace = response == "r"

    project.settings_manager.set_project_metadata(metadata, replace=replace)


def _exit_on_validation_failures(failures: list[dict]) -> None:
    """Print validation failures and exit if any were encountered."""
    if not failures:
        return

    click.echo(" The following errors were encountered, please correct and run this script again:")
    for failure in failures:
        click.echo(f"  {failure['video']}: {failure['message']}")
    raise click.exceptions.Exit(1)


def run_initialize_project(
    force: bool,
    processes: int,
    window_sizes: tuple[int, ...],
    force_pixel_distances: bool,
    metadata_path: Path | None,
    skip_feature_generation: bool,
    project_dir: Path,
) -> None:
    """Run project initialization and optional feature generation."""
    pool = Pool(processes)
    try:
        # user didn't specify any window sizes, use default
        if not window_sizes:
            resolved_window_sizes = [DEFAULT_WINDOW_SIZE]
        else:
            # make sure there are no duplicates
            resolved_window_sizes = list(set(window_sizes))

        click.echo(f"Initializing project directory: {project_dir}")

        # first do a quick check to make sure the h5 files exist for each video
        videos = VideoManager.get_videos(project_dir)
        metadata = _load_metadata(metadata_path)

        project = jabs.project.Project(project_dir, enable_session_tracker=False)
        distance_unit = project.feature_manager.distance_unit
        _apply_project_metadata(project, metadata)

        # iterate over each video and try to pair it with an h5 file
        # this test is quick, don't bother to parallelize
        results = []
        with Progress() as progress:
            task = progress.add_task(" Checking for pose files", total=len(videos))
            for video in videos:
                results.append(match_to_pose(video, project_dir))
                progress.update(task, advance=1)

        _exit_on_validation_failures([result for result in results if result["okay"] is False])

        # check project other errors such as being unable to open pose files,
        # pose file and video frame number mismatch, etc

        def validation_job_producer():
            for video in videos:
                yield {"video": video, "project_dir": project_dir}

        # do work in parallel (not really necessary for this test, but we already
        # have the worker pool for generating features)
        with Progress() as progress:
            results = []
            task = progress.add_task(" Validating videos", total=len(videos))
            for result in pool.imap_unordered(validate_video_worker, validation_job_producer()):
                progress.update(task, advance=1)
                results.append(result)

        _exit_on_validation_failures([result for result in results if result["okay"] is False])

        # compute features in parallel, this might take a while
        if not skip_feature_generation:
            compute_project_features(project, resolved_window_sizes, force, pool)

        # save window sizes to project settings
        deduped_window_sizes = set(
            project.settings_manager.project_settings.get("window_sizes", [])
            + resolved_window_sizes
        )
        project.settings_manager.save_project_file({"window_sizes": list(deduped_window_sizes)})

        if not skip_feature_generation:
            click.echo("\n" + "-" * 70)
            if force_pixel_distances:
                click.echo("Features computed using pixel distances.")
            elif distance_unit == ProjectDistanceUnit.PIXEL:
                click.echo("One or more pose files did not have the cm_per_pixel attribute")
                click.echo(" Falling back to using pixel distances")
            else:
                click.echo("Features computed using CM distances")
            click.echo("-" * 70)
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()


@click.command(name="jabs-init", context_settings={"max_content_width": 120}, help=__doc__)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="recompute features even if file already exists",
)
@click.option(
    "-p",
    "--processes",
    default=DEFAULT_PROCESSES,
    show_default=True,
    type=click.IntRange(min=1),
    help="number of multiprocessing workers to use; defaults to the logical CPU count",
)
@click.option(
    "-w",
    "window_sizes",
    multiple=True,
    type=int,
    callback=window_size_type,
    metavar="WINDOW_SIZE",
    help="Specify window sizes to use for computing window "
    "features. Argument can be repeated to specify "
    "multiple sizes (e.g. -w 2 -w 5). Size is number "
    "of frames before and after the current frame to "
    "include in the window. For example, '-w 2' "
    "results in a window size of 5 (2 frames before, "
    "2 frames after, plus the current frame). If no "
    "window size is specified, a default of "
    f"{DEFAULT_WINDOW_SIZE} will "
    "be used.",
)
@click.option(
    "--force-pixel-distances",
    is_flag=True,
    help="use pixel distances when computing features even if project supports cm",
)
@click.option(
    "--metadata",
    type=click.Path(path_type=Path),
    help="path to a JSON file containing project metadata to be validated and injected into the project",
)
@click.option(
    "--skip-feature-generation",
    is_flag=True,
    help="Skip feature calculation and only initialize/validate the project",
)
@click.argument("project_dir", type=click.Path(path_type=Path))
def main(
    force: bool,
    processes: int,
    window_sizes: tuple[int, ...],
    force_pixel_distances: bool,
    metadata: Path | None,
    skip_feature_generation: bool,
    project_dir: Path,
) -> None:
    """jabs-init."""
    run_initialize_project(
        force=force,
        processes=processes,
        window_sizes=window_sizes,
        force_pixel_distances=force_pixel_distances,
        metadata_path=metadata,
        skip_feature_generation=skip_feature_generation,
        project_dir=project_dir,
    )


if __name__ == "__main__":
    main()
