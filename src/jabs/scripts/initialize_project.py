#!/usr/bin/env python

"""Initialize a JABS project directory.

Computes features if they do not exist and can optionally regenerate and
overwrite existing feature H5 files.
"""

import json
import logging
import os
import time
from datetime import datetime
from multiprocessing.pool import Pool
from pathlib import Path

import click
from jsonschema.exceptions import ValidationError
from rich.progress import Progress

import jabs.feature_extraction
import jabs.pose_estimation
import jabs.project
from jabs.core.constants import CACHE_FORMAT_KEY
from jabs.core.enums import CacheFormat, ProjectDistanceUnit
from jabs.feature_extraction import DEFAULT_WINDOW_SIZE
from jabs.feature_extraction.feature_timing import FeatureTimings
from jabs.feature_extraction.feature_timing_report import render_timing_report
from jabs.project.video_manager import VideoManager
from jabs.schema.metadata import validate_metadata
from jabs.version import version_str
from jabs.video_reader import VideoReader

logger = logging.getLogger(__name__)

DEFAULT_PROCESSES = os.cpu_count() or 1


def generate_files_worker(params: dict) -> FeatureTimings | None:
    """worker function used for generating project feature and cache files for one video

    All identities in the video are processed with the same PoseEstimation object, so
    per-video work that feature groups cache on it (such as the social contact graph
    shared by every identity) is done once per video rather than once per identity.

    Returns:
        Feature compute times for every identity in the video if
        ``params["collect_timings"]`` is set, otherwise None.
    """
    project = params["project"]
    pose_est = project.load_pose_est(project.video_manager.video_path(params["video"]))
    cache_format = CacheFormat(params["cache_format"])
    timings = FeatureTimings() if params.get("collect_timings", False) else None
    op_settings = project.get_project_defaults()

    for identity in pose_est.identities:
        features = jabs.feature_extraction.IdentityFeatures(
            params["video"],
            identity,
            project.feature_dir,
            pose_est,
            force=params["force"],
            op_settings=op_settings,
            cache_format=cache_format,
            timings=timings,
        )

        # unlike per frame features, window features are not automatically
        # generated when opening the file. They are computed as needed based
        # on the requested window size. Force each window size to be
        # pre-computed by fetching it
        for w in params["window_sizes"]:
            _ = features.get_window_features(w, force=params["force"])

        _ = pose_est.get_identity_convex_hulls(identity)

    return timings


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


def validate_window_sizes(window_sizes: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Validate one or more window sizes.

    Window size must be at least one frame on each side of the current frame.
    """
    normalized: list[int] = []
    for window_size in window_sizes:
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise ValueError("window size must be an integer")
        if window_size < 1:
            raise ValueError("window size must be greater than or equal to 1")
        normalized.append(window_size)
    return tuple(normalized)


def window_size_type(_ctx: click.Context, _param: click.Parameter, value: tuple[int, ...]):
    """Validate one or more window sizes for the Click CLI."""
    try:
        return validate_window_sizes(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


def compute_project_features(
    project: jabs.project.Project,
    window_sizes: list[int],
    force: bool,
    pool: Pool,
    cache_format: CacheFormat = CacheFormat.PARQUET,
    collect_timings: bool = False,
) -> FeatureTimings | None:
    """Compute features for all identities in the project.

    Work is split into one job per video, so parallelism is limited to the number of
    videos in the project.

    Args:
        project: Project to compute features for.
        window_sizes: Window sizes to pre-compute.
        force: Recompute features even if they are already cached.
        pool: Worker pool used to compute features in parallel.
        cache_format: Feature cache storage format.
        collect_timings: Record how long each feature module takes to compute.

    Returns:
        Feature compute times merged across all identities if ``collect_timings`` is
        set, otherwise None.
    """

    def feature_job_producer():
        for video in project.video_manager.videos:
            yield {
                "video": video,
                "project": project,
                "force": force,
                "window_sizes": window_sizes,
                "cache_format": cache_format.value,
                "collect_timings": collect_timings,
            }

    timings = FeatureTimings() if collect_timings else None
    with Progress() as progress:
        task = progress.add_task(" Computing Features", total=len(project.video_manager.videos))
        for result in pool.imap_unordered(generate_files_worker, feature_job_producer()):
            if timings is not None and result is not None:
                timings.merge(result)
            progress.update(task, advance=1)
    return timings


def write_timing_report(
    report_path: Path,
    timings: FeatureTimings,
    top_n: int | None,
    metadata: dict[str, str],
    force: bool,
) -> None:
    """Write the feature compute time report to a Markdown file.

    Args:
        report_path: Destination file.
        timings: Feature compute times collected during feature generation.
        top_n: If given, list only the ``top_n`` slowest entries.
        metadata: Key/value pairs describing the run.
        force: Whether features were recomputed even if cached.

    Raises:
        OSError: If the report cannot be written.
    """
    notes = [
        "Times are wall-clock seconds summed across all identities and worker processes, "
        "so the total can exceed the elapsed time of the run.",
        "Window times are summed across all window sizes.",
        "`(group setup)` is shared work a feature group does before computing its features. "
        "Shared data a group computes lazily is attributed to the first feature that uses it.",
    ]
    if not force:
        notes.append(
            "Features loaded from the cache were not computed and are not timed. "
            "Run with `--force` to time every feature."
        )
    report = render_timing_report(timings, top_n=top_n, metadata=metadata, notes=notes)
    report_path.write_text(report)
    logger.info("Wrote feature timing report to %s", report_path)


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
        return metadata
    except json.JSONDecodeError as e:
        _exit_with_message(f"Error reading metadata file {metadata_path}: {e}")
    except OSError as e:
        _exit_with_message(f"Error opening metadata file {metadata_path}: {e}")
    except ValidationError as e:
        _exit_with_message(f"Metadata file {metadata_path} is not valid: {e.message}")


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
    cache_format: CacheFormat | None = None,
    timing_report: Path | None = None,
    timing_top_n: int | None = None,
) -> None:
    """Run project initialization and optional feature generation.

    Args:
        force: Recompute features even if they already exist.
        processes: Number of worker processes.
        window_sizes: Window sizes to pre-compute; the default is used if empty.
        force_pixel_distances: Use pixel distances even if the project supports cm.
        metadata_path: Optional project metadata JSON file.
        skip_feature_generation: Only initialize and validate the project.
        project_dir: JABS project directory.
        cache_format: Feature cache format, or None to keep the project setting.
        timing_report: If given, write a Markdown report of feature compute times here.
        timing_top_n: Limit the timing report to the slowest ``timing_top_n`` entries.
    """
    if timing_report is not None and skip_feature_generation:
        click.echo("Warning: --timing-report is ignored with --skip-feature-generation")
        timing_report = None

    pool = Pool(processes)
    try:
        validated_window_sizes = validate_window_sizes(window_sizes)

        # user didn't specify any window sizes, use default
        if not validated_window_sizes:
            resolved_window_sizes = [DEFAULT_WINDOW_SIZE]
        else:
            # make sure there are no duplicates
            resolved_window_sizes = list(set(validated_window_sizes))

        click.echo(f"Initializing project directory: {project_dir}")

        # first do a quick check to make sure the h5 files exist for each video
        videos = VideoManager.get_videos(project_dir)
        metadata = _load_metadata(metadata_path)

        # jabs-init performs a full up-front validation of the project, including
        # confirming each video and its pose file agree on frame count, so it fails
        # fast on an inconsistent project before any features are generated.
        project = jabs.project.Project(
            project_dir, enable_video_check=True, enable_session_tracker=False
        )
        distance_unit = project.feature_manager.distance_unit
        _apply_project_metadata(project, metadata)

        # Resolve the effective cache format. For new projects, Project.__init__ already
        # wrote CACHE_FORMAT_KEY into project.json, so existing_settings always has the key
        # (or it's an older HDf5 project that predates this setting key, which implies HDF5).
        # We only override it when the user explicitly passes --cache-format.
        existing_settings = dict(project.settings_manager.project_settings.get("settings", {}))
        if cache_format is None:
            cache_format = CacheFormat(existing_settings[CACHE_FORMAT_KEY])
        else:
            existing_settings[CACHE_FORMAT_KEY] = cache_format.value
            project.settings_manager.save_project_file({"settings": existing_settings})

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
            for video_name in videos:
                yield {"video": video_name, "project_dir": project_dir}

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
            start = time.perf_counter()
            timings = compute_project_features(
                project,
                resolved_window_sizes,
                force,
                pool,
                cache_format,
                collect_timings=timing_report is not None,
            )
            elapsed = time.perf_counter() - start

            if timing_report is not None and timings is not None:
                report_metadata = {
                    "Project": str(project_dir),
                    "Generated": datetime.now().isoformat(timespec="seconds"),
                    "JABS version": version_str(),
                    "Identities": str(project.total_project_identities),
                    "Window sizes": ", ".join(str(w) for w in sorted(resolved_window_sizes)),
                    "Worker processes": str(processes),
                    "Cache format": cache_format.value,
                    "Forced recompute": "yes" if force else "no",
                    "Elapsed feature generation time": f"{elapsed:.1f} s",
                }
                try:
                    write_timing_report(
                        timing_report, timings, timing_top_n, report_metadata, force
                    )
                except OSError as e:
                    _exit_with_message(f"Error writing timing report {timing_report}: {e}")
                click.echo(f"Feature timing report written to {timing_report}")

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
    help="Recompute features even if they already exist. Does not change the cache format; pass --cache-format explicitly to migrate.",
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
@click.option(
    "--cache-format",
    "cache_format",
    type=click.Choice([f.value for f in CacheFormat], case_sensitive=False),
    default=None,
    help=(
        "Feature cache storage format. If omitted, the existing project setting is "
        "preserved; new projects default to Parquet; projects created before this "
        "option existed default to HDF5."
    ),
)
@click.option(
    "--timing-report",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help=(
        "Write a Markdown report of how long each feature took to compute, sorted from "
        "slowest to fastest. Cached features are not timed; combine with --force to time "
        "every feature."
    ),
)
@click.option(
    "--timing-top-n",
    type=click.IntRange(min=1),
    default=None,
    help="Only list the N slowest features in the timing report (requires --timing-report).",
)
@click.argument("project_dir", type=click.Path(path_type=Path))
def main(
    force: bool,
    processes: int,
    window_sizes: tuple[int, ...],
    force_pixel_distances: bool,
    metadata: Path | None,
    skip_feature_generation: bool,
    cache_format: str | None,
    timing_report: Path | None,
    timing_top_n: int | None,
    project_dir: Path,
) -> None:
    """jabs-init."""
    if timing_top_n is not None and timing_report is None:
        raise click.UsageError("--timing-top-n requires --timing-report")

    run_initialize_project(
        force=force,
        processes=processes,
        window_sizes=window_sizes,
        force_pixel_distances=force_pixel_distances,
        metadata_path=metadata,
        skip_feature_generation=skip_feature_generation,
        project_dir=project_dir,
        cache_format=CacheFormat(cache_format) if cache_format is not None else None,
        timing_report=timing_report,
        timing_top_n=timing_top_n,
    )


if __name__ == "__main__":
    main()
