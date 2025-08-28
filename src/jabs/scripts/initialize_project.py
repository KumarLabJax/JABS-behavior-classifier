#!/usr/bin/env python

"""initialize a JABS project directory

computes features if they do not exist
optional regenerate and overwrite existing feature h5 files
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

from jsonschema.exceptions import ValidationError
from rich.progress import Progress

import jabs.feature_extraction
import jabs.pose_estimation
import jabs.project
from jabs.project.video_manager import VideoManager
from jabs.schema.metadata import validate_metadata
from jabs.types import ProjectDistanceUnit
from jabs.video_reader import VideoReader

DEFAULT_WINDOW_SIZE = 5


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


def window_size_type(x):
    """argparse type for window size

    We use this instead of an int type because we want to argparse to validate that the window size is valid.
    """
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError("window size must be greater than or equal to 1")
    return x


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


def main():
    """jabs-init"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="recompute features even if file already exists",
    )
    parser.add_argument(
        "-p",
        "--processes",
        default=4,
        type=int,
        help="number of multiprocessing workers",
    )
    parser.add_argument(
        "-w",
        dest="window_sizes",
        action="append",
        type=window_size_type,
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
    parser.add_argument(
        "--force-pixel-distances",
        action="store_true",
        help="use pixel distances when computing features even if project supports cm",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="path to a JSON file containing project metadata to be validated and injected into the project",
    )
    parser.add_argument(
        "--skip-feature-generation",
        action="store_true",
        help="Skip feature calculation and only initialize/validate the project",
    )
    parser.add_argument("project_dir", type=Path)
    args = parser.parse_args()

    # worker pool for computing features in parallel
    pool = Pool(args.processes)

    # user didn't specify any window sizes, use default
    if args.window_sizes is None:
        window_sizes = [DEFAULT_WINDOW_SIZE]
    else:
        # make sure there are no duplicates
        window_sizes = list(set(args.window_sizes))

    print(f"Initializing project directory: {args.project_dir}")

    # first to a quick check to make sure the h5 files exist for each video
    videos = VideoManager.get_videos(args.project_dir)

    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata.read_text())
            validate_metadata(metadata)
        except json.JSONDecodeError as e:
            print(f"Error reading metadata file {args.metadata}: {e}")
            sys.exit(1)
        except OSError as e:
            print(f"Error opening metadata file {args.metadata}: {e}")
            sys.exit(1)
        except ValidationError as e:
            print(f"Metadata file {args.metadata} is not valid: {e.message}")
            sys.exit(1)

    project = jabs.project.Project(args.project_dir, enable_session_tracker=False)
    distance_unit = project.feature_manager.distance_unit
    if metadata:
        has_metadata = False

        if project.settings_manager.project_metadata != {}:
            has_metadata = True

        for video in project.video_manager.videos:
            if project.settings_manager.video_metadata(video) != {}:
                has_metadata = True
                break

        if has_metadata and not args.force:
            response = (
                input("Warning: Project already has metadata. Overwrite? [y/N]: ").strip().lower()
            )
            if response != "y":
                print("Aborting. Use --force to overwrite without prompt.")
                sys.exit(1)
        project.settings_manager.set_project_metadata(metadata)

    # iterate over each video and try to pair it with an h5 file
    # this test is quick, don't bother to parallelize
    results = []
    with Progress() as progress:
        task = progress.add_task(" Checking for pose files", total=len(videos))
        for v in videos:
            results.append(match_to_pose(v, args.project_dir))
            progress.update(task, advance=1)

    failures = [r for r in results if r["okay"] is False]

    if failures:
        print(" The following errors were encountered, please correct and run this script again:")
        for f in failures:
            print(f"  {f['video']}: {f['message']}")
        sys.exit(1)

    # check project other errors such as being unable to open pose files,
    # pose file and video frame number missmatch, etc

    def validation_job_producer():
        for video in videos:
            yield ({"video": video, "project_dir": args.project_dir})

    # do work in parallel (not really necessary for this test, but we already
    # have the work pool for generating features)
    with Progress() as progress:
        results = []
        task = progress.add_task(" Validating videos", total=len(videos))
        for result in pool.imap_unordered(validate_video_worker, validation_job_producer()):
            # update progress bar
            progress.update(task, advance=1)
            results.append(result)

    failures = [r for r in results if r["okay"] is False]

    if failures:
        print(" The following errors were encountered, please correct and run this script again:")
        for f in failures:
            print(f"  {f['video']}: {f['message']}")
        sys.exit(1)

    # compute features in parallel, this might take a while
    if not args.skip_feature_generation:
        compute_project_features(project, window_sizes, args.force, pool)

    pool.close()

    # save window sizes to project settings
    deduped_window_sizes = set(
        project.settings_manager.project_settings.get("window_sizes", []) + window_sizes
    )
    project.settings_manager.save_project_file({"window_sizes": list(deduped_window_sizes)})

    if not args.skip_feature_generation:
        print("\n" + "-" * 70)
        if args.force_pixel_distances:
            print("Features computed using pixel distances.")
        elif distance_unit == ProjectDistanceUnit.PIXEL:
            print("One or more pose files did not have the cm_per_pixel attribute")
            print(" Falling back to using pixel distances")
        else:
            print("Features computed using CM distances")
        print("-" * 70)


if __name__ == "__main__":
    main()
