#!/usr/bin/env python

"""
initialize a rotta project directory

computes features if they do not exist
optional regenerate and overwrite existing feature h5 files
"""

import argparse
import sys

from multiprocessing import Pool
from pathlib import Path

import src.pose_estimation
from src.project import Project
from src.feature_extraction import IdentityFeatures
from src.cli import cli_progress_bar
from src.video_stream import VideoStream

DEFAULT_WINDOW_SIZE = 5


def generate_files_worker(params: dict):
    """ worker function used for generating project feature and cache files """
    project = params['project']
    pose_est = project.load_pose_est(
        project.video_path(params['video']))
    features = IdentityFeatures(params['video'], params['identity'],
                                project.feature_dir,
                                pose_est, force=params['force'])

    for w in params['window_sizes']:
        _ = features.get_window_features(w, force=params['force'])

    for identity in pose_est.identities:
        _ = pose_est.get_identity_convex_hulls(identity)


def validate_video_worker(params: dict):
    """ worker function for validating project video """

    vid_path = params['project_dir'] / params['video']

    # make sure we can open the video
    try:
        vid_frames = VideoStream.get_nframes_from_file(vid_path)
    except:
        return {'video': params['video'], 'okay': False,
                'message': "Unable to open video"}

    # make sure the video and pose file have the same number of frames
    pose_path = src.pose_estimation.get_pose_path(vid_path)
    if src.pose_estimation.get_frames_from_file(pose_path) != vid_frames:
        return {'video': params['video'], 'okay': False,
                'message': "Video and Pose File frame counts differ"}

    # make sure we can initialize a PoseEstimation object from this pose file
    try:
        _ = src.pose_estimation.open_pose_file(pose_path)
    except:
        return {'video': params['video'], 'okay': False,
                'message': "Unable to open pose file"}

    return {'video': params['video'], 'okay': True}


def match_to_pose(video: str, project_dir: Path):
    """ make sure a video has a corresponding h5 file """
    path = project_dir / video

    try:
        _ = src.pose_estimation.get_pose_path(path)
    except ValueError:
        return {'video': video, 'okay': False,
                'message': "Pose file not found"}
    return {'video': video, 'okay': True}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='store_true',
                        help='recompute features even if file already exists')
    parser.add_argument('-p', '--processes', default=4, type=int,
                        help="number of multiprocessing workers")
    parser.add_argument('-w', dest='window_sizes', action='append',
                        type=int, metavar='WINDOW_SIZE',
                        help="Specify window sizes to use for computing window "
                             "features. Argument can be repeated to specify "
                             "multiple sizes (e.g. -w 2 -w 5). Size is number "
                             "of frames before and after the current frame to "
                             "include in the window. For example, '-w 2' "
                             "results in a window size of 5 (2 frames before, "
                             "2 frames after, plus the current frame). If no "
                             "window size is specified, a default of "
                             f"{DEFAULT_WINDOW_SIZE} will "
                             "be used.")
    parser.add_argument('project_dir', type=Path)
    args = parser.parse_args()

    # worker pool for computing features in parallel
    pool = Pool(args.processes)

    # user didn't specify any window sizes, default to 5
    if args.window_sizes is None:
        window_sizes = [DEFAULT_WINDOW_SIZE]
    else:
        # make sure there are no duplicates
        window_sizes = set(args.window_sizes)

    print(f"Initializing project directory: {args.project_dir}")

    # first to a quick check to make sure the h5 files exist for each video
    videos = Project.get_videos(args.project_dir)

    # print the initial progress bar with 0% complete
    cli_progress_bar(0, len(videos),
                     prefix=" Checking for pose files: ")

    # iterate over each video and try to pair it with an h5 file
    # this test is quick, don't bother to parallelize
    complete = 0
    results = []
    for result in [match_to_pose(v, args.project_dir) for v in videos]:
        # update progress bar
        complete += 1
        cli_progress_bar(complete, len(videos),
                         prefix=" Checking for pose files: ")
        results.append(result)

    failures = [r for r in results if r['okay'] is False]

    if failures:
        print(" The following errors were encountered, "
              "please correct and run this script again:")
        for f in failures:
            print(f"  {f['video']}: {f['message']}")
        sys.exit(1)

    # check project other errors such as being unable to open pose files,
    # pose file and video frame number missmatch, etc

    def validation_job_producer():
        for video in videos:
            yield({
                'video': video,
                'project_dir': args.project_dir
            })

    # print the initial progress bar with 0% complete
    cli_progress_bar(0, len(videos),
                     prefix=" Validating Project: ")

    complete = 0
    results = []
    # do work in parallel (not really necessary for this test, but we already
    # have the work pool for generating features)
    for result in pool.imap_unordered(validate_video_worker,
                                      validation_job_producer()):
        # update progress bar
        complete += 1
        cli_progress_bar(complete, len(videos),
                         prefix=" Validating Project: ")
        results.append(result)

    failures = [r for r in results if r['okay'] is False]

    if failures:
        print(" The following errors were encountered, "
              "please correct and run this script again:")
        for f in failures:
            print(f"  {f['video']}: {f['message']}")
        sys.exit(1)

    # generate features -- this might be very slow
    project = Project(args.project_dir)
    total_identities = project.total_project_identities

    def feature_job_producer():
        """ producer for Pool.imap_unordered """
        for video in project.videos:
            for identity in project.load_pose_est(
                    project.video_path(video)).identities:
                yield ({
                    'video': video,
                    'identity': identity,
                    'project': project,
                    'force': args.force,
                    'window_sizes': window_sizes
                })

    # print the initial progress bar with 0% complete
    cli_progress_bar(0, total_identities,
                     prefix=" Computing Features: ")

    # compute features in parallel
    complete = 0
    for _ in pool.imap_unordered(generate_files_worker, feature_job_producer()):
        # update progress bar
        complete += 1
        cli_progress_bar(complete, total_identities,
                         prefix=" Computing Features: ")

    pool.close()


if __name__ == '__main__':
    main()
