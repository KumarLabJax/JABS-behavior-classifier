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
from src.labeler import Project
from src.feature_extraction import IdentityFeatures
from src.cli import cli_progress_bar
from src.video_stream import VideoStream


def generate_files_worker(params: dict):
    """ worker function used for generating project feature and cache files """
    project = params['project']
    pose_est = project.load_pose_est(
        project.video_path(params['video']))
    features = IdentityFeatures(params['video'], params['identity'],
                                project.feature_dir,
                                pose_est, force=params['force'])

    # TODO, allow user to specify different window size
    _ = features.get_window_features(5, force=params['force'])

    for identity in pose_est.identities:
        _ = pose_est.get_identity_convex_hulls(identity)


def validate_video(params: dict):
    """ worker function for validating project video """

    vid_path = params['project_dir'] / params['video']
    try:
        vid_frames = VideoStream.get_nframes_from_file(vid_path)
    except:
        return {'video': params['video'], 'okay': False,
                'message': "Unable to open video"}

    pose_path = src.pose_estimation.get_pose_path(vid_path)
    if src.pose_estimation.get_frames_from_file(pose_path) != vid_frames:
        return {'video': params['video'], 'okay': False,
                'message': "Video and Pose File frame counts differ"}

    try:
        _ = src.pose_estimation.open_pose_file(pose_path)
    except:
        return {'video': params['video'], 'okay': False,
                'message': "Unable to open pose file"}

    return {'video': params['video'], 'okay': True}


def match_to_pose(video: str, project_dir: Path):
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
    parser.add_argument('project_dir', type=Path)
    args = parser.parse_args()

    # worker pool for computing features in parallel
    pool = Pool(args.processes)

    print(f"Initializing project directory: {args.project_dir}")

    # first to a quick check to make sure the h5 files exist for each video
    videos = Project.get_videos(args.project_dir)

    # print the initial progress bar with 0% complete
    cli_progress_bar(0, len(videos),
                     prefix=" Checking for pose files:")

    # iterate over each vido and try to pair it with an h5 file
    complete = 0
    results = []
    for result in [match_to_pose(v, args.project_dir) for v in videos]:
        # update progress bar
        complete += 1
        cli_progress_bar(complete, len(videos),
                         prefix=" Checking for pose files:")
        results.append(result)

    failures = [r for r in results if r['okay'] is False]

    if failures:
        print(" The following errors were encountered. "
              "Please correct and run this script again:")
        for f in failures:
            print(f"  {f['video']}: {f['message']}")
        sys.exit(1)

    def validation_job_producer():
        for video in videos:
            yield({
                'video': video,
                'project_dir': args.project_dir
            })

    # print the initial progress bar with 0% complete
    cli_progress_bar(0, len(videos),
                     prefix=" Validating Project:")

    complete = 0
    results = []
    for result in pool.imap_unordered(validate_video,
                                      validation_job_producer()):
        # update progress bar
        complete += 1
        cli_progress_bar(complete, len(videos),
                         prefix=" Validating Project:")
        results.append(result)

    failures = [r for r in results if r['okay'] is False]

    if failures:
        print(" The following errors were encountered. "
              "Please correct and run this script again:")
        for f in failures:
            print(f"  {f['video']}: {f['message']}")
        sys.exit(1)

    # do additional project validation
    project = Project(args.project_dir)
    total_identities = project.total_project_identities

    # generate features / cache files for the project
    def feature_job_producer():
        """ producer for Pool.imap_unordered """
        for video in project.videos:
            for identity in project.load_pose_est(
                    project.video_path(video)).identities:
                yield ({
                    'video': video,
                    'identity': identity,
                    'project': project,
                    'force': args.force
                })

    # print the initial progress bar with 0% complete
    cli_progress_bar(0, total_identities,
                     prefix=" Computing Features:")

    # compute features in parallel
    complete = 0
    for _ in pool.imap_unordered(generate_files_worker, feature_job_producer()):
        # update progress bar
        complete += 1
        cli_progress_bar(complete, total_identities,
                         prefix=" Computing Features:")


if __name__ == '__main__':
    main()
