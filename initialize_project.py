#!/usr/bin/env python

"""
initialize a rotta project directory

computes features if they do not exist
optional regenerate and overwrite existing feature h5 files
"""

import argparse

from multiprocessing import Pool

from src.labeler import Project
from src.feature_extraction import IdentityFeatures
from src.cli import cli_progress_bar


def worker(params: dict):
    """ work function used for multiprocessing Pool.imap_unordered """
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='store_true',
                        help='recompute features even if file already exists')
    parser.add_argument('-p', '--processes', default=4, type=int,
                        help="number of multiprocessing workers")
    parser.add_argument('project_dir')
    args = parser.parse_args()

    project = Project(args.project_dir)
    total_identities = project.total_project_identities

    print(f"initializing project directory: {args.project_dir}")

    def producer():
        """ producer for Pool.imap_unordered """
        for video in project.videos:
            for identity in project.load_pose_est(project.video_path(video)).identities:
                yield({
                    'video': video,
                    'identity': identity,
                    'project': project,
                    'force': args.force
                })

    # print the initial progress bar with 0% complete
    cli_progress_bar(0, total_identities,
                     prefix=" Computing Features:")

    # compute features in parallel
    pool = Pool(args.processes)
    complete = 0
    for _ in pool.imap_unordered(worker, producer()):
        # update progress bar
        complete += 1
        cli_progress_bar(complete, total_identities,
                         prefix=" Computing Features:")


if __name__ == '__main__':
    main()
