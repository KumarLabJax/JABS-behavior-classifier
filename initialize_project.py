#!/usr/bin/env python

"""
initialize a rotta project directory

computes features if they do not exist
optional regenerate and overwrite existing feature h5 files
"""

import argparse

from src.labeler import Project
from src.feature_extraction import IdentityFeatures
from src.cli import cli_progress_bar


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='store_true',
                        help='recompute features even if file already exists')
    parser.add_argument('project_dir')
    args = parser.parse_args()

    project = Project(args.project_dir)
    total_identities = project.total_project_identities

    print(f"initializing project directory: {args.project_dir}")

    completed = 0

    for video in project.videos:

        pose_est = project.load_pose_est(project.video_path(video))

        for identity in pose_est.identities:

            # print current progress bar
            cli_progress_bar(completed, total_identities,
                             prefix=" Computing Features:",
                             suffix=f"[{video} - {identity}]")

            features = IdentityFeatures(video, identity,
                                        project.feature_dir,
                                        pose_est, force=args.force)

            # TODO, allow user to specify different window size
            _ = features.get_window_features(5, force=args.force)

            completed += 1

    # print completed progress bar
    cli_progress_bar(completed, total_identities,
                     prefix=" Computing Features:")


if __name__ == '__main__':
    main()
