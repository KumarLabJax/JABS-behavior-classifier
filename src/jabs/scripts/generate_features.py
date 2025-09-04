#!/usr/bin/env python

"""initialize JABS features for a pose file

computes features if they do not exist
"""

import argparse
import sys
from pathlib import Path

from jabs.feature_extraction.features import IdentityFeatures
from jabs.pose_estimation import open_pose_file
from jabs.project import Project
from jabs.types import ProjectDistanceUnit


def generate_feature_cache(args):
    """
    Generate and cache features for each identity in a pose file.

    Loads the pose file, computes features for each identity using the specified
    settings, and caches the results in the given feature directory. Optionally
    generates window features if a window size is provided.

    Args:
        args: argparse Namespace object containing script arguments, including pose file path,
              pose version, feature directory, distance unit, window size, and fps.
    """
    distance_unit = ProjectDistanceUnit.CM if args.cm_units else ProjectDistanceUnit.PIXEL
    settings = Project.settings_by_pose_version(args.pose_version, distance_unit)
    if args.window_size is not None:
        settings["window_size"] = args.window_size
        cache_window = True
    else:
        cache_window = False

    pose_est = open_pose_file(args.pose_file)
    for curr_id in pose_est.identities:
        # Note: Features are still cached with the highest pose version.
        # It isn't until get_features is called that filtering occurs
        features = IdentityFeatures(
            args.pose_file,
            curr_id,
            args.feature_dir,
            pose_est,
            fps=args.fps,
            op_settings=settings,
            cache_window=cache_window,
        )
        # Window features are not automatically generated.
        if cache_window:
            _ = features.get_window_features(
                settings["window_size"], settings["social"], force=True
            )


def main():
    """jabs-features"""
    parser = argparse.ArgumentParser(prog=f"{script_name()} features")
    parser.add_argument(
        "--pose-file",
        required=True,
        type=Path,
        help="pose file to compute features for",
    )
    parser.add_argument(
        "--pose-version",
        required=True,
        type=int,
        help="pose version to calculate features",
    )
    parser.add_argument(
        "--feature-dir",
        required=True,
        type=Path,
        help="directory to write output features",
    )
    parser.add_argument(
        "--use-cm-distances",
        action="store_true",
        dest="cm_units",
        default=False,
        help="use cm distance units instead of pixel",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="window size for features (default none)",
    )
    parser.add_argument(
        "--fps", default=30, help="frames per second to use for feature calculation"
    )
    args = parser.parse_args()

    generate_feature_cache(args)


def script_name() -> str:
    """return the script name"""
    return Path(sys.argv[0]).name


if __name__ == "__main__":
    main()
