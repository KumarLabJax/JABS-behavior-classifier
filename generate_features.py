import argparse
from pathlib import Path
from src.project import Project
from src.pose_estimation import open_pose_file
from src.feature_extraction.features import IdentityFeatures
from src.types import ProjectDistanceUnit


def generate_feature_cache(args):
    distance_unit = ProjectDistanceUnit.CM if args.cm_units else ProjectDistanceUnit.PIXEL
    settings = Project.settings_by_pose_version(args.pose_version, distance_unit)
    if args.window_size is not None:
        settings['window_size'] = args.window_size
        cache_window = True
    else:
        cache_window = False

    pose_est = open_pose_file(args.pose_file)
    for curr_id in pose_est.identities:
        # Note: Features are still cached with the highest pose version.
        # It isn't until get_features is called that filtering occurs
        _ = IdentityFeatures(
            args.pose_file, curr_id, args.feature_dir, pose_est, fps=args.fps, op_settings=settings, cache_window=cache_window
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose-file', required=True, type=Path,
                        help='pose file to compute features for')
    parser.add_argument('--pose-version', required=True, type=int,
                        help="pose version to calculate features")
    parser.add_argument('--feature-dir', required=True, type=Path,
                        help="directory to write output features")
    parser.add_argument('--use-cm-distances', action='store_true',
                        dest="cm_units", default=False,
                        help="use cm distance units instead of pixel")
    parser.add_argument('--window-size', type=int, default=None,
                        help="window size for features (default none)")
    parser.add_argument('--fps', default=30,
                        help="frames per second to use for feature calculation")
    args = parser.parse_args()

    generate_feature_cache(args)


if __name__ == '__main__':
    main()
