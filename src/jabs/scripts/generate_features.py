#!/usr/bin/env python

"""Deprecated entry point for computing JABS features for a pose file.

``jabs-features`` has been replaced by ``jabs-cli compute-features``. This module
is kept only as a transitional shim: it accepts the legacy argument surface,
prints a deprecation warning, translates the legacy options to their
``compute-features`` equivalents, and then runs that command.

Legacy option mapping:

* ``--pose-version`` is ignored. ``compute-features`` infers the pose version
  from the pose filename (e.g. ``*_pose_est_v6.h5``).
* ``--use-cm-distances`` selects cm units, which is the ``compute-features``
  default when the pose file provides a pixel-to-cm scale. Without the flag,
  ``--use-pixel-distances`` is forwarded to preserve the legacy pixel default.
* ``--window-size`` maps to a single ``-w`` value.
* ``--fps`` and ``--use-pose-hash`` are forwarded unchanged.

The feature cache is always written in Parquet format, which is also the
``compute-features`` default.
"""

import argparse
import sys
from pathlib import Path

from .cli.compute_features import compute_features_command


def _build_compute_features_args(args: argparse.Namespace) -> list[str]:
    """Translate legacy ``jabs-features`` arguments into ``compute-features`` arguments."""
    command_args = [
        "--pose-file",
        str(args.pose_file),
        "--feature-dir",
        str(args.feature_dir),
        "--fps",
        str(args.fps),
    ]

    # the legacy script defaulted to pixel units and opted into cm; compute-features
    # defaults to cm when the pose file has a pixel-to-cm scale and opts into pixel
    if not args.cm_units:
        command_args.append("--use-pixel-distances")

    if args.window_size is not None:
        command_args += ["-w", str(args.window_size)]

    if args.use_pose_hash:
        command_args.append("--use-pose-hash")

    return command_args


def main():
    """jabs-features (deprecated): forward to ``jabs-cli compute-features``."""
    script = Path(sys.argv[0]).name
    print(
        f"{script} is deprecated, use `jabs-cli compute-features` instead.",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser(
        description=(
            f"DEPRECATED: {script} forwards to `jabs-cli compute-features`. "
            "Use that command directly."
        )
    )
    parser.add_argument(
        "--pose-file",
        required=True,
        type=Path,
        help="pose file to compute features for",
    )
    parser.add_argument(
        "--pose-version",
        type=int,
        default=None,
        help="ignored, the pose version is inferred from the pose filename",
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
        "--fps", type=int, default=30, help="frames per second to use for feature calculation"
    )
    parser.add_argument(
        "--use-pose-hash",
        action="store_true",
        dest="use_pose_hash",
        default=False,
        help=(
            "include the pose file hash as a subdirectory level in the feature cache path "
            "(e.g. <feature-dir>/<video>/<pose-hash>/<identity>); "
            "prevents collisions when a shared cache dir is used across multiple pipelines"
        ),
    )
    args = parser.parse_args()

    if args.pose_version is not None:
        print(
            "--pose-version is ignored, the pose version is inferred from the pose filename.",
            file=sys.stderr,
        )

    compute_features_command.main(
        _build_compute_features_args(args),
        prog_name="jabs-cli compute-features",
    )


if __name__ == "__main__":
    main()
