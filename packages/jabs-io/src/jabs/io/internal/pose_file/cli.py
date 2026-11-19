"""The ``jabs-pose-file`` command: convert legacy pose files, and validate pose files.

::

    jabs-pose-file convert LEGACY.h5 OUT.h5 [--width W --height H --fps F] [--block-mb N]
        [--max-memory-gb G]
    jabs-pose-file validate FILE.h5 [--deep]

Exit codes: 0 on success, 1 when a conversion is refused or a file has a
validation error, 2 on a usage error (argparse's own). A refusal is the
user's input being outside what the converter accepts, so it is reported as
one ``error:`` line, never a traceback. Refusals include a file with a link or
storage outside itself, and one whose estimated peak memory, from its
declared shapes, is above ``--max-memory-gb``. Any other exception is a bug
and is left to raise.
"""

import argparse
import logging
import math
import sys

from jabs.io.internal.pose_file.convert import (
    ConversionReport,
    LegacyConversionError,
    VideoOverrides,
    convert_legacy,
)
from jabs.io.internal.pose_file.convert.convert import DEFAULT_MAX_MEMORY
from jabs.io.internal.pose_file.validate import validate

logger = logging.getLogger(__name__)

_MEBIBYTE = 2**20
_GIBIBYTE = 2**30


def _one_line(message: object) -> str:
    """Return a message with its line breaks and runs of whitespace collapsed to single spaces."""
    return " ".join(str(message).split())


def _positive_int(text: str) -> int:
    """Parse a positive integer option value."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not positive")
    return value


def _positive_float(text: str) -> float:
    """Parse a positive, finite number option value."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive, finite number")
    return value


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="jabs-pose-file",
        description="Convert legacy JABS pose files and validate jabs.pose-file files.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser(
        "convert",
        help="convert a legacy pose_est_v4..v8 file to a jabs.pose-file",
        description=(
            "Convert a legacy pose_est_v4..v8 HDF5 file to a jabs.pose-file. The legacy "
            "file is only read; the output is written atomically and deep-checked."
        ),
    )
    convert.add_argument("source", metavar="LEGACY.h5", help="the legacy pose file")
    convert.add_argument("destination", metavar="OUT.h5", help="where to write the pose file")
    convert.add_argument("--width", type=_positive_int, help="video frame width, in pixels")
    convert.add_argument("--height", type=_positive_int, help="video frame height, in pixels")
    convert.add_argument("--fps", type=_positive_float, help="video frame rate")
    convert.add_argument(
        "--block-mb",
        type=_positive_int,
        default=64,
        metavar="N",
        help="memory budget for one block of the widest component, in MiB (default: 64)",
    )
    convert.add_argument(
        "--max-memory-gb",
        type=_positive_float,
        default=DEFAULT_MAX_MEMORY / _GIBIBYTE,
        metavar="G",
        help=(
            "refuse a file whose estimated peak memory, from its declared shapes, is above "
            f"this many GiB (2^30 bytes) (default: {DEFAULT_MAX_MEMORY // _GIBIBYTE})"
        ),
    )

    check = commands.add_parser(
        "validate",
        help="validate a jabs.pose-file",
        description="Validate a jabs.pose-file. Prints one line per finding.",
    )
    check.add_argument("path", metavar="FILE.h5", help="the pose file")
    check.add_argument(
        "--deep",
        action="store_true",
        help="also decode every alternate and compare it with its baseline",
    )
    return parser


def _describe(report: ConversionReport) -> str:
    """Render a conversion report as one paragraph."""
    dimensions = ", ".join(f"{name}={size}" for name, size in report.dimensions.items())
    dropped = ", ".join(report.dropped) if report.dropped else "nothing"
    return (
        f"Converted {report.source} ({report.format}) to {report.destination} "
        f"in {report.seconds:.1f} s, dimensions {dimensions}. "
        f"{report.tail_slots} instance(s) placed in tail slots, "
        f"{report.tracklet_splits} tracklet split(s), "
        f"{report.longterm_mismatches} longterm_seg_id mismatch(es). "
        f"Left behind: {dropped}."
    )


def _convert(args: argparse.Namespace) -> int:
    """Run ``convert``."""
    video = VideoOverrides(width=args.width, height=args.height, fps=args.fps)
    try:
        report = convert_legacy(
            args.source,
            args.destination,
            video=video,
            block_bytes=args.block_mb * _MEBIBYTE,
            max_memory=int(args.max_memory_gb * _GIBIBYTE),
        )
    except (LegacyConversionError, OSError) as error:
        # One line per error: a deep-check refusal lists its findings on
        # separate lines, and a reader of stderr takes one line per error.
        print(f"error: {_one_line(error)}", file=sys.stderr)
        return 1
    print(_describe(report))
    return 0


def _validate(args: argparse.Namespace) -> int:
    """Run ``validate``."""
    # validate never raises for a malformed file, a non-HDF5 one included:
    # every problem comes back as a finding.
    findings = validate(args.path, deep=args.deep)
    for finding in findings:
        print(f"{finding.severity} {finding.check}: {_one_line(finding.message)}")
    return 1 if any(finding.severity == "error" for finding in findings) else 0


def main(argv: list[str] | None = None) -> int:
    """Run the ``jabs-pose-file`` command.

    Args:
        argv: The arguments, without the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        The exit code: 0 on success, 1 on a refused conversion or a file
        with a validation error. A usage error exits 2 through argparse.
    """
    args = _parser().parse_args(argv)
    if args.command == "convert":
        return _convert(args)
    return _validate(args)


if __name__ == "__main__":
    sys.exit(main())
