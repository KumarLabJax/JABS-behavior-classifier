"""Find the longest positive behavior bouts across a directory of JABS prediction files.

Scans a directory of JABS HDF5 prediction files, extracts contiguous runs
("bouts") of positive predictions from the *postprocessed* prediction dataset
(``predicted_class_postprocessed``), and writes the top N longest bouts to a CSV
file.

Usage::

    uv run python scratch/top_prediction_bouts.py /path/to/predictions -o top_bouts.csv
    uv run python scratch/top_prediction_bouts.py /path/to/predictions --top-n 250
    uv run python scratch/top_prediction_bouts.py /path/to/predictions --behavior Drinking

Prediction file layout (see jabs.io.internal.prediction.hdf5)::

    /predictions/<safe_behavior_name>/predicted_class_postprocessed
        shape: (n_identities, n_frames), values: -1 = no prediction,
               0 = not behavior, 1 = behavior

The video file name is derived from the prediction file name by replacing the
``_behavior.h5`` suffix with ``.mp4``, e.g.::

    org-3-prod.study_617.cage_7485.2026-03-15.15.32_behavior.h5
    -> org-3-prod.study_617.cage_7485.2026-03-15.15.32.mp4
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

logger = logging.getLogger("top_prediction_bouts")

PREDICTION_SUFFIX = "_behavior.h5"
POSTPROCESSED_DATASET = "predicted_class_postprocessed"


@dataclass(frozen=True)
class Bout:
    """One contiguous run of positive predictions for a single identity.

    Attributes:
        video: Video file name the bout came from.
        behavior: Behavior group name in the prediction file.
        identity: Row index into the prediction array (JABS identity index).
        start: First frame of the bout (inclusive).
        end: Last frame of the bout (inclusive).
    """

    video: str
    behavior: str
    identity: int
    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the bout length in frames (inclusive of both endpoints)."""
        return self.end - self.start + 1


def video_name_from_prediction_file(prediction_file: Path) -> str:
    """Return the source video file name for a prediction file.

    Args:
        prediction_file: Path to a JABS HDF5 prediction file.

    Returns:
        Video file name, e.g. ``<stem>.mp4``.
    """
    name = prediction_file.name
    if name.endswith(PREDICTION_SUFFIX):
        return f"{name[: -len(PREDICTION_SUFFIX)]}.mp4"

    logger.warning(
        "%s does not end with %r; falling back to the file stem for the video name",
        name,
        PREDICTION_SUFFIX,
    )
    return f"{prediction_file.stem}.mp4"


def find_runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Find contiguous ``True`` runs in a 1-D boolean array.

    Args:
        mask: Boolean array, one element per frame.

    Returns:
        List of ``(start, end)`` frame index pairs, both endpoints inclusive.
    """
    if mask.size == 0:
        return []

    # pad with False on both ends so every run has a rising and falling edge
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts = edges[0::2]
    ends = edges[1::2] - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def bouts_from_prediction_file(
    prediction_file: Path,
    *,
    positive_class: int,
    behavior_filter: str | None,
) -> list[Bout]:
    """Extract positive postprocessed prediction bouts from one prediction file.

    Args:
        prediction_file: Path to a JABS HDF5 prediction file.
        positive_class: Predicted class value treated as "behavior present".
        behavior_filter: When set, only this behavior group is considered
            (matched case-insensitively against the group name).

    Returns:
        All positive bouts found in the file, one entry per contiguous run per
        identity. Behavior groups without a postprocessed dataset are skipped
        with a warning.
    """
    video = video_name_from_prediction_file(prediction_file)
    bouts: list[Bout] = []

    with h5py.File(prediction_file, "r") as h5:
        prediction_group = h5.get("predictions")
        if prediction_group is None:
            logger.warning("Skipping %s: no /predictions group", prediction_file.name)
            return []

        for key, behavior_group in prediction_group.items():
            if key == "external_identity_mapping" or not isinstance(behavior_group, h5py.Group):
                continue
            if behavior_filter is not None and key.lower() != behavior_filter.lower():
                continue

            if POSTPROCESSED_DATASET not in behavior_group:
                logger.warning(
                    "Skipping %s/%s: no %s dataset (predictions were not postprocessed)",
                    prediction_file.name,
                    key,
                    POSTPROCESSED_DATASET,
                )
                continue

            predictions = np.atleast_2d(behavior_group[POSTPROCESSED_DATASET][()])
            for identity, identity_predictions in enumerate(predictions):
                for start, end in find_runs(identity_predictions == positive_class):
                    bouts.append(
                        Bout(
                            video=video,
                            behavior=key,
                            identity=identity,
                            start=start,
                            end=end,
                        )
                    )

    logger.debug("%s: found %d bouts", prediction_file.name, len(bouts))
    return bouts


def collect_bouts(
    prediction_dir: Path,
    *,
    positive_class: int,
    behavior_filter: str | None,
) -> list[Bout]:
    """Collect positive bouts from every prediction file in a directory.

    Args:
        prediction_dir: Directory containing JABS HDF5 prediction files.
        positive_class: Predicted class value treated as "behavior present".
        behavior_filter: When set, only this behavior group is considered.

    Returns:
        All bouts found across all readable prediction files.
    """
    prediction_files = sorted(prediction_dir.glob("*.h5"))
    if not prediction_files:
        raise SystemExit(f"No .h5 files found in {prediction_dir}")

    logger.info("Scanning %d prediction files in %s", len(prediction_files), prediction_dir)

    bouts: list[Bout] = []
    for prediction_file in prediction_files:
        try:
            bouts.extend(
                bouts_from_prediction_file(
                    prediction_file,
                    positive_class=positive_class,
                    behavior_filter=behavior_filter,
                )
            )
        except (OSError, KeyError):
            logger.exception("Skipping unreadable prediction file: %s", prediction_file.name)

    return bouts


def write_csv(bouts: list[Bout], output_path: Path) -> None:
    """Write bouts to CSV, longest first.

    Args:
        bouts: Bouts to write, in the order they should appear.
        output_path: Destination CSV path (parent directories are created).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["video", "identity", "bout_start_frame", "bout_end_frame", "bout_length"])
        for bout in bouts:
            writer.writerow([bout.video, bout.identity, bout.start, bout.end, bout.length])


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "prediction_dir",
        type=Path,
        help="directory containing JABS HDF5 prediction files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("top_bouts.csv"),
        help="output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=100,
        help="number of longest bouts to report (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--behavior",
        default=None,
        help="only consider this behavior group (default: all behaviors in each file)",
    )
    parser.add_argument(
        "--positive-class",
        type=int,
        default=1,
        help="predicted class value treated as behavior present (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    """Rank positive postprocessed prediction bouts and write them to CSV."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.top_n < 1:
        raise SystemExit(f"--top-n must be positive, got {args.top_n}")
    if not args.prediction_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.prediction_dir}")

    bouts = collect_bouts(
        args.prediction_dir,
        positive_class=args.positive_class,
        behavior_filter=args.behavior,
    )
    if not bouts:
        raise SystemExit("No positive postprocessed prediction bouts found.")

    behaviors = sorted({bout.behavior for bout in bouts})
    logger.info("Found %d bouts across behaviors: %s", len(bouts), ", ".join(behaviors))
    if args.behavior is None and len(behaviors) > 1:
        logger.warning(
            "Prediction files contain multiple behaviors; bouts from all of them are "
            "pooled in the output. Use --behavior to restrict to one."
        )

    # longest first; ties broken by video name then start frame for stable output
    ranked = sorted(bouts, key=lambda bout: (-bout.length, bout.video, bout.start))
    selected = ranked[: args.top_n]

    write_csv(selected, args.output)
    logger.info(
        "Wrote %d bouts (lengths %d-%d frames) to %s",
        len(selected),
        selected[-1].length,
        selected[0].length,
        args.output,
    )


if __name__ == "__main__":
    main()
