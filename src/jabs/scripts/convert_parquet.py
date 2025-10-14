"""Convert Apache Parquet pose file with identity to HDF5 pose file that can be read by JABS

Example:
    $ python convert_parquet.py --lixit-csv lixit.csv input1.parquet

    multiple parquet files can be provided as positional arguments

    $ python convert_parquet.py --lixit-csv lixit.csv input1.parquet input2.parquet input3.parquet

    This script assumes that the parquet files are for videos with 1800 frames (some frames might be ommitted
    from the parquet file if no animals were detected in that frame). If the video lenght is different, then
    the --num-frames argument should be provided to ensure the output h5 file has the correct number of frames.

Input parquet file:

    The input parquet file is expected to have the following columns:
    - animal_id: identifier for each animal, unique per frame
    - frame: frame number
    - eartag_code: external identity string (used for external_identity_mapping)
    - kpt_1_x: x coordinate of keypoint 1
    - kpt_1_y: y coordinate of keypoint 1
    - kpt_2_x: x coordinate of keypoint 2
    - kpt_2_y: y coordinate of keypoint 2
    - kpt_3_x: x coordinate of keypoint 3
    - kpt_3_y: y coordinate of keypoint 3
    - kpt_4_x: x coordinate of keypoint 4
    - kpt_4_y: y coordinate of keypoint 4
    - kpt_5_x: x coordinate of keypoint 5
    - kpt_5_y: y coordinate of keypoint 5
    - kpt_6_x: x coordinate of keypoint 6
    - kpt_6_y: y coordinate of keypoint 6

Output h5 file:

    By default, outputs a JABS Pose version 5 file in the same directory as the input parquet file.
    The output file will have the same name as the input file, but with the .h5 extension.

    NOTE: The JABS Pose file will be missing some attributes and data sets that are typically found in
    a pose file produced by our multi-mouse pose estimation pipeline. This script only populates the
    datasets and attributes that are read by the JABS behavior classifier.

    Parquet Key point to JABS Key point mapping:
        Parquet Keypoint 1: JABS Keypoint NOSE
        Parquet Keypoint 2: JABS Keypoint LEFT_EAR
        Parquet Keypoint 3: JABS Keypoint RIGHT_EAR
        Parquet Keypoint 4: JABS Keypoint BASE_TAIL
        Parquet Keypoint 5: JABS Keypoint TIP_TAIL
        Parquet Keypoint 6: ignored

    NOTE: Parquet Keypoint 6 is not inferred by the pose estimation pipeline. It is near, but not exactly
    the centroid. JABS computes a centroid, so we ignore this keypoint.

Lixit keypoints:

    Lixit keypoints can be provided in a CSV file. The CSV file should have the following columns (header with these
    column names is required):
            tip.x
            tip.y
            left_side.x
            left_side.y
            right_side.x
            right_side.y

    the Lixit keypoints are stored in the static_objects group of the h5 file. The lixit keypoints are stored as
    three keypoints: tip, left_side, and right_side. In order to support multiple lixits, the lixit keypoints are stored
    as an array of shape (num lixit, 3, 2) where the first dimension is the number of lixits and the second
    dimension is the number of keypoints (tip, left_side, right_side) and the third dimension is the y and x coordinates

    Note: jabs expects lixit key points are y,x order

Todo:
    - add support for food hopper key points

"""

import csv
import math
import sys
import textwrap
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd

from jabs.pose_estimation import PoseEstimation

KEYPOINT_MAP = {
    1: PoseEstimation.KeypointIndex.NOSE,
    2: PoseEstimation.KeypointIndex.LEFT_EAR,
    3: PoseEstimation.KeypointIndex.RIGHT_EAR,
    4: PoseEstimation.KeypointIndex.BASE_TAIL,
    5: PoseEstimation.KeypointIndex.TIP_TAIL,
}


def convert(
    parquet_path: Path,
    output_path: Path,
    lixit_predictions: dict[str, tuple[float, float]] | None,
    num_frames: int,
) -> None:
    """Convert a parquet file to JABS Pose format.

    Args:
        parquet_path: path to input parquet file
        output_path: output path for the converted h5 file
        lixit_predictions: inferred lixit points or None
        num_frames: total number of frames in the video

    Returns:
        None
    """
    # Read the parquet file into dataframe
    df = pd.read_parquet(parquet_path)

    convert_data_frame(df, output_path, lixit_predictions, num_frames)


def convert_data_frame(
    df: pd.DataFrame,
    output_path: Path,
    lixit_predictions: dict[str, tuple[float, float]] | None,
    num_frames: int,
) -> None:
    """Convert a pandas dataframe to JABS Pose format.

    Args:
        df: pandas dataframe with required columns
        output_path: output path for the converted h5 file
        lixit_predictions: inferred lixit points or None
        num_frames: total number of frames in the video

    Returns:
        None
    """
    # Build identities from the string eartag_code field (external IDs).
    # Drop missing/empty values as well as tag no-reads and sort for stable index mapping (0..N-1).
    identities = [
        s
        for s in (df["eartag_code"].dropna().astype(str).str.strip().unique().tolist())
        if s not in ["", "00"]
    ]
    identities.sort()
    num_identities = len(identities)
    df = df[df["eartag_code"].isin(identities)].copy()

    # Create "jabs identities" for each row (sequential integers starting at 0)
    df["jabs_identity"] = (
        df["eartag_code"].astype(str).str.strip().apply(lambda x: identities.index(x))
    )

    # build the jabs pose data structure
    jabs_points = np.zeros((num_frames, num_identities, 12, 2), dtype=np.uint16)
    jabs_confidences = np.zeros((num_frames, num_identities, 12), dtype=np.float32)
    jabs_id_mask = np.ones((num_frames, num_identities), dtype=np.bool_)
    jabs_embed_id = np.zeros((num_frames, num_identities), dtype=np.uint32)
    jabs_bboxes = np.full((num_frames, num_identities, 2, 2), np.nan, dtype=np.float32)

    for _, row in df.iterrows():
        frame = row["frame"]
        identity = row["jabs_identity"]

        # jabs "instance_embed_id" uses 1-based indexing (zero is used to fill missing data)
        jabs_embed_id[frame, identity] = identity + 1

        jabs_id_mask[frame, identity] = False

        # we only iterate over keypoints 1-5, since keypoint 6 is computed and
        # doesn't map to a jabs keypoint. It's similar to our computed
        # centroids
        for keypoint in range(1, 6):
            jabs_keypoint = KEYPOINT_MAP[keypoint]
            x = row[f"kpt_{keypoint}_x"]
            y = row[f"kpt_{keypoint}_y"]

            # Fill in the jabs pose data structure
            if not math.isnan(x) and not math.isnan(y):
                jabs_points[frame, identity, jabs_keypoint.value, :] = np.array(
                    [int(y), int(x)], dtype=np.uint16
                )
                jabs_confidences[frame, identity, jabs_keypoint.value] = 1.0

        jabs_bboxes[frame, identity, 0] = [row["bb_left"], row["bb_top"]]
        jabs_bboxes[frame, identity, 1] = [row["bb_right"], row["bb_bottom"]]

    # Replace NaN with -1 as placeholder for missing bounding box coordinates
    jabs_bboxes = np.where(np.isnan(jabs_bboxes), -1, jabs_bboxes)

    with h5py.File(output_path, "w") as pose_out:
        pose_group = pose_out.create_group("poseest")
        pose_group.create_dataset("points", data=jabs_points, dtype=np.uint16)
        pose_group.create_dataset("confidence", data=jabs_confidences, dtype=np.float32)
        pose_group.create_dataset("id_mask", data=jabs_id_mask, dtype=np.bool_)
        pose_group.create_dataset("instance_embed_id", data=jabs_embed_id, dtype=np.uint32)
        pose_group.create_dataset(
            "instance_id_center", data=np.zeros((num_identities, 1)), dtype=np.float64
        )
        bbox_dataset = pose_group.create_dataset("bbox", data=jabs_bboxes, dtype=np.float32)
        bbox_dataset.attrs["bboxes_generated"] = True

        # The parquet file provides external IDs via the string field `eartag_code`, while JABS uses 0..(num_identities-1).
        # Save the original `eartag_code` values in the pose file so we can map back to the original IDs downstream.
        string_dt = h5py.string_dtype(encoding="utf-8")
        pose_group.create_dataset(
            "external_identity_mapping",
            data=np.array(identities, dtype=object),
            dtype=string_dt,
        )

        static_objects_group = pose_out.create_group("static_objects")
        if lixit_predictions is not None:
            static_objects_group["lixit"] = np.array(
                [
                    [
                        lixit_predictions["tip"],
                        lixit_predictions["left_side"],
                        lixit_predictions["right_side"],
                    ]
                ],
                dtype=np.float32,
            )

        pose_group.attrs["version"] = np.array([5, 0])


def read_lixit_csv(path: Path) -> dict[str, tuple[float, float]]:
    """Read the csv file with the lixit predictions

    Averages the x and y coordinates of the three key points (tip, left side, and right side) for
    all of the lixit predictions in the file.

    Args:
        path (Path): Path to the csv file

    Returns:
        dict: A dictionary with the average x and y coordinates of the three key points
    """
    tip_x = []
    tip_y = []
    left_side_x = []
    left_side_y = []
    right_side_x = []
    right_side_y = []

    with open(path) as f:
        reader = csv.DictReader(f)

        try:
            for row in reader:
                if row["tip.x"] and row["tip.y"]:
                    tip_x.append(float(row["tip.x"]))
                    tip_y.append(float(row["tip.y"]))
                if row["left_side.x"] and row["left_side.y"]:
                    left_side_x.append(float(row["left_side.x"]))
                    left_side_y.append(float(row["left_side.y"]))
                if row["right_side.x"] and row["right_side.y"]:
                    right_side_x.append(float(row["right_side.x"]))
                    right_side_y.append(float(row["right_side.y"]))
        except KeyError:
            sys.exit(
                "CSV file does not contain the required columns: tip.x, tip.y, left_side.x, left_side.y, right_side.x, right_side.y"
            )
    return {
        "tip": (
            np.array(tip_y).mean(dtype=np.float32),
            np.array(tip_x).mean(dtype=np.float32),
        ),
        "left_side": (
            np.array(left_side_y).mean(dtype=np.float32),
            np.array(left_side_x).mean(dtype=np.float32),
        ),
        "right_side": (
            np.array(right_side_y).mean(dtype=np.float32),
            np.array(right_side_x).mean(dtype=np.float32),
        ),
    }


@click.command(
    help="\b\n"
    + textwrap.dedent("""\
        Convert parquet pose file to JABS Pose format.
        
        \b
        Expects the input parquet file to have the following columns:
          - frame: frame number
          - animal_id: identifier for each animal, unique per frame
          - eartag_code: external identity string (used for external_identity_mapping)
          - kpt_1_x: x coordinate of keypoint 1 (nose)
          - kpt_1_y: y coordinate of keypoint 1
          - kpt_2_x: x coordinate of keypoint 2 (left ear)
          - kpt_2_y: y coordinate of keypoint 2
          - kpt_3_x: x coordinate of keypoint 3 (right ear)
          - kpt_3_y: y coordinate of keypoint 3
          - kpt_4_x: x coordinate of keypoint 4 (base tail)
          - kpt_4_y: y coordinate of keypoint 4
          - kpt_5_x: x coordinate of keypoint 5 (tip tail)
          - kpt_5_y: y coordinate of keypoint 5
    """)
)
@click.argument("parquet_path", nargs=-1, type=click.Path(path_type=Path))
@click.option(
    "--lixit-csv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the csv file containing inferred lixit points",
)
@click.option(
    "--num-frames",
    "num_frames",
    type=int,
    default=1800,
    show_default=True,
    help="Total number of frames in the video",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="Output directory path for the converted h5 files. Default is the same directory as the input parquet file.",
)
def cli(
    parquet_path: tuple[Path, ...], lixit_csv: Path | None, num_frames: int, out_dir: Path | None
) -> None:
    """Convert parquet pose file to JABS Pose format."""
    try:
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        print(f"Unable to create output directory {out_dir}", file=sys.stderr)
        sys.exit(1)

    lixit_predictions = read_lixit_csv(lixit_csv) if lixit_csv else None

    for parquet_file in parquet_path:
        if not parquet_file.exists():
            print(f"{parquet_file} does not exist. Skipping.", file=sys.stderr)
            continue

        if out_dir is None:
            output_file = parquet_file.with_name(
                parquet_file.name.replace(".parquet", "_pose_est_v8.h5")
            )
        else:
            output_file = out_dir / Path(parquet_file.name.replace(".parquet", "_pose_est_v8.h5"))
        convert(parquet_file, output_file, lixit_predictions, num_frames)


def main():
    """Entry point for the JABS CLI."""
    cli(obj={})


if __name__ == "__main__":
    cli()
