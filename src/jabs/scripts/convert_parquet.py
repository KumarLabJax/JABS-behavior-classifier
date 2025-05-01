"""
Convert Apache Parquet pose file with identity to HDF5 pose file that can be read by JABS

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
        Parquet Keypoint 6: JABS Keypoint CENTER_SPINE

    NOTE: Parquet Keypoint 6 is not inferred by the pose estimation pipeline. It is a computed centroid
    (perhaps of the bounding box?). Currently, we're mapping this to the JABS center spine keypoint.

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

import argparse
import csv
import math
import sys
from pathlib import Path

import h5py
import pandas as pd
import numpy as np

from jabs.pose_estimation import PoseEstimation

KEYPOINT_MAP = {
    1: PoseEstimation.KeypointIndex.NOSE,
    2: PoseEstimation.KeypointIndex.LEFT_EAR,
    3: PoseEstimation.KeypointIndex.RIGHT_EAR,
    4: PoseEstimation.KeypointIndex.BASE_TAIL,
    5: PoseEstimation.KeypointIndex.TIP_TAIL,
    6: PoseEstimation.KeypointIndex.CENTER_SPINE,
}


def convert(
    parquet_path: Path,
    output_path: Path,
    lixit_predictions: dict[str, tuple[float, float]] | None,
    num_frames: int,
) -> None:
    """
    Convert a parquet file to JABS Pose format.
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
    """
    Convert a pandas dataframe to JABS Pose format.
    Args:
        df: pandas dataframe with required columns
        output_path: output path for the converted h5 file
        lixit_predictions: inferred lixit points or None
        num_frames: total number of frames in the video

    Returns:
        None
    """

    identities = df["animal_id"].unique()
    num_identities = len(identities)

    # create "jabs identities" for each row
    # jabs identities are sequential integers starting at 0
    df["jabs_identity"] = df["animal_id"].apply(lambda x: identities.tolist().index(x))

    # build the jabs pose data structure
    jabs_points = np.zeros((num_frames, num_identities, 12, 2), dtype=np.uint16)
    jabs_confidences = np.zeros((num_frames, num_identities, 12), dtype=np.float32)
    jabs_id_mask = np.ones((num_frames, num_identities), dtype=np.bool_)
    jabs_embed_id = np.zeros((num_frames, num_identities), dtype=np.uint32)

    for index, row in df.iterrows():
        frame = row["frame"]
        identity = row["jabs_identity"]
        jabs_embed_id[frame, identity] = identity + 1

        jabs_id_mask[frame, identity] = False

        for keypoint in range(1,7):
            jabs_keypoint = KEYPOINT_MAP[keypoint]
            x = row[f"kpt_{keypoint}_x"]
            y = row[f"kpt_{keypoint}_y"]

            # Fill in the jabs pose data structure
            if not math.isnan(x) and not math.isnan(y):
                jabs_points[frame, identity, jabs_keypoint.value, :] = np.array(
                    [int(y), int(x)], dtype=np.uint16
                )
                jabs_confidences[frame, identity, jabs_keypoint.value] = 1.0

    with h5py.File(output_path, "w") as pose_out:
        pose_group = pose_out.create_group("poseest")
        pose_group["points"] = jabs_points
        pose_group["confidence"] = jabs_confidences
        pose_group["id_mask"] = jabs_id_mask
        pose_group["instance_embed_id"] = jabs_embed_id

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
    """
    Read the csv file with the lixit predictions and return the average x and y coordinates of the three key
    points (tip, left side, and right side).
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

    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        try:
            tip_x_index = header.index("tip.x")
            tip_y_index = header.index("tip.y")
            left_side_x_index = header.index("left_side.x")
            left_side_y_index = header.index("left_side.y")
            right_side_x_index = header.index("right_side.x")
            right_side_y_index = header.index("right_side.y")
        except ValueError:
            sys.exit(
                "CSV file does not contain the required columns: tip.x, tip.y, left_side.x, left_side.y, right_side.x, right_side.y"
            )

        for row in reader:
            tip_x.append(float(row[tip_x_index]))
            tip_y.append(float(row[tip_y_index]))
            left_side_x.append(float(row[left_side_x_index]))
            left_side_y.append(float(row[left_side_y_index]))
            right_side_x.append(float(row[right_side_x_index]))
            right_side_y.append(float(row[right_side_y_index]))

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


def main():
    parser = argparse.ArgumentParser(
        description="Convert Envision parquet file to JABS Pose format"
    )

    parser.add_argument(
        "parquet_path", type=Path, help="Path to the parquet file", nargs="+"
    )
    parser.add_argument(
        "--lixit-csv",
        type=Path,
        help="Path to the csv file containing inferred lixit points",
    )
    parser.add_argument(
        "--num-frames",
        "-f",
        type=int,
        help="Total number of frames in the video (default=1800)",
        default=1800,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory path for the converted h5 files. "
        "Default is the same directory as the input parquet file.",
    )

    args = parser.parse_args()

    try:
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
    except IOError:
        print(f"Unable to create output directory {args.out_dir}", file=sys.stderr)
        sys.exit(1)

    # read the lixit csv file if provided
    lixit_predictions = read_lixit_csv(args.lixit_csv) if args.lixit_csv else None

    # convert each file that was passed as a positional argument
    for parquet_file in args.parquet_path:
        if not parquet_file.exists():
            print(f"{parquet_file} does not exist. Skipping.", file=sys.stderr)
            continue

        if args.out_dir is None:
            output_file = parquet_file.with_suffix(".h5")
        else:
            output_file = args.out_dir / parquet_file.with_suffix(".h5").name
        convert(args.parquet_path, output_file, lixit_predictions, args.num_frames)


if __name__ == "__main__":
    main()
