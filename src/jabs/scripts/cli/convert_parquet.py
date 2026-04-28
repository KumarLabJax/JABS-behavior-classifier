"""Convert parquet pose files to JABS HDF5 pose format."""

import math
import textwrap
from pathlib import Path

import click
import h5py
import numpy as np
import numpy.typing as npt
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
    lixit_predictions: npt.NDArray[np.float32] | None,
    num_frames: int,
) -> None:
    """Convert a parquet file to JABS Pose format.

    Args:
        parquet_path: path to input parquet file
        output_path: output path for the converted h5 file
        lixit_predictions: lixit keypoints array of shape (num_lixits, 3, 2) in (y, x)
            order, or None
        num_frames: total number of frames in the video
    """
    df = pd.read_parquet(parquet_path)
    convert_data_frame(df, output_path, lixit_predictions, num_frames)


def convert_data_frame(
    df: pd.DataFrame,
    output_path: Path,
    lixit_predictions: npt.NDArray[np.float32] | None,
    num_frames: int,
) -> None:
    """Convert a pandas dataframe to JABS Pose format.

    Args:
        df: pandas dataframe with required columns
        output_path: output path for the converted h5 file
        lixit_predictions: lixit keypoints array of shape (num_lixits, 3, 2) in (y, x)
            order, or None
        num_frames: total number of frames in the video
    """
    # Build identities from the string eartag_code field (external IDs).
    # Drop missing/empty values as well as tag no-reads and sort for stable index mapping (0..N-1).
    identities = [
        s
        for s in (df["eartag_code"].dropna().astype(str).str.strip().unique().tolist())
        if s not in ["", "00", "01"]
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
        # doesn't map to a jabs keypoint. It's similar to our computed centroids
        for keypoint in range(1, 6):
            jabs_keypoint = KEYPOINT_MAP[keypoint]
            x = row[f"kpt_{keypoint}_x"]
            y = row[f"kpt_{keypoint}_y"]

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

        # The parquet file provides external IDs via the string field `eartag_code`, while JABS
        # uses 0..(num_identities-1). Save the original values so we can map back downstream.
        string_dt = h5py.string_dtype(encoding="utf-8")
        pose_group.create_dataset(
            "external_identity_mapping",
            data=np.array(identities, dtype=object),
            dtype=string_dt,
        )

        static_objects_group = pose_out.create_group("static_objects")
        if lixit_predictions is not None:
            static_objects_group["lixit"] = lixit_predictions

        pose_group.attrs["version"] = np.array([5, 0])


def read_lixit_parquet(path: Path) -> npt.NDArray[np.float32]:
    """Read lixit keypoints from a summary parquet file.

    The parquet file must have a single row with a 'keypoints' column containing
    an array of (x, y) coordinate pairs. Each consecutive group of 3 keypoints
    represents one lixit in tip, left_side, right_side order. For example, a file
    with 6 keypoints describes 2 lixits.

    Args:
        path: path to the summary parquet file.

    Returns:
        Array of shape (num_lixits, 3, 2) with (y, x) coordinates in float32.

    Raises:
        ValueError: if the keypoint count is not a multiple of 3.
    """
    df = pd.read_parquet(path)
    # keypoints is stored as an array-like of (x, y) pairs
    keypoints = np.array(list(df["keypoints"].iloc[0]), dtype=np.float32)  # (n_kp, 2) in x,y

    n_kp = len(keypoints)
    if n_kp % 3 != 0:
        raise ValueError(
            f"Lixit parquet keypoint count ({n_kp}) must be a multiple of 3 "
            "(tip, left_side, right_side per lixit)."
        )

    n_lixits = n_kp // 3
    grouped = keypoints.reshape(n_lixits, 3, 2)  # (n_lixits, 3, [x, y])
    # JABS expects (y, x) order - swap the last axis
    return grouped[:, :, ::-1].copy()


@click.command(
    name="convert-parquet",
    help="\b\n"
    + textwrap.dedent("""\
        Convert a parquet pose file to JABS HDF5 pose format.

        \b
        Expects the input parquet file to have the following columns:
          - frame: frame number
          - animal_id: identifier for each animal, unique per frame
          - eartag_code: external identity string (used for external_identity_mapping)
          - kpt_1_x/y: keypoint 1 (nose)
          - kpt_2_x/y: keypoint 2 (left ear)
          - kpt_3_x/y: keypoint 3 (right ear)
          - kpt_4_x/y: keypoint 4 (base tail)
          - kpt_5_x/y: keypoint 5 (tip tail)

        \b
        Lixit keypoints (--lixit-parquet):
          Summary parquet with a single row and a 'keypoints' column containing
          (x, y) pairs in groups of 3 (tip, left_side, right_side per lixit).
          6 keypoints = 2 lixits; all lixits are written to the output file.
    """),
)
@click.argument("parquet_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--lixit-parquet",
    "lixit_parquet",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to summary parquet file with lixit keypoints. "
    "Supports multiple lixits (6 keypoints = 2 lixits).",
)
@click.option(
    "--num-frames",
    "num_frames",
    type=int,
    default=1800,
    show_default=True,
    help="Total number of frames in the video.",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="Output directory for the converted h5 file. Defaults to the same directory as the input.",
)
def convert_parquet_command(
    parquet_path: Path,
    lixit_parquet: Path | None,
    num_frames: int,
    out_dir: Path | None,
) -> None:
    """Convert a parquet pose file to JABS HDF5 pose format."""
    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise click.ClickException(f"Unable to create output directory {out_dir}: {e}") from e

    lixit_predictions: npt.NDArray[np.float32] | None = None
    if lixit_parquet is not None:
        try:
            lixit_predictions = read_lixit_parquet(lixit_parquet)
        except Exception as e:
            raise click.ClickException(str(e)) from e

    if out_dir is None:
        output_file = parquet_path.with_name(
            parquet_path.name.replace(".parquet", "_pose_est_v8.h5")
        )
    else:
        output_file = out_dir / parquet_path.name.replace(".parquet", "_pose_est_v8.h5")

    try:
        convert(parquet_path, output_file, lixit_predictions, num_frames)
    except Exception as e:
        raise click.ClickException(f"Failed to convert {parquet_path.name}: {e}") from e

    click.echo(f"Wrote {output_file}")
