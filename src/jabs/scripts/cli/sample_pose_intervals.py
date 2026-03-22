"""Sample contiguous intervals from a batch of JABS pose and video files.

For each entry in the batch file, reads the pose HDF5 to determine frame count,
selects a start frame (randomly or at a fixed offset), and writes a clipped pose
HDF5 and (optionally) a clipped AVI to the output directory. The pose file version
is inferred automatically from the filename (highest available version is used).

\b
Example:
  jabs-cli sample-pose-intervals \\
      --batch-file batch.txt \\
      --root-dir /data/videos \\
      --out-dir /data/sampled \\
      --out-frame-count 9000 \\
      --start-frame 54000
"""

from __future__ import annotations

import random
from pathlib import Path

import click
import cv2
import h5py
import numpy as np

from jabs.pose_estimation import get_pose_path


def _sample_one(
    vid_filename: str,
    root_dir: Path,
    out_dir: Path,
    out_frame_count: int,
    start_frame: int | None,
    only_pose: bool,
) -> None:
    """Sample one interval from a single video/pose pair.

    Args:
        vid_filename: Relative path to the video file from root_dir.
        root_dir: Root directory; all batch paths are relative to this.
        out_dir: Output directory for clipped files.
        out_frame_count: Number of frames to include in the clipped output.
        start_frame: 1-based start frame index, or None for random selection.
        only_pose: If True, skip video output and only write the clipped pose file.
    """
    vid_path = root_dir / vid_filename

    if not only_pose and not vid_path.is_file():
        click.echo(f"WARNING: missing video path: {vid_path}", err=True)
        return

    try:
        pose_in_path = get_pose_path(vid_path)
    except ValueError:
        click.echo(f"WARNING: no pose file found for {vid_filename}", err=True)
        return

    # Derive the output suffix from the discovered pose filename (e.g. _pose_est_v6.h5)
    pose_suffix = pose_in_path.name[len(vid_path.stem) :]

    with h5py.File(pose_in_path, "r") as pose_in:
        frame_count = pose_in["poseest"]["confidence"].shape[0]

        max_start = frame_count - out_frame_count  # inclusive max start index (0-based)
        if max_start < 0:
            click.echo(
                f"WARNING: {vid_filename} skipped: only {frame_count} frames available, "
                f"need at least {out_frame_count}",
                err=True,
            )
            return

        if start_frame is not None:
            out_start_frame_index = start_frame - 1
            if out_start_frame_index < 0 or out_start_frame_index > max_start:
                click.echo(
                    f"WARNING: {vid_filename} skipped: --start-frame {start_frame} is out of "
                    f"range [1, {max_start + 1}] for a {frame_count}-frame video with "
                    f"--out-frame-count {out_frame_count}",
                    err=True,
                )
                return
        else:
            out_start_frame_index = random.randint(0, max_start)

        vid_out_filename = vid_filename.replace("/", "+").replace("\\", "+")
        vid_out_stem = out_dir / Path(vid_out_filename).with_suffix("").name
        frame_tag = f"_{out_start_frame_index + 1}"
        vid_out_path = vid_out_stem.with_name(vid_out_stem.name + frame_tag + ".avi")
        pose_out_path = vid_out_stem.with_name(vid_out_stem.name + frame_tag + pose_suffix)

        start = out_start_frame_index
        stop = start + out_frame_count

        with h5py.File(pose_out_path, "w") as pose_out:
            pose_out["poseest/points"] = pose_in["poseest/points"][start:stop, ...]
            pose_out["poseest/confidence"] = pose_in["poseest/confidence"][start:stop, ...]

            for dataset_path in [
                "poseest/instance_count",
                "poseest/instance_embedding",
                "poseest/instance_track_id",
                "poseest/id_mask",
                "poseest/identity_embeds",
                "poseest/instance_embed_id",
                "poseest/bbox",  # v8: only present when bounding boxes were generated
                "poseest/seg_data",  # v6: segmentation masks
                "poseest/longterm_seg_id",
                "poseest/instance_seg_id",
                "poseest/seg_external_flag",
            ]:
                if dataset_path in pose_in:
                    pose_out[dataset_path] = pose_in[dataset_path][start:stop, ...]

            # instance_id_center is a per-identity array, not per-frame; copy as-is
            if "poseest/instance_id_center" in pose_in:
                pose_out["poseest/instance_id_center"] = pose_in["poseest/instance_id_center"][:]

            if "static_objects" in pose_in:
                static_group = pose_out.create_group("static_objects")
                for dataset in pose_in["static_objects"]:
                    static_group.create_dataset(dataset, data=pose_in["static_objects"][dataset])

            # v7: dynamic objects — rebase sample_indices to clip-relative frame numbers.
            #
            # Dynamic objects (e.g. fecal boli) are sparsely sampled: `points` and `counts`
            # are indexed by `sample_indices`, not by frame. To avoid false absences at clip
            # boundaries, we include one sample on each side of the clip window:
            #   - The last sample strictly before `start` is clamped to clip-relative frame 0,
            #     carrying the most recent pre-clip state into the clip.
            #   - All samples in [start, stop) are rebased by subtracting `start`.
            #   - Samples at or after `stop` are excluded.
            # If no samples exist before the clip, that boundary entry is simply omitted.
            if "dynamic_objects" in pose_in:
                dyn_out = pose_out.create_group("dynamic_objects")
                for obj_name, obj_group in pose_in["dynamic_objects"].items():
                    if not {"points", "counts", "sample_indices"}.issubset(obj_group.keys()):
                        continue
                    sample_indices = obj_group["sample_indices"][:]
                    counts = obj_group["counts"][:]
                    pts_ds = obj_group["points"]
                    pts = pts_ds[:]

                    before = np.where(sample_indices < start)[0]
                    within = np.where((sample_indices >= start) & (sample_indices < stop))[0]
                    selected = np.concatenate([before[-1:], within])

                    out_obj = dyn_out.create_group(obj_name)
                    if len(selected) == 0:
                        empty = np.array([], dtype=sample_indices.dtype)
                        out_obj.create_dataset("sample_indices", data=empty)
                        out_obj.create_dataset("counts", data=np.array([], dtype=counts.dtype))
                        out_pts = out_obj.create_dataset(
                            "points", data=np.empty((0,) + pts.shape[1:], dtype=pts.dtype)
                        )
                    else:
                        new_indices = np.clip(
                            sample_indices[selected] - start, 0, out_frame_count - 1
                        )
                        out_obj.create_dataset("sample_indices", data=new_indices)
                        out_obj.create_dataset("counts", data=counts[selected])
                        out_pts = out_obj.create_dataset("points", data=pts[selected])

                    for attr_name, attr_val in pts_ds.attrs.items():
                        out_pts.attrs[attr_name] = attr_val

            for attr in pose_in["poseest"].attrs:
                pose_out["poseest"].attrs[attr] = pose_in["poseest"].attrs[attr]

    if only_pose:
        return

    cap = None
    writer = None
    try:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            click.echo(f"WARNING: failed to open {vid_filename}", err=True)
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, out_start_frame_index)
        if not cap.isOpened():
            click.echo(f"WARNING: failed to seek to start frame {vid_filename}", err=True)
            return

        writer = cv2.VideoWriter(
            str(vid_out_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            30,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
        for _ in range(out_frame_count):
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
            else:
                click.echo(f"WARNING: {vid_filename} ended prematurely", err=True)
                break
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def sample_pose_intervals(
    batch_file: Path,
    root_dir: Path,
    out_dir: Path,
    out_frame_count: int,
    start_frame: int | None,
    only_pose: bool,
) -> None:
    """Sample intervals from every video listed in a batch file.

    Args:
        batch_file: Path to a newline-separated list of video filenames.
        root_dir: Root directory; all batch paths are relative to this.
        out_dir: Output directory for clipped files.
        out_frame_count: Number of frames to include in each clipped output.
        start_frame: 1-based start frame index, or None for random selection.
        only_pose: If True, skip video output and only write the clipped pose files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with batch_file.open() as f:
        for line in f:
            vid_filename = line.strip()
            if not vid_filename:
                continue
            click.echo(f"Processing: {vid_filename}")
            _sample_one(
                vid_filename=vid_filename,
                root_dir=root_dir,
                out_dir=out_dir,
                out_frame_count=out_frame_count,
                start_frame=start_frame,
                only_pose=only_pose,
            )


@click.command(
    name="sample-pose-intervals",
    context_settings={"max_content_width": 120},
    help=__doc__,
)
@click.option(
    "--batch-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Newline-separated list of video filenames to process.",
)
@click.option(
    "--root-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Root directory; all paths in the batch file are relative to this.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for clipped pose and video files.",
)
@click.option(
    "--out-frame-count",
    type=int,
    required=True,
    help="Number of frames to save per interval. At 30 fps, 1800 ≈ one minute.",
)
@click.option(
    "--start-frame",
    type=int,
    default=None,
    help="1-based start frame index. If omitted, a random start frame is chosen.",
)
@click.option(
    "--pose-version",
    type=click.Choice(["2", "3", "4", "5", "6", "7", "8"]),
    default=None,
    hidden=True,
    is_eager=True,
    expose_value=False,
    callback=lambda ctx, param, value: (
        click.echo(
            "WARNING: --pose-version is deprecated and has no effect; "
            "the pose version is now inferred automatically from the filename.",
            err=True,
        )
        if value is not None
        else None
    ),
    help="Deprecated. Pose version is now inferred automatically.",
)
@click.option(
    "--only-pose",
    is_flag=True,
    help="Write only the clipped pose file; skip video output.",
)
def sample_pose_intervals_command(
    batch_file: Path,
    root_dir: Path,
    out_dir: Path,
    out_frame_count: int,
    start_frame: int | None,
    only_pose: bool,
) -> None:
    """Sample pose (and optionally video) intervals from a batch of JABS files."""
    sample_pose_intervals(
        batch_file=batch_file,
        root_dir=root_dir,
        out_dir=out_dir,
        out_frame_count=out_frame_count,
        start_frame=start_frame,
        only_pose=only_pose,
    )
