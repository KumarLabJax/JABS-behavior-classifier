"""``jabs-pose`` CLI: video -> single-mouse pose_est_v2.h5."""

import logging
from itertools import pairwise
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt

from jabs.core.abstract.pose_est import MINIMUM_CONFIDENCE, PoseEstimation
from jabs.core.enums import JabsPoseVersion
from jabs.core.types import PoseData
from jabs.io import save
from jabs.vision.cli._logging import configure_logging
from jabs.vision.hrnet_msfork import load_pose_model, predict_single_pose
from jabs.vision.io import read_frames, video_fps

logger = logging.getLogger(__name__)

_BODY_PARTS = [kp.name.lower() for kp in PoseEstimation.KeypointIndex]


def _skeleton_edges() -> list[tuple[int, int]]:
    """Build keypoint edge pairs from the canonical connected segments."""
    edges: list[tuple[int, int]] = []
    for segment in PoseEstimation.FULL_CONNECTED_SEGMENTS:
        for a, b in pairwise(segment):
            edges.append((int(a), int(b)))
    return edges


_SKELETON_EDGES = _skeleton_edges()


def _build_pose_data(
    pose_yx: npt.NDArray[np.uint16],
    confidence: npt.NDArray[np.float32],
    *,
    fps: int,
    config: Path,
    checkpoint: Path,
) -> PoseData:
    """Assemble a canonical PoseData from inference output.

    Args:
        pose_yx: Inference coordinates, shape (n_frames, 12, 2), order (y, x).
        confidence: Per-keypoint confidence, shape (n_frames, 12).
        fps: Video frame rate.
        config: HRNet config path (recorded for provenance).
        checkpoint: Checkpoint path (recorded for provenance).

    Returns:
        A single-identity PoseData with points in (x, y) order. ``point_mask`` is
        derived from the confidence threshold so it matches how the legacy v2 reader
        interprets confidence on read-back.
    """
    points_xy = np.flip(pose_yx.astype(np.float64), axis=-1)[np.newaxis, ...]
    conf_2d = confidence.astype(np.float32)
    conf = conf_2d[np.newaxis, ...]
    point_mask = (conf_2d > MINIMUM_CONFIDENCE)[np.newaxis, ...]
    n_frames = points_xy.shape[1]
    return PoseData(
        points=points_xy,
        point_mask=point_mask,
        identity_mask=np.ones((1, n_frames), dtype=bool),
        body_parts=_BODY_PARTS,
        edges=_SKELETON_EDGES,
        fps=fps,
        confidence=conf,
        metadata={"config": Path(config).name, "model": Path(checkpoint).name},
    )


def run_pose_inference(
    *,
    video: Path,
    out: Path,
    config: Path,
    checkpoint: Path,
    batch_size: int = 1,
    device: str | None = None,
) -> None:
    """Run pose inference on a video and write a pose_est_v2.h5.

    Args:
        video: Input video path.
        out: Output pose HDF5 path.
        config: HRNet YAML config path.
        checkpoint: HRNet checkpoint (.pth) path.
        batch_size: Frames per inference batch.
        device: Torch device ("cuda"/"cpu"), or None to auto-select.

    Raises:
        ValueError: If the video yields no frames.
        RuntimeError: If the checkpoint does not match the model (strict load).
    """
    logger.info("loading model config=%s checkpoint=%s", config, checkpoint)
    model, _ = load_pose_model(config, checkpoint, device=device, strict=True)

    result = predict_single_pose(read_frames(video), model, batch_size=batch_size, device=device)
    if result.pose.shape[0] == 0:
        raise ValueError(f"inference produced no frames for video {video}")

    pose_data = _build_pose_data(
        result.pose,
        result.confidence,
        fps=video_fps(video),
        config=config,
        checkpoint=checkpoint,
    )
    save(pose_data, out, legacy=JabsPoseVersion.V2)
    logger.info("wrote %d frames to %s", result.pose.shape[0], out)


@click.command(name="pose", context_settings={"max_content_width": 120})
@click.option(
    "--video", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--config", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--batch-size", default=1, show_default=True, type=int)
@click.option(
    "--device", default="auto", show_default=True, type=click.Choice(["auto", "cuda", "cpu"])
)
def pose_command(
    video: Path, out: Path, config: Path, checkpoint: Path, batch_size: int, device: str
) -> None:
    """Run single-mouse HRNet pose inference and write a pose_est_v2.h5."""
    configure_logging()
    resolved_device = None if device == "auto" else device
    run_pose_inference(
        video=video,
        out=out,
        config=config,
        checkpoint=checkpoint,
        batch_size=batch_size,
        device=resolved_device,
    )


def main() -> None:
    """Entry point for the ``jabs-pose`` console script."""
    pose_command()
