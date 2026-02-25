"""Single-pose inference loop for HRNet (Leoxiaobin) models."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .decode import argmax_2d_torch
from .preprocess import preprocess_hrnet

DEFAULT_HEATMAP_KEY = "heatmaps"


@dataclass
class PerformanceAccumulator:
    """Collects per-batch timings for single-pose inference."""

    frame_per_batch: int = 1
    preprocess_times: list[float] = field(default_factory=list)
    model_times: list[float] = field(default_factory=list)
    postprocess_times: list[float] = field(default_factory=list)
    _frames_processed: int = 0

    def add_batch_times(
        self,
        t1: float,
        t2: float,
        t3: float,
        t4: float,
        *,
        frames_in_batch: int | None = None,
    ) -> None:
        """Add timestamps for one processed batch."""
        self.preprocess_times.append(t2 - t1)
        self.model_times.append(t3 - t2)
        self.postprocess_times.append(t4 - t3)
        self._frames_processed += (
            self.frame_per_batch if frames_in_batch is None else frames_in_batch
        )

    @property
    def batches_processed(self) -> int:
        """Number of processed batches."""
        return len(self.preprocess_times)

    @property
    def frames_processed(self) -> int:
        """Total number of processed frames."""
        return self._frames_processed

    def summary(self) -> dict[str, float]:
        """Return average stage timings."""
        if self.batches_processed == 0:
            return {
                "preprocess_seconds": 0.0,
                "model_seconds": 0.0,
                "postprocess_seconds": 0.0,
                "total_seconds": 0.0,
            }

        preprocess = float(np.mean(self.preprocess_times))
        model = float(np.mean(self.model_times))
        postprocess = float(np.mean(self.postprocess_times))
        total = preprocess + model + postprocess
        return {
            "preprocess_seconds": preprocess,
            "model_seconds": model,
            "postprocess_seconds": postprocess,
            "total_seconds": total,
        }


@dataclass
class SinglePoseInferenceResult:
    """Single-pose inference outputs."""

    pose: np.ndarray
    confidence: np.ndarray
    performance: PerformanceAccumulator


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _extract_heatmaps(model_output: Any) -> Tensor:
    if isinstance(model_output, Tensor):
        return model_output
    if isinstance(model_output, dict):
        if DEFAULT_HEATMAP_KEY in model_output:
            return model_output[DEFAULT_HEATMAP_KEY]
        if len(model_output) == 1:
            return next(iter(model_output.values()))
        raise ValueError(
            f"Model output dict is missing '{DEFAULT_HEATMAP_KEY}'. "
            f"Available keys: {list(model_output.keys())}"
        )
    raise TypeError(f"Unsupported model output type: {type(model_output)!r}")


def predict_single_pose(
    input_iter: Iterator[np.ndarray],
    model: torch.nn.Module,
    *,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    render: str | Path | None = None,
    render_pose_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> SinglePoseInferenceResult:
    """Run single-pose inference for a frame iterator.

    Args:
        input_iter: Iterator yielding input frames (H, W, 3).
        model: Loaded torch model.
        batch_size: Number of frames per batch.
        device: Device override. Defaults to CUDA if available.
        render: Optional path to rendered output video.
        render_pose_fn: Optional rendering callback used when `render` is provided.

    Returns:
        SinglePoseInferenceResult with pose coordinates and confidences.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, received {batch_size}")
    if render is not None and render_pose_fn is None:
        raise ValueError("render_pose_fn must be provided when render output is enabled.")

    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    pose_batches: list[np.ndarray] = []
    confidence_batches: list[np.ndarray] = []
    performance = PerformanceAccumulator(frame_per_batch=batch_size)

    writer = None
    if render is not None:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError(
                "Optional dependency 'imageio' is required for rendering video output. "
                "Install it with:\n"
                "  pip install imageio imageio-ffmpeg"
            ) from exc

        writer = imageio.get_writer(render, fps=30)

    try:
        while True:
            batch_frames: list[np.ndarray] = []
            for _ in range(batch_size):
                try:
                    batch_frames.append(next(input_iter))
                except StopIteration:
                    break

            if not batch_frames:
                break

            t1 = time.perf_counter()
            batch_tensor = torch.cat(
                [preprocess_hrnet(frame) for frame in batch_frames], dim=0
            ).to(resolved_device)
            t2 = time.perf_counter()

            with torch.no_grad():
                model_output = model(batch_tensor)
            heatmaps = _extract_heatmaps(model_output)
            t3 = time.perf_counter()

            confidence_cuda, pose_cuda = argmax_2d_torch(heatmaps)
            confidence = confidence_cuda.detach().cpu().numpy().astype(np.float32, copy=False)
            pose = pose_cuda.detach().cpu().numpy().astype(np.uint16, copy=False)

            pose_batches.append(pose)
            confidence_batches.append(confidence)

            if writer is not None and render_pose_fn is not None:
                for frame_idx, frame in enumerate(batch_frames):
                    writer.append_data(render_pose_fn(frame.astype(np.uint8), pose[frame_idx]))

            t4 = time.perf_counter()
            performance.add_batch_times(t1, t2, t3, t4, frames_in_batch=len(batch_frames))
    finally:
        if writer is not None:
            writer.close()

    if pose_batches:
        pose_out = np.concatenate(pose_batches, axis=0)
        confidence_out = np.concatenate(confidence_batches, axis=0)
    else:
        pose_out = np.empty((0, 0, 2), dtype=np.uint16)
        confidence_out = np.empty((0, 0), dtype=np.float32)

    return SinglePoseInferenceResult(
        pose=pose_out,
        confidence=confidence_out,
        performance=performance,
    )
