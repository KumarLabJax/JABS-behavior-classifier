"""Tests for single-pose HRNet inference loop."""

import numpy as np
import torch

from jabs.vision.hrnet_leoxiaobin import predict_single_pose


class DummyPoseModel(torch.nn.Module):
    """Small model stub that emits deterministic heatmaps."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        batch = x.shape[0]
        heatmaps = torch.zeros((batch, 2, 4, 4), device=x.device)
        heatmaps[:, 0, 1, 1] = 1.0
        heatmaps[:, 1, 2, 3] = 2.0
        return heatmaps


def test_predict_single_pose_batches_and_shapes() -> None:
    """Inference loop preserves frame count and decodes expected keypoint peaks."""
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]

    result = predict_single_pose(iter(frames), DummyPoseModel(), batch_size=2, device="cpu")

    assert result.pose.shape == (5, 2, 2)
    assert result.confidence.shape == (5, 2)
    assert result.performance.batches_processed == 3
    assert result.performance.frames_processed == 5

    expected_pose = np.array([[1, 1], [2, 3]], dtype=np.uint16)
    assert np.all(result.pose == expected_pose)
    assert np.all(result.confidence > 0.0)
