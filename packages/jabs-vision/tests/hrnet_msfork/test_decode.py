"""Tests for HRNet heatmap decode helpers."""

import torch

from jabs.vision.hrnet_msfork import argmax_2d_torch


def test_argmax_2d_torch_returns_expected_coordinates() -> None:
    """Argmax decode returns expected peak coordinates per heatmap."""
    heatmaps = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
    heatmaps[0, 0, 1, 2] = 0.7
    heatmaps[0, 1, 3, 4] = 0.9
    heatmaps[0, 2, 0, 0] = 0.2

    heatmaps[1, 0, 2, 1] = 0.3
    heatmaps[1, 1, 1, 3] = 0.4
    heatmaps[1, 2, 3, 2] = 0.8

    values, coords = argmax_2d_torch(heatmaps)

    assert values.shape == (2, 3)
    assert coords.shape == (2, 3, 2)

    assert torch.equal(coords[0, 0], torch.tensor([1, 2]))
    assert torch.equal(coords[0, 1], torch.tensor([3, 4]))
    assert torch.equal(coords[1, 2], torch.tensor([3, 2]))
