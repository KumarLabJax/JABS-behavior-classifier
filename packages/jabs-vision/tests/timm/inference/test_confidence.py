"""Tests for timm confidence helpers."""

from __future__ import annotations

import pytest
import torch

from jabs.vision.timm.inference import (
    compute_confidence_labels,
    compute_heatmap_confidence,
    sample_confidence_at_coords,
)


def test_compute_heatmap_confidence_uses_sigmoid_peak() -> None:
    """Heatmap confidence should be the max sigmoid score per keypoint."""
    heatmaps = torch.tensor(
        [[[[0.0, 1.0], [-2.0, 0.5]], [[-1.0, 0.0], [2.0, -3.0]]]],
        dtype=torch.float32,
    )

    conf = compute_heatmap_confidence(heatmaps)

    expected = torch.sigmoid(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert torch.allclose(conf, expected)


def test_sample_confidence_at_coords_samples_each_keypoint() -> None:
    """Sampling should return the value at integral coordinates for every keypoint."""
    confidence_maps = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    confidence_maps[0, 0, 1, 2] = 0.7
    confidence_maps[0, 1, 2, 1] = 0.4
    coords = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]], dtype=torch.float32)

    sampled = sample_confidence_at_coords(confidence_maps, coords)

    assert torch.allclose(sampled, torch.tensor([[0.7, 0.4]], dtype=torch.float32))


def test_sample_confidence_at_coords_validates_shape() -> None:
    """Bad coordinate shapes should fail early."""
    with pytest.raises(ValueError, match="shape \\(B, K, 2\\)"):
        sample_confidence_at_coords(
            torch.zeros((1, 2, 3, 4), dtype=torch.float32),
            torch.zeros((1, 2), dtype=torch.float32),
        )


def test_compute_confidence_labels_accepts_tensor_diagonal() -> None:
    """Tensor diagonals with a different dtype should be normalized before comparison."""
    predictions = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]], dtype=torch.float32)
    ground_truth = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    image_diagonal = torch.tensor(100.0, dtype=torch.float64)

    labels = compute_confidence_labels(
        predictions,
        ground_truth,
        threshold_fraction=0.05,
        image_diagonal=image_diagonal,
    )

    assert torch.equal(labels, torch.tensor([[1.0, 1.0]], dtype=torch.float32))
