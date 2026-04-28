"""Tests for timm heatmap decoding helpers."""

from __future__ import annotations

import pytest
import torch

from jabs.vision.timm.inference import decode_heatmaps, get_max_preds


def test_get_max_preds_handles_non_contiguous_heatmaps() -> None:
    """Argmax decoding should work for non-contiguous tensors."""
    base = torch.zeros((1, 5, 6, 2), dtype=torch.float32)
    base[0, 2, 4, 1] = 3.0
    heatmaps = base.permute(0, 3, 1, 2)

    coords, maxvals = get_max_preds(heatmaps)

    assert not heatmaps.is_contiguous()
    assert coords.dtype == heatmaps.dtype
    assert torch.equal(coords[0, 1], torch.tensor([4.0, 2.0]))
    assert torch.equal(maxvals[0, 1], torch.tensor([3.0]))


def test_get_max_preds_validates_rank() -> None:
    """Bad heatmap rank should fail with a clear error."""
    with pytest.raises(ValueError, match="4 dimensions"):
        get_max_preds(torch.zeros((2, 3, 4), dtype=torch.float32))


def test_decode_heatmaps_dark_falls_back_on_small_heatmaps() -> None:
    """DARK should return plain argmax coords when a full neighborhood is unavailable."""
    heatmaps = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    heatmaps[0, 0, 1, 2] = 5.0

    plain = decode_heatmaps(heatmaps, use_dark=False)
    refined = decode_heatmaps(heatmaps, use_dark=True)

    assert torch.equal(refined, plain)


def test_decode_heatmaps_dark_applies_subpixel_refinement() -> None:
    """DARK refinement should move coordinates off the integer argmax when the gradient supports it."""
    heatmaps = torch.zeros((1, 1, 7, 7), dtype=torch.float32)
    patch = torch.tensor(
        [
            [0.4, 0.6, 0.7],
            [0.5, 1.0, 0.9],
            [0.3, 0.5, 0.6],
        ],
        dtype=torch.float32,
    )
    heatmaps[0, 0, 2:5, 2:5] = patch

    refined = decode_heatmaps(heatmaps, use_dark=True)

    assert refined[0, 0, 0] > 3.0
    assert refined[0, 0, 1] < 3.0
