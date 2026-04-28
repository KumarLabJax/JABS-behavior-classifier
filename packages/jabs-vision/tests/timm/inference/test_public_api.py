"""Smoke tests for the public timm inference API."""

from __future__ import annotations

from jabs.vision.timm import inference
from jabs.vision.timm.inference import confidence, decoding


def test_public_exports() -> None:
    """Public package re-exports the intended inference helpers."""
    assert inference.compute_heatmap_confidence is confidence.compute_heatmap_confidence
    assert inference.sample_confidence_at_coords is confidence.sample_confidence_at_coords
    assert inference.compute_confidence_labels is confidence.compute_confidence_labels
    assert inference.get_max_preds is decoding.get_max_preds
    assert inference.decode_heatmaps is decoding.decode_heatmaps
