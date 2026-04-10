"""Confidence extraction and sampling utilities."""

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_heatmap_confidence(heatmaps: Tensor) -> Tensor:
    """Derive confidence from heatmap peak values.

    Simple approach: sigmoid of the max value per keypoint.

    Args:
        heatmaps: (B, K, H, W) heatmap tensor.

    Returns:
        (B, K) confidence scores in [0, 1].
    """
    if heatmaps.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {heatmaps.shape}")
    # sigmoid is monotonic, so max-then-sigmoid == sigmoid-then-max but cheaper.
    max_logits = heatmaps.flatten(start_dim=2).max(dim=-1).values
    return torch.sigmoid(max_logits)


def sample_confidence_at_coords(
    confidence_maps: Tensor,
    coords: Tensor,
) -> Tensor:
    """Sample confidence logits at predicted coordinate locations.

    Uses bilinear interpolation via grid_sample for sub-pixel accuracy.

    Args:
        confidence_maps: (B, K, H, W) confidence logit maps.
        coords: (B, K, 2) coordinates in heatmap space (x, y).

    Returns:
        (B, K) confidence logits sampled at each coordinate.
    """
    if confidence_maps.ndim != 4:
        raise ValueError(
            f"Expected confidence_maps to have shape (B, K, H, W), "
            f"but got {tuple(confidence_maps.shape)}"
        )
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords to have shape (B, K, 2), but got {tuple(coords.shape)}")

    B, K, H, W = confidence_maps.shape

    if coords.shape[0] != B or coords.shape[1] != K:
        raise ValueError(
            f"Expected coords to have leading dimensions {(B, K)} to match "
            f"confidence_maps, but got {tuple(coords.shape[:2])}"
        )

    if H < 2 or W < 2:
        # Fallback for tiny maps
        return confidence_maps.mean(dim=(-2, -1))

    # Clamp coordinates to valid range
    coords = coords.to(device=confidence_maps.device, dtype=confidence_maps.dtype).clone()
    coords[..., 0] = coords[..., 0].clamp(0, W - 1)
    coords[..., 1] = coords[..., 1].clamp(0, H - 1)

    # Normalize to [-1, 1] for grid_sample
    grid_x = (coords[..., 0] / (W - 1)) * 2 - 1
    grid_y = (coords[..., 1] / (H - 1)) * 2 - 1

    flat_conf_maps = confidence_maps.reshape(B * K, 1, H, W)
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B * K, 1, 1, 2)

    sampled = F.grid_sample(
        flat_conf_maps,
        grid,
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    return sampled.reshape(B, K)


def compute_confidence_labels(
    predictions: Tensor,
    ground_truth: Tensor,
    threshold_fraction: float = 0.003,
    image_diagonal: float | Tensor | None = None,
    image_size: tuple[int, int] | None = None,
) -> Tensor:
    """Compute binary labels for confidence supervision.

    A prediction is "correct" if within threshold_fraction * diagonal of GT.

    Args:
        predictions: (B, K, 2) predicted coordinates.
        ground_truth: (B, K, 2) ground truth coordinates.
        threshold_fraction: Fraction of image diagonal for threshold.
        image_diagonal: Precomputed diagonal, or computed from image_size.
        image_size: (H, W) to compute diagonal if not provided.

    Returns:
        (B, K) binary labels (1.0 = correct, 0.0 = incorrect).
    """
    if image_diagonal is None:
        if image_size is None:
            raise ValueError("Must provide image_diagonal or image_size")
        H, W = image_size
        image_diagonal = (H**2 + W**2) ** 0.5

    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"predictions and ground_truth must have the same shape, "
            f"got {tuple(predictions.shape)} and {tuple(ground_truth.shape)}"
        )
    if predictions.shape[-1] != 2:
        raise ValueError(
            f"Expected last dimension to be 2 (x, y coordinates), got {predictions.shape[-1]}"
        )

    distances = torch.norm(predictions - ground_truth, dim=-1)
    threshold = torch.as_tensor(
        threshold_fraction,
        device=distances.device,
        dtype=distances.dtype,
    ) * torch.as_tensor(
        image_diagonal,
        device=distances.device,
        dtype=distances.dtype,
    )
    return (distances <= threshold).float()
