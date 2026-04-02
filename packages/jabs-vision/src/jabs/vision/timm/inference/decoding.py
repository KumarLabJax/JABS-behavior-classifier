"""Heatmap decoding utilities for keypoint detection."""

from __future__ import annotations

import torch
from torch import Tensor


def get_max_preds(heatmaps: Tensor) -> tuple[Tensor, Tensor]:
    """Get keypoint predictions from the maximum value of each heatmap.

    Args:
        heatmaps: Heatmaps of shape (B, K, H, W).

    Returns:
        Tuple of (coords, maxvals) where:
            - coords: (B, K, 2) integer coordinates (x, y)
            - maxvals: (B, K, 1) maximum values
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            f"Expected heatmaps to have 4 dimensions (B, K, H, W), "
            f"but got shape {tuple(heatmaps.shape)}"
        )

    B, K, H, W = heatmaps.shape

    # Flatten spatial dimensions: (B, K, H*W)
    heatmaps_flat = heatmaps.reshape(B, K, -1)

    # Get max values and indices
    maxvals, idx = torch.max(heatmaps_flat, dim=2)
    maxvals = maxvals.unsqueeze(-1)

    # Convert flat index to (x, y) coordinates
    preds = torch.zeros((B, K, 2), device=heatmaps.device, dtype=heatmaps.dtype)
    preds[..., 0] = idx % W  # x coordinate
    preds[..., 1] = idx // W  # y coordinate

    return preds, maxvals


def decode_heatmaps(
    heatmaps: Tensor,
    use_dark: bool = False,
    sigma: float = 2.0,
) -> Tensor:
    """Decode heatmaps to (x, y) coordinates.

    Args:
        heatmaps: Heatmaps of shape (B, K, H, W).
        use_dark: If True, apply DARK refinement for sub-pixel accuracy.
        sigma: Gaussian sigma (used for DARK refinement documentation).

    Returns:
        Coordinates of shape (B, K, 2).

    Note:
        DARK (Distribution-Aware coordinate Representation of Keypoints) uses
        Taylor expansion to achieve sub-pixel accuracy. Reference:
        https://arxiv.org/abs/1910.06278
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device

    # Get integer argmax coordinates
    coords, _ = get_max_preds(heatmaps)

    if not use_dark:
        return coords

    # DARK refinement using Taylor expansion
    # p = p_int - H^-1 * g
    # where g is gradient (1st derivative), H is Hessian (2nd derivative)

    # Clamp coordinates to avoid boundary issues during gradient computation
    px = coords[..., 0].long().clamp(2, W - 3)
    py = coords[..., 1].long().clamp(2, H - 3)

    refined_coords = coords.clone()

    for b in range(B):
        for k in range(K):
            cx, cy = px[b, k], py[b, k]
            val = heatmaps[b, k]

            # First derivative (central difference)
            # dx = 0.5 * (h[x+1] - h[x-1])
            # dy = 0.5 * (h[y+1] - h[y-1])
            dx = 0.5 * (val[cy, cx + 1] - val[cy, cx - 1])
            dy = 0.5 * (val[cy + 1, cx] - val[cy - 1, cx])

            # Second derivative (central difference)
            # dxx = h[x+1] - 2*h[x] + h[x-1]
            dxx = val[cy, cx + 1] - 2 * val[cy, cx] + val[cy, cx - 1]
            dyy = val[cy + 1, cx] - 2 * val[cy, cx] + val[cy - 1, cx]
            dxy = 0.25 * (
                val[cy + 1, cx + 1]
                - val[cy + 1, cx - 1]
                - val[cy - 1, cx + 1]
                + val[cy - 1, cx - 1]
            )

            # Hessian matrix and gradient vector
            H = torch.tensor([[dxx, dxy], [dxy, dyy]], device=device, dtype=val.dtype)
            g = torch.tensor([dx, dy], device=device, dtype=val.dtype)

            try:
                H_inv = torch.inverse(H)
                delta = -torch.matmul(H_inv, g)
                refined_coords[b, k, 0] += delta[0]
                refined_coords[b, k, 1] += delta[1]
            except RuntimeError:
                # Singular matrix, skip refinement for this keypoint
                continue

    return refined_coords
