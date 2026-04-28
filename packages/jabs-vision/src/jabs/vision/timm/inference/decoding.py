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
            - coords: (B, K, 2) floating-point pixel coordinates (x, y)
              whose values are integer-valued argmax indices
            - maxvals: (B, K, 1) maximum values
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            f"Expected heatmaps to have 4 dimensions (B, K, H, W), "
            f"but got shape {tuple(heatmaps.shape)}"
        )

    B, K, _, W = heatmaps.shape

    # Flatten spatial dimensions: (B, K, H*W)
    heatmaps_flat = heatmaps.reshape(B, K, -1)

    # Get max values and indices
    maxvals, idx = torch.max(heatmaps_flat, dim=2)
    maxvals = maxvals.unsqueeze(-1)

    # Convert flat index to (x, y) coordinates
    preds = torch.stack(
        [
            (idx % W).to(dtype=heatmaps.dtype),
            torch.div(idx, W, rounding_mode="floor").to(dtype=heatmaps.dtype),
        ],
        dim=-1,
    )

    return preds, maxvals


def decode_heatmaps(
    heatmaps: Tensor,
    use_dark: bool = False,
) -> Tensor:
    """Decode heatmaps to (x, y) coordinates.

    Args:
        heatmaps: Heatmaps of shape (B, K, H, W).
        use_dark: If True, apply DARK refinement for sub-pixel accuracy.

    Returns:
        Coordinates of shape (B, K, 2).

    Note:
        DARK (Distribution-Aware coordinate Representation of Keypoints) uses
        Taylor expansion to achieve sub-pixel accuracy. Reference:
        https://arxiv.org/abs/1910.06278
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            f"Expected heatmaps to have 4 dimensions (B, K, H, W), "
            f"but got shape {tuple(heatmaps.shape)}"
        )

    B, K, H, W = heatmaps.shape

    # Get argmax coordinates in floating point so DARK can refine them in place.
    coords, _ = get_max_preds(heatmaps)

    if not use_dark or H < 5 or W < 5:
        return coords

    # DARK refinement using Taylor expansion
    # p = p_int - H^-1 * g
    # where g is gradient (1st derivative), H is Hessian (2nd derivative)

    # Clamp coordinates to avoid boundary issues during gradient computation
    px = coords[..., 0].long().clamp(2, W - 3)  # (B, K)
    py = coords[..., 1].long().clamp(2, H - 3)  # (B, K)

    # Batch indices for advanced indexing: (B, K) each
    batch_idx = torch.arange(B, device=heatmaps.device)[:, None]
    keypoint_idx = torch.arange(K, device=heatmaps.device)[None, :]

    # First derivative (central difference)
    dx = 0.5 * (
        heatmaps[batch_idx, keypoint_idx, py, px + 1]
        - heatmaps[batch_idx, keypoint_idx, py, px - 1]
    )
    dy = 0.5 * (
        heatmaps[batch_idx, keypoint_idx, py + 1, px]
        - heatmaps[batch_idx, keypoint_idx, py - 1, px]
    )

    # Second derivative (central difference)
    center = heatmaps[batch_idx, keypoint_idx, py, px]
    dxx = (
        heatmaps[batch_idx, keypoint_idx, py, px + 1]
        - 2 * center
        + heatmaps[batch_idx, keypoint_idx, py, px - 1]
    )
    dyy = (
        heatmaps[batch_idx, keypoint_idx, py + 1, px]
        - 2 * center
        + heatmaps[batch_idx, keypoint_idx, py - 1, px]
    )
    dxy = 0.25 * (
        heatmaps[batch_idx, keypoint_idx, py + 1, px + 1]
        - heatmaps[batch_idx, keypoint_idx, py + 1, px - 1]
        - heatmaps[batch_idx, keypoint_idx, py - 1, px + 1]
        + heatmaps[batch_idx, keypoint_idx, py - 1, px - 1]
    )

    # Build batched Hessians (B, K, 2, 2) and gradients (B, K, 2)
    hessian = torch.stack(
        [
            torch.stack([dxx, dxy], dim=-1),
            torch.stack([dxy, dyy], dim=-1),
        ],
        dim=-2,
    )
    grad = torch.stack([dx, dy], dim=-1)

    # Flatten to (B*K, 2, 2) and (B*K, 2) for batched solve
    hessian_flat = hessian.reshape(-1, 2, 2)
    grad_flat = grad.reshape(-1, 2)

    # Only solve where the Hessian is non-singular and finite
    det = hessian_flat[:, 0, 0] * hessian_flat[:, 1, 1] - hessian_flat[:, 0, 1] ** 2
    solvable = torch.isfinite(hessian_flat).all(dim=(-2, -1)) & torch.isfinite(grad_flat).all(
        dim=-1
    )
    solvable = solvable & (det.abs() > torch.finfo(hessian_flat.dtype).eps)

    refined_coords = coords.clone()

    if torch.any(solvable):
        delta = torch.linalg.solve(
            hessian_flat[solvable],
            -grad_flat[solvable].unsqueeze(-1),
        ).squeeze(-1)
        refined_flat = refined_coords.reshape(-1, 2)
        refined_flat[solvable] += delta

    return refined_coords
