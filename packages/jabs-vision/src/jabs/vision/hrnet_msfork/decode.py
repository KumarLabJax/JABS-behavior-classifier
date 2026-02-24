"""Heatmap decoding utilities for HRNet inference."""

import torch
from torch import Tensor


def argmax_2d_torch(heatmaps: Tensor) -> tuple[Tensor, Tensor]:
    """Get maximum score and coordinates per keypoint heatmap.

    Args:
        heatmaps: Tensor of shape (B, K, H, W).

    Returns:
        Tuple of:
            values: Tensor of shape (B, K)
            coords: Tensor of shape (B, K, 2), coordinate order is (row, col)
    """
    if heatmaps.ndim < 4:
        raise ValueError(
            f"Expected heatmaps with shape (B, K, H, W), received {tuple(heatmaps.shape)}"
        )

    max_col_vals, max_cols = torch.max(heatmaps, -1, keepdim=True)
    max_vals, max_rows = torch.max(max_col_vals, -2, keepdim=True)
    max_cols = torch.gather(max_cols, -2, max_rows)

    max_vals = max_vals.squeeze(-1).squeeze(-1)
    max_rows = max_rows.squeeze(-1).squeeze(-1)
    max_cols = max_cols.squeeze(-1).squeeze(-1)

    return max_vals, torch.stack([max_rows, max_cols], -1)
