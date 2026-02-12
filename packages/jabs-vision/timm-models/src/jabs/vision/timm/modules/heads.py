"""Task-specific head modules."""

import torch.nn as nn
from torch import Tensor


class HeatmapHead(nn.Module):
    """Head for heatmap regression (keypoint detection)."""

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int,
        intermediate_channels: int | None = None,
    ):
        super().__init__()
        mid_ch = intermediate_channels or in_channels
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, num_keypoints, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward computation."""
        return self.head(x)
