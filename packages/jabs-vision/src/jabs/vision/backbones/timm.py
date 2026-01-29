"""Timm backbone wrapper for feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn
from torch import Tensor

try:
    import timm
except ImportError:
    timm = None  # type: ignore[assignment]


@dataclass
class TimmBackboneConfig:
    """Configuration for TimmBackbone.

    Attributes:
        name: Name of the timm model to use.
        pretrained: Whether to load pretrained weights.
        out_indices: Which feature stages to return (0=stem, 1-4=stages).
    """

    name: str = "mobilenetv3_large_100"
    pretrained: bool = True
    out_indices: tuple[int, ...] = field(default_factory=lambda: (0, 1, 2, 3, 4))


class TimmBackbone(nn.Module):
    """Wrapper around timm models for multi-scale feature extraction.

    This module creates a feature extractor from any timm model that supports
    `features_only=True`, returning feature maps at multiple scales.

    Example:
        ``` py
        cfg = TimmBackboneConfig(name="mobilenetv3_small_100", pretrained=True)
        backbone = TimmBackbone(cfg)
        x = torch.randn(1, 3, 256, 256)
        features = backbone(x)  # List of feature tensors
        print([f.shape for f in features])
        ```
    """

    def __init__(self, cfg: TimmBackboneConfig) -> None:
        super().__init__()

        if timm is None:
            raise ImportError(
                "timm is required for TimmBackbone. "
                "Install it with: pip install 'jabs-vision[timm]' or pip install timm"
            )

        self.cfg = cfg

        # Create feature extractor
        self.model = timm.create_model(
            cfg.name,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=cfg.out_indices,
        )

        # Get feature info for channels and strides
        self._channels = [info["num_chs"] for info in self.model.feature_info]
        self._strides = [info["reduction"] for info in self.model.feature_info]

    @property
    def channels(self) -> list[int]:
        """Number of channels at each feature level."""
        return self._channels

    @property
    def strides(self) -> list[int]:
        """Spatial reduction (stride) at each feature level."""
        return self._strides

    def forward(self, x: Tensor) -> list[Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            List of feature tensors, one per output index.
        """
        return self.model(x)
