from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from mmpose.registry import MODELS
from torch import nn
from torchvision.models import (
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
)
from torchvision.models.efficientnet import (
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
)

try:  # Optional: allow this backbone to be used in MMDetection configs.
    from mmdet.registry import MODELS as MMDET_MODELS
except Exception:  # pragma: no cover - mmdet may not be installed
    MMDET_MODELS = None


@MODELS.register_module()
class EfficientNetV2Backbone(nn.Module):
    """EfficientNetV2 backbone wrapper using torchvision models.

    Returns a tuple of feature maps based on out_indices.
    """

    def __init__(
        self,
        variant: str = "s",
        out_indices: Sequence[int] = (-1,),
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.out_indices = tuple(out_indices)

        model = self._build_model(variant=variant, pretrained=pretrained)
        self.features = model.features
        self._out_indices = self._normalize_out_indices(self.out_indices, len(self.features))

    def _build_model(self, variant: str, pretrained: bool) -> nn.Module:
        if variant == "s":
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            return efficientnet_v2_s(weights=weights)
        if variant == "m":
            weights = EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
            return efficientnet_v2_m(weights=weights)
        if variant == "l":
            weights = EfficientNet_V2_L_Weights.DEFAULT if pretrained else None
            return efficientnet_v2_l(weights=weights)
        raise ValueError(f"Unsupported EfficientNetV2 variant: {variant}")

    @staticmethod
    def _normalize_out_indices(out_indices: Iterable[int], num_layers: int) -> tuple[int, ...]:
        normalized = []
        for idx in out_indices:
            if idx < 0:
                idx = num_layers + idx
            if idx < 0 or idx >= num_layers:
                raise IndexError(f"out_indices {out_indices} is out of range for {num_layers} layers")
            normalized.append(idx)
        return tuple(normalized)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass through the backbone.

        Args:
            x: Input tensor.

        Returns:
            Tuple of feature maps at the specified out_indices.
        """
        outs = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self._out_indices:
                outs.append(x)
        return tuple(outs)


if MMDET_MODELS is not None:
    MMDET_MODELS.register_module(module=EfficientNetV2Backbone)
