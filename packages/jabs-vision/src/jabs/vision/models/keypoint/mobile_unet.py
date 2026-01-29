"""MobileNet U-Net for keypoint detection."""

import logging
from dataclasses import dataclass, field

import torch.nn as nn
from torch import Tensor

from jabs.vision.backbones.timm import TimmBackbone, TimmBackboneConfig
from jabs.vision.core.interfaces import BaseVisionModel, OutputKeys
from jabs.vision.core.registry import MODEL_REGISTRY
from jabs.vision.modules.decoders import UNetDecoder
from jabs.vision.modules.heads import HeatmapHead

logger = logging.getLogger(__name__)


def _normalize_size(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize input size to (H, W) tuple."""
    if isinstance(value, list | tuple):
        if len(value) != 2:
            raise ValueError("input_size must be int or 2-item sequence")
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def _build_output_adjust(from_stride: int, to_stride: int) -> nn.Module:
    """Build module to adjust spatial resolution."""
    if to_stride == from_stride:
        return nn.Identity()
    elif to_stride < from_stride:
        scale = from_stride // to_stride
        return nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
    else:
        scale = to_stride // from_stride
        return nn.AvgPool2d(kernel_size=scale, stride=scale)


@dataclass
class MobileNetUNetConfig:
    """Configuration for MobileNetUNet."""

    # Backbone
    backbone: str = "mobilenetv3_large_100"
    pretrained: bool = True

    # Architecture
    num_keypoints: int = 4
    output_stride: int = 4
    decoder_channels: list[int] = field(default_factory=lambda: [256, 128, 64, 32])

    # Heatmap parameters (used by predictor, stored here for convenience)
    heatmap_sigma: float = 2.0

    # Input
    input_size: int | tuple[int, int] = 800


@MODEL_REGISTRY.register("mobilenet_unet")
class MobileNetUNet(BaseVisionModel):
    """MobileNetV3 encoder with U-Net decoder for keypoint heatmap regression.

    This model outputs raw heatmaps. Coordinate decoding is handled by
    the KeypointPredictor class for separation of concerns.
    """

    def __init__(self, cfg: MobileNetUNetConfig):
        super().__init__()
        self.cfg = cfg
        self.input_size = _normalize_size(cfg.input_size)

        logger.info(
            f"Building MobileNetUNet | backbone={cfg.backbone} | stride={cfg.output_stride}"
        )

        # 1. Backbone
        backbone_cfg = TimmBackboneConfig(
            name=cfg.backbone,
            pretrained=cfg.pretrained,
            out_indices=(0, 1, 2, 3, 4),
        )
        self.backbone = TimmBackbone(backbone_cfg)

        logger.info(f"Backbone channels: {self.backbone.channels}")
        logger.info(f"Backbone strides: {self.backbone.strides}")

        # 2. Decoder
        self.decoder = UNetDecoder(
            encoder_channels=self.backbone.channels,
            decoder_channels=cfg.decoder_channels,
        )

        # 3. Output adjustment (decoder outputs at stride 2)
        decoder_stride = 2
        self.output_adjust = _build_output_adjust(decoder_stride, cfg.output_stride)

        # 4. Heatmap head
        self.heatmap_head = HeatmapHead(
            in_channels=self.decoder.out_channels,
            num_keypoints=cfg.num_keypoints,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Dictionary with 'heatmaps' key containing (B, K, H/stride, W/stride).
        """
        features = self.backbone(x)
        decoded = self.decoder(features)
        adjusted = self.output_adjust(decoded)
        heatmaps = self.heatmap_head(adjusted)

        return {OutputKeys.HEATMAPS: heatmaps}


# =============================================================================
# Factory function
# =============================================================================


def build_mobilenet_unet(
    backbone: str = "mobilenetv3_large_100",
    pretrained: bool = True,
    num_keypoints: int = 4,
    output_stride: int = 4,
    decoder_channels: list[int] | None = None,
    heatmap_sigma: float = 2.0,
    input_size: int | tuple[int, int] = 800,
    **kwargs,
) -> MobileNetUNet:
    """Factory function for Hydra instantiation."""
    cfg = MobileNetUNetConfig(
        backbone=backbone,
        pretrained=pretrained,
        num_keypoints=num_keypoints,
        output_stride=output_stride,
        decoder_channels=decoder_channels or [256, 128, 64, 32],
        heatmap_sigma=heatmap_sigma,
        input_size=input_size,
    )
    return MobileNetUNet(cfg)
