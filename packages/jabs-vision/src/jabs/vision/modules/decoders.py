"""Decoder modules for vision models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Single U-Net decoder block: upsample, concat skip, double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        """Initialize decoder block.

        Args:
            in_ch: Input channels from previous decoder stage.
            skip_ch: Channels from skip connection (0 if no skip).
            out_ch: Output channels.
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        """Apply decoder block.

        Args:
            x: Input tensor from previous stage.
            skip: Optional skip connection tensor.

        Returns:
            Decoded tensor.
        """
        x = self.upsample(x)

        if skip is not None:
            # Handle size mismatch from odd dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class UNetDecoder(nn.Module):
    """U-Net style decoder with skip connections.

    Takes multi-scale encoder features and progressively upsamples
    while fusing with skip connections.
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int] | None = None,
    ):
        """Initialize U-Net decoder.

        Args:
            encoder_channels: Channel dims from encoder, low-to-high stride
                (e.g., [16, 24, 40, 112, 960] for MobileNetV3).
            decoder_channels: Output channels per decoder stage.
                Defaults to [256, 128, 64, 32].
        """
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        # Reverse: we decode from deepest to shallowest
        enc_ch = encoder_channels[::-1]

        self.blocks = nn.ModuleList()
        in_ch = enc_ch[0]

        for i, out_ch in enumerate(decoder_channels):
            # Skip from next encoder stage (going backwards through reversed list)
            skip_ch = enc_ch[i + 1] if i + 1 < len(enc_ch) else 0
            self.blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.out_channels: int = decoder_channels[-1]

    def forward(self, features: list[Tensor]) -> Tensor:
        """Run decoder forward pass.

        Args:
            features: Encoder features ordered low-to-high stride
                (e.g., [stride2, stride4, stride8, stride16, stride32]).

        Returns:
            Decoded feature map at stride 2.
        """
        features = features[::-1]  # Reverse: deepest first

        x = features[0]
        for i, block in enumerate(self.blocks):
            skip = features[i + 1] if i + 1 < len(features) else None
            x = block(x, skip)

        return x
