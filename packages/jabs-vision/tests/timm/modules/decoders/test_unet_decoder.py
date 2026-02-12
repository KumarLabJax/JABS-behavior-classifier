"""Tests for UNetDecoder."""

import torch

from jabs.vision.timm.modules import UNetDecoder


def test_unet_decoder_init_defaults() -> None:
    """Test initialization with default decoder channels."""
    # Example: Encoder with 5 levels (e.g., MobileNetV3)
    # Channels: [16, 24, 40, 112, 960]
    encoder_channels = [16, 24, 40, 112, 960]

    decoder = UNetDecoder(encoder_channels)

    # Defaults are [256, 128, 64, 32]
    assert len(decoder.blocks) == 4
    assert decoder.out_channels == 32

    # Verify first block (deepest)
    # Input: 960 (enc[-1])
    # Skip: 112 (enc[-2])
    # Output: 256 (dec[0])
    block0 = decoder.blocks[0]
    # Check conv input channels: (In + Skip)
    # Note: DecoderBlock structure is upsample -> cat -> conv
    # Conv input = 960 + 112 = 1072
    assert block0.conv[0][0].in_channels == 960 + 112
    assert block0.conv[0][0].out_channels == 256

    # Verify last block (shallowest)
    # Input: 64 (dec[-2])
    # Skip: 16 (enc[0]) - Note: enc[0] is the 16ch layer
    # Output: 32 (dec[-1])
    block_last = decoder.blocks[-1]
    assert block_last.conv[0][0].in_channels == 64 + 16
    assert block_last.conv[0][0].out_channels == 32


def test_unet_decoder_custom_channels() -> None:
    """Test initialization with custom decoder channels."""
    encoder_channels = [64, 128, 256, 512]
    decoder_channels = [256, 128, 64]

    decoder = UNetDecoder(encoder_channels, decoder_channels)

    assert len(decoder.blocks) == 3
    assert decoder.out_channels == 64


def test_unet_decoder_forward() -> None:
    """Test forward pass with correct feature maps."""
    # Setup matching MobileNetV3-Large structure
    encoder_channels = [16, 24, 40, 112, 960]
    strides = [2, 4, 8, 16, 32]

    decoder = UNetDecoder(encoder_channels)

    # Create dummy features
    features = []
    base_size = 256
    for ch, stride in zip(encoder_channels, strides, strict=False):
        size = base_size // stride
        features.append(torch.randn(1, ch, size, size))

    out = decoder(features)

    # Expected output:
    # Final decoder block aligns with the earliest skip connection used
    # The loop goes through len(decoder_channels) = 4 blocks.
    # Features used:
    # 0: 960 (32x) + 112 (16x) -> 256 (16x)
    # 1: 256 (16x) + 40 (8x) -> 128 (8x)
    # 2: 128 (8x) + 24 (4x) -> 64 (4x)
    # 3: 64 (4x) + 16 (2x) -> 32 (2x)

    expected_stride = 2
    expected_size = base_size // expected_stride

    assert out.shape == (1, 32, expected_size, expected_size)


def test_unet_decoder_forward_mismatch() -> None:
    """Test forward pass where encoder features might be different sizes (e.g. padding)."""
    encoder_channels = [10, 20, 30]
    decoder_channels = [15, 5]

    decoder = UNetDecoder(encoder_channels, decoder_channels)

    # Simulate sizes that aren't perfect powers of 2 (odd sizes)
    f1 = torch.randn(1, 10, 100, 100)
    f2 = torch.randn(1, 20, 50, 50)
    f3 = torch.randn(1, 30, 25, 25)

    out = decoder([f1, f2, f3])

    # f3(25) -> up(50) + f2(50) -> dec1(15, 50x50)
    # dec1(50) -> up(100) + f1(100) -> dec2(5, 100x100)
    assert out.shape == (1, 5, 100, 100)


def test_unet_decoder_skip_logic() -> None:
    """Verify logic when fewer decoder blocks than encoder stages."""
    # 4 encoder stages, but only 1 decoder block
    encoder_channels = [10, 20, 30, 40]
    decoder_channels = [25]

    decoder = UNetDecoder(encoder_channels, decoder_channels)

    assert len(decoder.blocks) == 1

    # Should use enc[-1] (40) as input and enc[-2] (30) as skip
    block = decoder.blocks[0]
    assert block.conv[0][0].in_channels == 40 + 30
    assert block.conv[0][0].out_channels == 25


def test_unet_decoder_no_skip_available() -> None:
    """Edge case: Decoder deeper than encoder allows?

    The code has: `skip_ch = enc_ch[i + 1] if i + 1 < len(enc_ch) else 0`
    This implies we can have more decoder blocks than skips if configured.
    """
    encoder_channels = [10, 20]  # 2 levels
    decoder_channels = [15, 10, 5]  # 3 levels

    decoder = UNetDecoder(encoder_channels, decoder_channels)

    # Block 0: In=20, Skip=10 (enc[0]), Out=15
    b0 = decoder.blocks[0]
    assert b0.conv[0][0].in_channels == 20 + 10

    # Block 1: In=15, Skip=0 (no more encoder layers), Out=10
    b1 = decoder.blocks[1]
    assert b1.conv[0][0].in_channels == 15 + 0

    # Block 2: In=10, Skip=0, Out=5
    b2 = decoder.blocks[2]
    assert b2.conv[0][0].in_channels == 10 + 0


def test_unet_decoder_jit_trace() -> None:
    """Test JIT traceability of the full decoder."""
    enc_ch = [16, 32, 64]
    dec_ch = [32, 16]
    decoder = UNetDecoder(enc_ch, dec_ch)

    # Features: [16ch (32x32), 32ch (16x16), 64ch (8x8)]
    features = [
        torch.randn(1, 16, 32, 32),
        torch.randn(1, 32, 16, 16),
        torch.randn(1, 64, 8, 8),
    ]

    # JIT requires list inputs to be typed or specialized.
    # We can trace it by passing the example input.
    traced = torch.jit.trace(decoder, (features,))

    out_orig = decoder(features)
    out_traced = traced(features)

    assert torch.allclose(out_orig, out_traced)
