"""Tests for DecoderBlock."""

import pytest
import torch
import torch.nn as nn

from jabs.vision.timm.modules import DecoderBlock


def test_decoder_block_init() -> None:
    """Test initialization structure."""
    in_ch = 64
    skip_ch = 32
    out_ch = 16

    block = DecoderBlock(in_ch, skip_ch, out_ch)

    assert isinstance(block.upsample, nn.Upsample)
    assert isinstance(block.conv, nn.Sequential)
    # Check that conv block handles concatenated channels
    # First ConvBNReLU input should be in_ch + skip_ch
    assert block.conv[0][0].in_channels == in_ch + skip_ch
    assert block.conv[0][0].out_channels == out_ch
    # Second ConvBNReLU input/output should be out_ch
    assert block.conv[1][0].in_channels == out_ch
    assert block.conv[1][0].out_channels == out_ch


def test_decoder_block_no_skip() -> None:
    """Test forward pass without skip connection."""
    block = DecoderBlock(in_ch=64, skip_ch=0, out_ch=32)

    # Input 64ch, 16x16
    x = torch.randn(1, 64, 16, 16)

    # Expect upsample to 32x32
    out = block(x, skip=None)

    assert out.shape == (1, 32, 32, 32)


def test_decoder_block_with_skip() -> None:
    """Test forward pass with skip connection of matching spatial dimensions."""
    block = DecoderBlock(in_ch=64, skip_ch=32, out_ch=16)

    x = torch.randn(1, 64, 16, 16)
    skip = torch.randn(1, 32, 32, 32)  # Already 2x spatial size of x

    out = block(x, skip)

    assert out.shape == (1, 16, 32, 32)


def test_decoder_block_size_mismatch() -> None:
    """Test robustness to size mismatches (e.g., odd input dimensions)."""
    block = DecoderBlock(in_ch=64, skip_ch=32, out_ch=16)

    # Case where 2x upsample doesn't exactly match skip
    # x: 16x16 -> upsample -> 32x32
    # skip: 33x33 (odd dimension in encoder)
    x = torch.randn(1, 64, 16, 16)
    skip = torch.randn(1, 32, 33, 33)

    out = block(x, skip)

    # Output should match skip dimensions
    assert out.shape == (1, 16, 33, 33)


@pytest.mark.parametrize("in_ch, skip_ch, out_ch", [(32, 16, 16), (128, 64, 32)])
def test_decoder_block_channels(in_ch: int, skip_ch: int, out_ch: int) -> None:
    """Parametrized test for channel configurations."""
    block = DecoderBlock(in_ch, skip_ch, out_ch)
    x = torch.randn(1, in_ch, 10, 10)
    skip = torch.randn(1, skip_ch, 20, 20)

    out = block(x, skip)
    assert out.shape == (1, out_ch, 20, 20)
