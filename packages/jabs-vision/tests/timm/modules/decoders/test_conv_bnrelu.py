"""Tests for ConvBNReLU block."""

import pytest
import torch
import torch.nn as nn

from jabs.vision.timm.modules import ConvBNReLU


def test_conv_bnrelu_init() -> None:
    """Test proper initialization of layers."""
    in_ch = 32
    out_ch = 64
    block = ConvBNReLU(in_ch, out_ch)

    # Check structure
    assert len(block) == 3
    assert isinstance(block[0], nn.Conv2d)
    assert isinstance(block[1], nn.BatchNorm2d)
    assert isinstance(block[2], nn.ReLU)

    # Check Conv2d properties
    assert block[0].in_channels == in_ch
    assert block[0].out_channels == out_ch
    assert block[0].kernel_size == (3, 3)
    assert block[0].padding == (1, 1)
    assert block[0].bias is None  # Should be False due to BatchNorm


@pytest.mark.parametrize("kernel_size, padding", [(3, 1), (1, 0), (5, 2)])
def test_conv_bnrelu_params(kernel_size: int, padding: int) -> None:
    """Test initialization with different kernel sizes and padding."""
    block = ConvBNReLU(16, 32, kernel_size=kernel_size, padding=padding)

    assert block[0].kernel_size == (kernel_size, kernel_size)
    assert block[0].padding == (padding, padding)


@pytest.mark.parametrize(
    "in_ch, out_ch, height, width",
    [
        (3, 16, 32, 32),
        (32, 64, 16, 16),
        (1, 1, 10, 10),
    ],
)
def test_conv_bnrelu_forward(in_ch: int, out_ch: int, height: int, width: int) -> None:
    """Test forward pass output shape."""
    block = ConvBNReLU(in_ch, out_ch)
    x = torch.randn(1, in_ch, height, width)

    out = block(x)

    assert out.shape == (1, out_ch, height, width)
    assert not torch.isnan(out).any()


def test_conv_bnrelu_jit_trace() -> None:
    """Test that the module is JIT traceable."""
    block = ConvBNReLU(16, 32)
    x = torch.randn(1, 16, 24, 24)
    traced = torch.jit.trace(block, x)
    assert torch.allclose(block(x), traced(x))
