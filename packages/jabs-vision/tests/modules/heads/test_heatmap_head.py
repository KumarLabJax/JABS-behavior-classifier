"""Tests for HeatmapHead."""

import pytest
import torch
import torch.nn as nn

from jabs.vision.modules.heads import HeatmapHead


@pytest.mark.parametrize("in_channels", [16, 32])
@pytest.mark.parametrize("num_keypoints", [5, 17])
def test_heatmap_head_init_defaults(in_channels: int, num_keypoints: int) -> None:
    """Test initialization with default arguments."""
    head = HeatmapHead(in_channels, num_keypoints)

    # Check overall structure
    assert isinstance(head.head, nn.Sequential)
    assert len(head.head) == 4

    # Layer 0: Conv2d (3x3)
    conv1 = head.head[0]
    assert isinstance(conv1, nn.Conv2d)
    assert conv1.in_channels == in_channels
    assert conv1.out_channels == in_channels  # Default is in_channels
    assert conv1.kernel_size == (3, 3)
    assert conv1.padding == (1, 1)
    assert conv1.bias is None

    # Layer 1: BatchNorm
    bn = head.head[1]
    assert isinstance(bn, nn.BatchNorm2d)
    assert bn.num_features == in_channels

    # Layer 2: ReLU
    relu = head.head[2]
    assert isinstance(relu, nn.ReLU)
    assert relu.inplace

    # Layer 3: Conv2d (1x1)
    conv2 = head.head[3]
    assert isinstance(conv2, nn.Conv2d)
    assert conv2.in_channels == in_channels
    assert conv2.out_channels == num_keypoints
    assert conv2.kernel_size == (1, 1)


@pytest.mark.parametrize("in_channels", [32])
@pytest.mark.parametrize("intermediate_channels", [16, 64])
def test_heatmap_head_init_custom_intermediate(
    in_channels: int, intermediate_channels: int
) -> None:
    """Test initialization with custom intermediate channels."""
    num_keypoints = 10
    head = HeatmapHead(in_channels, num_keypoints, intermediate_channels=intermediate_channels)

    # Check channel dimensions flow
    conv1 = head.head[0]
    assert conv1.out_channels == intermediate_channels

    bn = head.head[1]
    assert bn.num_features == intermediate_channels

    conv2 = head.head[3]
    assert conv2.in_channels == intermediate_channels
    assert conv2.out_channels == num_keypoints


def test_heatmap_head_forward() -> None:
    """Test forward pass shape and output validity."""
    in_channels = 16
    num_keypoints = 4
    spatial_dim = 32
    batch_size = 2

    head = HeatmapHead(in_channels, num_keypoints)

    # Create dummy input: [B, C, H, W]
    x = torch.randn(batch_size, in_channels, spatial_dim, spatial_dim)
    output = head(x)

    # Check output shape: [B, num_keypoints, H, W]
    expected_shape = (batch_size, num_keypoints, spatial_dim, spatial_dim)
    assert output.shape == expected_shape

    # Check that output is a tensor and not NaN
    assert isinstance(output, torch.Tensor)
    assert not torch.isnan(output).any()


def test_heatmap_head_backward() -> None:
    """Test backward pass to ensure gradients flow correctly."""
    head = HeatmapHead(in_channels=8, num_keypoints=2)
    x = torch.randn(1, 8, 16, 16, requires_grad=True)

    output = head(x)
    loss = output.sum()
    loss.backward()

    # Input gradients should exist
    assert x.grad is not None

    # Model parameter gradients should exist
    has_grad = False
    for param in head.parameters():
        assert param.grad is not None
        has_grad = True
    assert has_grad


def test_heatmap_head_jit_script() -> None:
    """Test TorchScript compatibility."""
    head = HeatmapHead(in_channels=8, num_keypoints=2)
    head.eval()

    # Script the model
    scripted_head = torch.jit.script(head)

    x = torch.randn(1, 8, 16, 16)

    # Compare outputs
    with torch.no_grad():
        original_out = head(x)
        scripted_out = scripted_head(x)

    assert torch.allclose(original_out, scripted_out)


@pytest.mark.parametrize("dim", [1, 3])
def test_heatmap_head_invalid_input_dim(dim: int) -> None:
    """Test that input with wrong dimensions raises error."""
    head = HeatmapHead(in_channels=8, num_keypoints=2)

    # Create input with wrong number of dimensions
    # Conv2d expects 4D input (B, C, H, W) or 3D (C, H, W) unbatched
    shape = [8] * dim
    x = torch.randn(*shape)

    with pytest.raises((RuntimeError, ValueError)):
        head(x)
