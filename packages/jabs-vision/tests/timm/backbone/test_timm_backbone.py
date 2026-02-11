"""Tests for TimmBackbone."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from jabs.vision.timm.backbone import TimmBackbone, TimmBackboneConfig


@pytest.fixture
def mock_timm():
    """Fixture to mock timm package."""
    with patch("jabs.vision.backbones.timm.timm") as mock_timm_pkg:
        # Setup create_model mock
        mock_model = MagicMock()
        mock_timm_pkg.create_model.return_value = mock_model

        # Setup default feature info for the mock model
        # Simulating a model with 5 stages (0-4)
        mock_model.feature_info = [
            {"num_chs": 16, "reduction": 2},  # Stage 0
            {"num_chs": 24, "reduction": 4},  # Stage 1
            {"num_chs": 40, "reduction": 8},  # Stage 2
            {"num_chs": 112, "reduction": 16},  # Stage 3
            {"num_chs": 960, "reduction": 32},  # Stage 4
        ]

        # Setup forward pass to return dummy tensors based on out_indices
        def side_effect_forward(x):
            # Just return a list of tensors matching the number of output stages expected
            # Note: In reality, create_model(features_only=True) returns a model that
            # returns a list of tensors. The wrapper assumes this list corresponds
            # to the requested indices.
            # We'll simulate returning 5 features for the default case
            return [torch.randn(1, info["num_chs"], 10, 10) for info in mock_model.feature_info]

        mock_model.side_effect = side_effect_forward

        yield mock_timm_pkg


def test_timm_missing_dependency():
    """Test error when timm is not installed."""
    # We need to simulate timm being None in the module
    with patch("jabs.vision.backbones.timm.timm", None):
        cfg = TimmBackboneConfig()
        with pytest.raises(ImportError, match="timm is required"):
            TimmBackbone(cfg)


def test_timm_backbone_init(mock_timm):
    """Test successful initialization."""
    cfg = TimmBackboneConfig(name="mobilenetv3_large_100", pretrained=True)
    backbone = TimmBackbone(cfg)

    assert backbone.cfg == cfg
    mock_timm.create_model.assert_called_once_with(
        "mobilenetv3_large_100",
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3, 4),
    )

    # Verify properties derived from feature_info
    expected_channels = [16, 24, 40, 112, 960]
    expected_strides = [2, 4, 8, 16, 32]

    assert backbone.channels == expected_channels
    assert backbone.strides == expected_strides


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0"])
@pytest.mark.parametrize("pretrained", [True, False])
def test_timm_backbone_params(mock_timm, name, pretrained):
    """Test initialization with different parameters."""
    cfg = TimmBackboneConfig(name=name, pretrained=pretrained)
    TimmBackbone(cfg)

    mock_timm.create_model.assert_called_with(
        name,
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3, 4),
    )


def test_timm_backbone_forward(mock_timm):
    """Test forward pass."""
    cfg = TimmBackboneConfig()
    backbone = TimmBackbone(cfg)

    x = torch.randn(2, 3, 224, 224)
    features = backbone(x)

    # Our mock side effect returns 5 tensors
    assert isinstance(features, list)
    assert len(features) == 5
    assert all(isinstance(f, torch.Tensor) for f in features)

    # Verify the model was called with input
    backbone.model.assert_called_once_with(x)


@pytest.mark.parametrize("out_indices", [(1, 2, 3), (0, 4)])
def test_timm_backbone_out_indices(mock_timm, out_indices):
    """Test initialization with specific out_indices."""
    cfg = TimmBackboneConfig(out_indices=out_indices)
    TimmBackbone(cfg)

    mock_timm.create_model.assert_called_with(
        "mobilenetv3_large_100",
        pretrained=True,
        features_only=True,
        out_indices=out_indices,
    )


def test_timm_backbone_properties_read_only(mock_timm):
    """Verify properties are read-only (implicitly by lack of setters)."""
    cfg = TimmBackboneConfig()
    backbone = TimmBackbone(cfg)

    with pytest.raises(AttributeError):
        backbone.channels = [1, 2, 3]

    with pytest.raises(AttributeError):
        backbone.strides = [1, 2, 3]
