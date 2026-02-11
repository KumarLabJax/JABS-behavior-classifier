"""Tests for TimmBackboneConfig."""

from jabs.vision.timm.backbone import TimmBackboneConfig


def test_timm_backbone_config_defaults() -> None:
    """Test default values of TimmBackboneConfig."""
    config = TimmBackboneConfig()

    assert config.name == "mobilenetv3_large_100"
    assert config.pretrained is True
    assert config.out_indices == (0, 1, 2, 3, 4)


def test_timm_backbone_config_custom() -> None:
    """Test custom values for TimmBackboneConfig."""
    config = TimmBackboneConfig(name="resnet18", pretrained=False, out_indices=(1, 2, 3))

    assert config.name == "resnet18"
    assert config.pretrained is False
    assert config.out_indices == (1, 2, 3)
