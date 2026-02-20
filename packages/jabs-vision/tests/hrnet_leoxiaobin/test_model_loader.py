"""Tests for Leoxiaobin HRNet config/model loading."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import torch

from jabs.vision.hrnet_leoxiaobin.config import load_cfg_from_file
from jabs.vision.hrnet_leoxiaobin.model_loader import load_pose_model
from jabs.vision.hrnet_leoxiaobin.models import get_pose_net

_MINIMAL_HRNET_YAML = dedent(
    """
    CUDNN:
      DETERMINISTIC: false
      ENABLED: true
    MODEL:
      NUM_JOINTS: 2
      EXTRA:
        PRETRAINED_LAYERS: ["*"]
        FINAL_CONV_KERNEL: 1
        STAGE2:
          NUM_MODULES: 1
          NUM_BRANCHES: 2
          NUM_BLOCKS: [1, 1]
          NUM_CHANNELS: [8, 16]
          BLOCK: BASIC
          FUSE_METHOD: SUM
        STAGE3:
          NUM_MODULES: 1
          NUM_BRANCHES: 3
          NUM_BLOCKS: [1, 1, 1]
          NUM_CHANNELS: [8, 16, 32]
          BLOCK: BASIC
          FUSE_METHOD: SUM
        STAGE4:
          NUM_MODULES: 1
          NUM_BRANCHES: 4
          NUM_BLOCKS: [1, 1, 1, 1]
          NUM_CHANNELS: [8, 16, 32, 64]
          BLOCK: BASIC
          FUSE_METHOD: SUM
    TEST:
      MODEL_FILE: ""
    """
)


def _write_cfg(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "hrnet_minimal.yaml"
    cfg_path.write_text(_MINIMAL_HRNET_YAML)
    return cfg_path


def test_load_cfg_from_file_merges_values(tmp_path: Path) -> None:
    """YAML config is merged into default YACS config as expected."""
    cfg_path = _write_cfg(tmp_path)
    cfg = load_cfg_from_file(cfg_path)
    assert cfg.MODEL.NUM_JOINTS == 2
    assert cfg.MODEL.EXTRA.STAGE2.NUM_BRANCHES == 2
    assert cfg.TEST.MODEL_FILE == ""


def test_load_pose_model_builds_and_loads_checkpoint(tmp_path: Path) -> None:
    """Model loader builds HRNet and loads a checkpoint for inference."""
    cfg_path = _write_cfg(tmp_path)
    cfg = load_cfg_from_file(cfg_path)
    seed_model = get_pose_net(cfg, is_train=False)

    checkpoint_path = tmp_path / "hrnet_seed.pth"
    torch.save(seed_model.state_dict(), checkpoint_path)

    model, loaded_cfg = load_pose_model(
        config_path=cfg_path,
        checkpoint_path=checkpoint_path,
        device="cpu",
    )

    assert str(checkpoint_path) == loaded_cfg.TEST.MODEL_FILE
    assert model.training is False
    assert next(model.parameters()).device.type == "cpu"

    output = model(torch.randn(1, 3, 64, 64))
    assert output.shape[1] == 2
