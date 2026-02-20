"""Model/config loading for the Leoxiaobin HRNet backend."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .models import get_pose_net
from .single_pose import SinglePoseInferenceResult, predict_single_pose


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _require_config_loader() -> Any:
    try:
        from .config import load_cfg_from_file
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised when optional dep missing
        if exc.name == "yacs":
            raise ImportError(
                "The Leoxiaobin HRNet config loader requires 'yacs'. "
                "Install with: pip install 'jabs-vision[hrnet_leoxiaobin]' "
                "(the extra name may also appear as 'hrnet-leoxiaobin')."
            ) from exc
        raise
    return load_cfg_from_file


def _apply_torch_runtime_settings(cfg: Any) -> None:
    """Apply runtime flags from HRNet config."""
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cuda.matmul.allow_tf32 = True


def load_pose_model(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    *,
    device: str | torch.device | None = None,
    strict: bool = False,
) -> tuple[torch.nn.Module, Any]:
    """Load a Leoxiaobin HRNet model from YAML config and checkpoint.

    Args:
        config_path: Path to HRNet YAML config.
        checkpoint_path: Optional path to checkpoint weights. If omitted, uses cfg.TEST.MODEL_FILE.
        device: Optional torch device override.
        strict: Whether to enforce strict state_dict key matching.

    Returns:
        Tuple of (loaded model, frozen cfg object).
    """
    load_cfg_from_file = _require_config_loader()
    cfg = load_cfg_from_file(config_path)

    if checkpoint_path is not None:
        cfg.defrost()
        cfg.TEST.MODEL_FILE = str(checkpoint_path)
        cfg.freeze()

    if not cfg.TEST.MODEL_FILE:
        raise ValueError(
            "No checkpoint path provided and cfg.TEST.MODEL_FILE is empty. "
            "Set checkpoint_path or populate TEST.MODEL_FILE in config."
        )

    _apply_torch_runtime_settings(cfg)
    resolved_device = _resolve_device(device)

    model = get_pose_net(cfg, is_train=False)
    state_dict = torch.load(cfg.TEST.MODEL_FILE, map_location=resolved_device, weights_only=True)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    model = model.to(resolved_device)
    return model, cfg


def predict_single_pose_from_model_files(
    input_iter: Iterator[np.ndarray],
    *,
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    batch_size: int = 1,
    device: str | torch.device | None = None,
) -> SinglePoseInferenceResult:
    """Run single-pose inference directly from model/config paths."""
    model, _ = load_pose_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return predict_single_pose(
        input_iter=input_iter,
        model=model,
        batch_size=batch_size,
        device=device,
    )
