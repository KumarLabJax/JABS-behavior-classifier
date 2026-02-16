# Base runtime settings for our MMPose configs.
#
# These values are consumed by MMEngine's Runner (used by MMPose) and typically
# get pulled in via `_base_ = [...]` from the model configs in this repo.
#
# Note: this file is the *pose* runtime (MMPose scope). For the detector runtime
# used by MMPose top-down pipelines, see `_base_/det_runtime.py` (MMDet scope).
default_scope = "mmpose"

# Register our local project/module so MMPose can find custom dataset/model
# components (via MMEngine registries) when building from config.
custom_imports = {"imports": ["simple_pose"], "allow_failed_imports": False}

# Runtime environment flags used by MMEngine/MMPose. `cudnn_benchmark` can speed
# up training/inference for fixed input sizes.
env_cfg = {"cudnn_benchmark": True}

# Controls whether logs/metrics are aggregated by epoch or by iteration.
log_processor = {"by_epoch": True}

# Visualization backend used by MMPose (keypoints/poses). Tensorboard backend is
# useful during training; Local backend can save images locally when enabled.
visualizer = {
    "type": "PoseLocalVisualizer",
    "vis_backends": [{"type": "LocalVisBackend"}, {"type": "TensorboardVisBackend"}],
    "name": "visualizer",
}

# Standard MMEngine hooks: timing, logging, LR scheduling, checkpointing, etc.
# `save_best` is the metric name reported by MMPose evaluators.
default_hooks = {
    "timer": {"type": "IterTimerHook"},
    "logger": {"type": "LoggerHook", "interval": 500, "interval_exp_name": 1000000},
    "param_scheduler": {"type": "ParamSchedulerHook"},
    "checkpoint": {"type": "CheckpointHook", "interval": 1, "save_best": "PCK", "rule": "greater"},
    "sampler_seed": {"type": "DistSamplerSeedHook"},
}

# Training determinism controls (seed + whether to force deterministic ops).
randomness = {"seed": 42, "deterministic": False}

# Distributed launcher selection. "none" means single-process by default; other
# configs may override this (e.g., "pytorch", "slurm").
launcher = "none"
