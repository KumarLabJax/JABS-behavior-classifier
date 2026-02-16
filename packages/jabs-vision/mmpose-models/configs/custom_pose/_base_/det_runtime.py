# Base runtime settings for detector configs (MMDetection scope).
#
# In a top-down pose pipeline, MMPose typically relies on an object detector
# (from MMDetection) to produce bounding boxes before running the pose model.
# When training/evaluating that detector, these are the shared MMEngine runtime
# defaults used by our custom MMDet configs in this repo.
default_scope = "mmdet"

# Register our local project/module so MMDet can find any custom components
# referenced by config (datasets, transforms, models, hooks, etc.).
custom_imports = {"imports": ["simple_pose"], "allow_failed_imports": False}

# Runtime environment flags used by MMEngine/MMDet.
env_cfg = {"cudnn_benchmark": True}

# Controls whether logs/metrics are aggregated by epoch or by iteration.
log_processor = {"by_epoch": True}

# Visualization backend used by MMDetection (detections/boxes/masks).
visualizer = {
    "type": "DetLocalVisualizer",
    "vis_backends": [{"type": "LocalVisBackend"}, {"type": "TensorboardVisBackend"}],
    "name": "visualizer",
}

# Standard MMEngine hooks. `save_best` matches the metric key produced by the
# MMDet evaluator (here a single-class AP at IoU=0.50).
default_hooks = {
    "timer": {"type": "IterTimerHook"},
    "logger": {"type": "LoggerHook", "interval": 500, "interval_exp_name": 1000000},
    "param_scheduler": {"type": "ParamSchedulerHook"},
    "checkpoint": {"type": "CheckpointHook", "interval": 1, "save_best": "single_class_ap/AP@0.50", "rule": "greater"},
    "sampler_seed": {"type": "DistSamplerSeedHook"},
}

# Training determinism controls (seed + whether to force deterministic ops).
randomness = {"seed": 42, "deterministic": False}

# Distributed launcher selection. "none" means single-process by default.
launcher = "none"
