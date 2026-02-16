default_scope = "mmdet"

custom_imports = {"imports": ["simple_pose"], "allow_failed_imports": False}

env_cfg = {"cudnn_benchmark": True}

log_processor = {"by_epoch": True}

visualizer = {
    "type": "DetLocalVisualizer",
    "vis_backends": [{"type": "LocalVisBackend"}, {"type": "TensorboardVisBackend"}],
    "name": "visualizer",
}

default_hooks = {
    "timer": {"type": "IterTimerHook"},
    "logger": {"type": "LoggerHook", "interval": 500, "interval_exp_name": 1000000},
    "param_scheduler": {"type": "ParamSchedulerHook"},
    "checkpoint": {"type": "CheckpointHook", "interval": 1, "save_best": "single_class_ap/AP@0.50", "rule": "greater"},
    "sampler_seed": {"type": "DistSamplerSeedHook"},
}

randomness = {"seed": 42, "deterministic": False}

launcher = "none"
