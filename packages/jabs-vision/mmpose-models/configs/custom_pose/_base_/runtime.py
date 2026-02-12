default_scope = "mmpose"

custom_imports = dict(imports=["simple_pose"], allow_failed_imports=False)

env_cfg = dict(cudnn_benchmark=True)

log_processor = dict(by_epoch=True)

visualizer = dict(
    type="PoseLocalVisualizer",
    vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")],
    name="visualizer",
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=500, interval_exp_name=1000000),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, save_best="PCK", rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

randomness = dict(seed=42, deterministic=False)

launcher = "none"
