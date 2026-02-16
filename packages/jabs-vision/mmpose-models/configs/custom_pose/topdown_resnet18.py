from __future__ import annotations

import copy
import pathlib

_base_ = ["./_base_/runtime.py"]

# MMPose top-down keypoint config: assumes person/animal boxes are available
# (typically from an MMDetection model trained via configs in this same folder).
max_epochs = 200
val_interval = 10
base_lr = 5e-4
input_size = (256, 256)
heatmap_size = (input_size[0] // 4, input_size[1] // 4)
num_keypoints = 5  # will be overridden by script if pose meta provides a different spec

train_cfg = {"type": "EpochBasedTrainLoop", "max_epochs": max_epochs, "val_interval": val_interval}
val_cfg = {"type": "ValLoop"}
test_cfg = {"type": "TestLoop"}

optim_wrapper = {"type": "OptimWrapper", "optimizer": {"type": "Adam", "lr": base_lr}}
param_scheduler = [
    {"type": "LinearLR", "begin": 0, "end": 500, "start_factor": 0.001, "by_epoch": False},
    {
        "type": "MultiStepLR",
        "begin": 0,
        "end": max_epochs,
        "milestones": [int(max_epochs * 0.6), int(max_epochs * 0.85)],
        "by_epoch": True,
        "gamma": 0.1,
    },
]

codec = {"type": "MSRAHeatmap", "input_size": input_size, "heatmap_size": heatmap_size, "sigma": 2}

# Pose model stack (backbone + heatmap head) built through MMPose registries.
model = {
    "type": "TopdownPoseEstimator",
    "data_preprocessor": {
        "type": "PoseDataPreprocessor",
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "bgr_to_rgb": True,
    },
    "backbone": {
        "type": "ResNet",
        "depth": 18,
        "num_stages": 4,
        "out_indices": (3,),
        "frozen_stages": -1,
        "norm_cfg": {"type": "BN", "requires_grad": True},
        "norm_eval": False,
        "style": "pytorch",
    },
    "head": {
        "type": "HeatmapHead",
        "in_channels": 512,
        "out_channels": num_keypoints,
        "deconv_out_channels": (256, 256, 256),
        "deconv_kernel_sizes": (4, 4, 4),
        "loss": {"type": "KeypointMSELoss", "use_target_weight": True},
        "decoder": {
            "type": "MSRAHeatmap",
            "input_size": input_size,
            "heatmap_size": heatmap_size,
            "sigma": 2.0,
        },
    },
    "test_cfg": {"flip_test": True, "flip_mode": "heatmap", "shift_heatmap": True},
}

dataset_type = "CocoDataset"
data_mode = "topdown"
# COCO-style keypoint annotations exported from our internal dataset cache.
data_root = str(pathlib.Path.home() / "datasets/hydra-label-cache")
train_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_train_coco.json"
val_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_val_coco.json"
test_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_test_coco.json"
pose_meta_file = None

coco_val_ann_file = str(pathlib.Path(data_root) / val_ann_file)
coco_test_ann_file = str(pathlib.Path(data_root) / test_ann_file)

train_pipeline = [
    {"type": "LoadImage"},
    {"type": "GetBBoxCenterScale", "padding": 1.25},
    {
        "type": "RandomFlip",
        "direction": ["horizontal", "vertical"],
        "prob": [0.25, 0.25],
    },
    {
        "type": "RandomBBoxTransform",
        "shift_factor": 0.2,
        "shift_prob": 0.25,
        "scale_factor": (0.7, 1.3),
        "scale_prob": 0.25,
        "rotate_factor": 45.0,
        "rotate_prob": 0.25,
    },
    {"type": "TopdownAffine", "input_size": input_size},
    {
        "type": "PhotometricDistortion",
        "brightness_delta": 32,
        "contrast_range": (0.5, 1.5),
        "saturation_range": (0.5, 1.5),
        "hue_delta": 18,
    },
    {"type": "GenerateTarget", "encoder": codec},
    {"type": "PackPoseInputs", "pack_transformed": False},
]

# Val/test use deterministic preprocessing only.
test_pipeline = [
    {"type": "LoadImage"},
    {"type": "GetBBoxCenterScale", "padding": 1.25},
    {"type": "TopdownAffine", "input_size": input_size},
    {"type": "PackPoseInputs", "pack_transformed": False},
]

val_pipeline = copy.deepcopy(test_pipeline)

# Dataset metainfo consumed by MMPose for keypoint ordering, flips, and skeleton.
metainfo = {
    "dataset_name": "custom_mouse_pose",
    "keypoint_info": {
        0: {
            "id": 0,
            "name": "tip of nose",
            "type": "upper",
            "swap": "",
            "color": [255, 85, 0],
        },
        1: {
            "id": 1,
            "name": "left ear",
            "type": "upper",
            "swap": "right ear",
            "color": [0, 255, 0],
        },
        2: {
            "id": 2,
            "name": "right ear",
            "type": "upper",
            "swap": "left ear",
            "color": [0, 255, 0],
        },
        3: {
            "id": 3,
            "name": "base of tail",
            "type": "lower",
            "swap": "",
            "color": [0, 0, 255],
        },
        4: {
            "id": 4,
            "name": "tip of tail",
            "type": "lower",
            "swap": "",
            "color": [255, 255, 0],
        },
    },
    "skeleton_info": {
        0: {"id": 0, "link": ("left ear", "tip of nose"), "color": [255, 255, 0]},
        1: {"id": 1, "link": ("right ear", "tip of nose"), "color": [0, 255, 255]},
        2: {"id": 2, "link": ("base of tail", "tip of nose"), "color": [255, 255, 255]},
        3: {"id": 3, "link": ("tip of tail", "base of tail"), "color": [255, 128, 0]},
    },
    "joint_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
    "sigmas": [0.05, 0.05, 0.05, 0.05, 0.05],
}

train_dataloader = {
    "batch_size": 16,
    "num_workers": 8,
    "persistent_workers": True,
    "pin_memory": True,
    "prefetch_factor": 4,
    "sampler": {"type": "DefaultSampler", "shuffle": True},
    "dataset": {
        "type": dataset_type,
        "metainfo": copy.deepcopy(metainfo),
        "data_root": data_root,
        "data_mode": data_mode,
        "ann_file": train_ann_file,
        "data_prefix": {"img": ""},
        "pipeline": train_pipeline,
        # raw_meta=copy.deepcopy(raw_meta),
    },
}

val_dataloader = {
    "batch_size": 8,
    "num_workers": 4,
    "persistent_workers": False,
    "pin_memory": True,
    "sampler": {"type": "DefaultSampler", "shuffle": False},
    "dataset": {
        "type": dataset_type,
        "metainfo": copy.deepcopy(metainfo),
        "data_root": data_root,
        "data_mode": data_mode,
        "ann_file": val_ann_file,
        "data_prefix": {"img": ""},
        "pipeline": val_pipeline,
        # raw_meta=copy.deepcopy(raw_meta),
        "test_mode": True,
    },
}

test_dataloader = {
    "batch_size": 8,
    "num_workers": 4,
    "persistent_workers": False,
    "pin_memory": True,
    "sampler": {"type": "DefaultSampler", "shuffle": False},
    "dataset": {
        "type": dataset_type,
        "metainfo": copy.deepcopy(metainfo),
        "data_root": data_root,
        "data_mode": data_mode,
        "ann_file": test_ann_file,
        "data_prefix": {"img": ""},
        "pipeline": test_pipeline,
        # raw_meta=copy.deepcopy(raw_meta),
        "test_mode": True,
    },
}

# Track both keypoint localization quality and COCO-style aggregate metrics.
val_evaluator = [
    {"type": "PCKAccuracy", "thr": 0.05, "norm_item": "bbox"},
    {"type": "CocoMetric", "ann_file": coco_val_ann_file},
]
test_evaluator = [
    {"type": "PCKAccuracy", "thr": 0.05, "norm_item": "bbox"},
    {"type": "CocoMetric", "ann_file": coco_test_ann_file},
]

work_dir = "runs/topdown_resnet18"
