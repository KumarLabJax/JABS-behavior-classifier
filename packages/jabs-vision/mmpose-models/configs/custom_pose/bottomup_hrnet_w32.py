from __future__ import annotations

import copy
import pathlib

_base_ = ["./_base_/runtime.py"]

# MMPose bottom-up keypoint config: predicts all instances/keypoints directly
# from the image, so it does not depend on an MMDetection detector stage.
max_epochs = 140
val_interval = 5
# input_size = (512, 512)
input_size = (1024, 1024) # TODO: starting out with higher res because mice are small in the images. investigate later if this can be lowered.
heatmap_size = (input_size[0] // 4, input_size[1] // 4)
num_keypoints = 5

train_cfg = {"type": "EpochBasedTrainLoop", "max_epochs": max_epochs, "val_interval": val_interval}
val_cfg = {"type": "ValLoop"}
test_cfg = {"type": "TestLoop"}

optim_wrapper = {"type": "OptimWrapper", "optimizer": {"type": "Adam", "lr": 1e-3}}
param_scheduler = [
    {"type": "LinearLR", "begin": 0, "end": 500, "start_factor": 0.001, "by_epoch": False},
    {
        "type": "MultiStepLR",
        "begin": 0,
        "end": max_epochs,
        "milestones": [int(max_epochs * 0.65), int(max_epochs * 0.85)],
        "by_epoch": True,
        "gamma": 0.1,
    },
]

codec = {
    "type": "AssociativeEmbedding",
    "input_size": input_size,
    "heatmap_size": heatmap_size,
    "sigma": 2.0,
    "decode_topk": 20,
    "decode_max_instances": 10,
    "decode_center_shift": 0.5,
}

# Bottom-up model stack using HRNet + associative embedding decoding.
model = {
    "type": "BottomupPoseEstimator",
    "data_preprocessor": {
        "type": "PoseDataPreprocessor",
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "bgr_to_rgb": True,
    },
    "backbone": {
        "type": "HRNet",
        "in_channels": 3,
        "extra": {
            "stage1": {"num_modules": 1, "num_branches": 1, "block": "BOTTLENECK", "num_blocks": (4,), "num_channels": (64,)},
            "stage2": {"num_modules": 1, "num_branches": 2, "block": "BASIC", "num_blocks": (4, 4), "num_channels": (32, 64)},
            "stage3": {
                "num_modules": 4,
                "num_branches": 3,
                "block": "BASIC",
                "num_blocks": (4, 4, 4),
                "num_channels": (32, 64, 128),
            },
            "stage4": {
                "num_modules": 3,
                "num_branches": 4,
                "block": "BASIC",
                "num_blocks": (4, 4, 4, 4),
                "num_channels": (32, 64, 128, 256),
                "multiscale_output": True,
            },
        },
        "init_cfg": {
            "type": "Pretrained",
            "checkpoint": "https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth",
        },
    },
    "neck": {"type": "FeatureMapProcessor", "concat": True},
    "head": {
        "type": "AssociativeEmbeddingHeadNoKptWeight",
        "in_channels": 480,
        "num_keypoints": num_keypoints,
        "tag_dim": 1,
        "tag_per_keypoint": True,
        "deconv_out_channels": None,
        "keypoint_loss": {"type": "KeypointMSELoss", "use_target_weight": False},
        "tag_loss": {"type": "AssociativeEmbeddingLoss", "loss_weight": 1e-3},
        "decoder": dict(codec, heatmap_size=codec["input_size"]),
    },
    "train_cfg": {"max_train_instances": 50},
    "test_cfg": {
        "multiscale_test": False,
        "flip_test": True,
        "shift_heatmap": False,
        "restore_heatmap_size": True,
        "align_corners": False,
    },
}

# dataset_type = "JsonKeypointDataset"
dataset_type = "CocoDataset"
data_mode = "bottomup"
# COCO-style keypoint annotations exported from our internal dataset cache.
data_root = str(pathlib.Path.home() / "datasets/hydra-label-cache")
# train_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_train.json"
# val_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_val.json"
# test_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_test.json"
train_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_train_coco.json"
val_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_val_coco.json"
test_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_test_coco.json"

coco_val_ann_file = str(pathlib.Path(data_root) / val_ann_file)
coco_test_ann_file = str(pathlib.Path(data_root) / test_ann_file)

train_pipeline = [
    {"type": "LoadImage"},
    {
        "type": "BottomupRandomAffine",
        "input_size": input_size,
        "scale_factor": [0.7, 1.35],
        "shift_factor": 0.2,
        "rotate_factor": 40,
    },
    {"type": "RandomFlip", "direction": "horizontal", "prob": 0.5},
    {"type": "GenerateTarget", "encoder": codec},
    {"type": "BottomupGetHeatmapMask"},
    {"type": "PackPoseInputsWithAE"},
]
val_pipeline = [
    {"type": "LoadImage"},
    {"type": "BottomupResize", "input_size": input_size, "size_factor": 64, "resize_mode": "expand"},
    {
        "type": "PackPoseInputsWithAE",
        "meta_keys": (
            "id",
            "img_id",
            "img_path",
            "crowd_index",
            "ori_shape",
            "img_shape",
            "input_size",
            "input_center",
            "input_scale",
            "flip",
            "flip_direction",
            "flip_indices",
            "raw_ann_info",
            "skeleton_links",
        ),
    },
]
test_pipeline = copy.deepcopy(val_pipeline)

raw_meta = {
    "dataset_name": "custom_mouse_pose",
    "keypoints": [
        {"name": "tip of nose", "type": "upper", "color": [255, 85, 0]},
        {"name": "left ear", "type": "upper", "swap": "right ear", "color": [0, 255, 0]},
        {"name": "right ear", "type": "upper", "swap": "left ear", "color": [0, 255, 0]},
        {"name": "base of tail", "type": "lower", "color": [0, 0, 255]},
        {"name": "tip of tail", "type": "lower", "color": [255, 255, 0]},
    ],
    "skeleton": [
        {"link": ["left ear", "tip of nose"], "color": [255, 255, 0]},
        {"link": ["right ear", "tip of nose"], "color": [0, 255, 255]},
        {"link": ["base of tail", "tip of nose"], "color": [255, 255, 255]},
        {"link": ["tip of tail", "base of tail"], "color": [255, 128, 0]},
    ],
    "joint_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
    "sigmas": [0.05, 0.05, 0.05, 0.05, 0.05],
}

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
    "batch_size": 12,
    "num_workers": 12,
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
    "batch_size": 1,
    "num_workers": 4,
    "persistent_workers": False,
    "pin_memory": True,
    "sampler": {"type": "DefaultSampler", "shuffle": False, "round_up": False},
    "dataset": {
        "type": dataset_type,
        "metainfo": copy.deepcopy(metainfo),
        "data_root": data_root,
        "data_mode": data_mode,
        "ann_file": val_ann_file,
        "data_prefix": {"img": ""},
        "test_mode": True,
        "pipeline": val_pipeline,
        # raw_meta=copy.deepcopy(raw_meta),
    },
}
test_dataloader = {
    "batch_size": 1,
    "num_workers": 4,
    "persistent_workers": False,
    "pin_memory": True,
    "sampler": {"type": "DefaultSampler", "shuffle": False, "round_up": False},
    "dataset": {
        "type": dataset_type,
        "metainfo": copy.deepcopy(metainfo),
        "data_root": data_root,
        "data_mode": data_mode,
        "ann_file": test_ann_file,
        "data_prefix": {"img": ""},
        "test_mode": True,
        "pipeline": test_pipeline,
        # raw_meta=copy.deepcopy(raw_meta),
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

work_dir = "runs/bottomup_hrnet_w32"
