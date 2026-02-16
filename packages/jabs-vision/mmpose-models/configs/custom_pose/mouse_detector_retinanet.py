from __future__ import annotations

import copy
import pathlib

_base_ = ["./_base_/det_runtime.py"]

num_classes = 1
data_root = str(pathlib.Path.home() / "datasets/hydra-label-cache")
train_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_train_coco.json"
val_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_val_coco.json"
test_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_test_coco.json"

metainfo = {"classes": ("mouse",), "palette": [(255, 0, 0)]}

model = {
    "type": "RetinaNet",
    "data_preprocessor": {
        "type": "DetDataPreprocessor",
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "bgr_to_rgb": True,
        "pad_size_divisor": 32,
    },
    "backbone": {
        "type": "ResNet",
        "depth": 50,
        "num_stages": 4,
        "out_indices": (0, 1, 2, 3),
        "frozen_stages": 1,
        "norm_cfg": {"type": "BN", "requires_grad": True},
        "norm_eval": True,
        "style": "pytorch",
        "init_cfg": {"type": "Pretrained", "checkpoint": "torchvision://resnet50"},
    },
    "neck": {
        "type": "FPN",
        "in_channels": [256, 512, 1024, 2048],
        "out_channels": 256,
        "start_level": 1,
        "add_extra_convs": "on_input",
        "num_outs": 5,
    },
    "bbox_head": {
        "type": "RetinaHead",
        "num_classes": num_classes,
        "in_channels": 256,
        "stacked_convs": 4,
        "feat_channels": 256,
        "anchor_generator": {
            "type": "AnchorGenerator",
            "octave_base_scale": 4,
            "scales_per_octave": 3,
            "ratios": [0.5, 1.0, 2.0],
            "strides": [8, 16, 32, 64, 128],
        },
        "bbox_coder": {"type": "DeltaXYWHBBoxCoder", "target_means": [0.0, 0.0, 0.0, 0.0], "target_stds": [0.1, 0.1, 0.2, 0.2]},
        "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 1.0},
        "loss_bbox": {"type": "L1Loss", "loss_weight": 1.0},
    },
    "train_cfg": {
        "assigner": {"type": "MaxIoUAssigner", "pos_iou_thr": 0.5, "neg_iou_thr": 0.4, "min_pos_iou": 0.0, "ignore_iof_thr": -1},
        "allowed_border": -1,
        "pos_weight": -1,
        "debug": False,
    },
    "test_cfg": {
        "nms_pre": 1000,
        "min_bbox_size": 0,
        "score_thr": 0.05,
        "nms": {"type": "nms", "iou_threshold": 0.5},
        "max_per_img": 100,
    },
}

train_pipeline = [
    {"type": "LoadImageFromFile"},
    {"type": "LoadAnnotations", "with_bbox": True},
    {"type": "Resize", "scale": (1333, 800), "keep_ratio": True},
    {"type": "RandomFlip", "prob": 0.5},
    {"type": "PackDetInputs"},
]

test_pipeline = [
    {"type": "LoadImageFromFile"},
    {"type": "Resize", "scale": (1333, 800), "keep_ratio": True},
    {"type": "LoadAnnotations", "with_bbox": True},
    {"type": "PackDetInputs"},
]

dataset_type = "CocoDataset"

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

train_dataloader = {
    "batch_size": 2,
    "num_workers": 2,
    "persistent_workers": True,
    "sampler": {"type": "DefaultSampler", "shuffle": True},
    "batch_sampler": {"type": "AspectRatioBatchSampler"},
    "dataset": {
        "type": dataset_type,
        # metainfo=metainfo,
        "metainfo": copy.deepcopy(metainfo),
        "data_root": data_root,
        # data_mode="topdown",
        "ann_file": train_ann_file,
        "data_prefix": {"img": ""},
        "filter_cfg": {"filter_empty_gt": True, "min_size": 1},
        "pipeline": train_pipeline,
        # raw_meta=copy.deepcopy(raw_meta),
    },
}

val_dataloader = {
    "batch_size": 1,
    "num_workers": 2,
    "persistent_workers": False,
    "sampler": {"type": "DefaultSampler", "shuffle": False},
    "dataset": {
        "type": dataset_type,
        # metainfo=metainfo,
        "metainfo": copy.deepcopy(metainfo),
        "data_root": data_root,
        # data_mode="topdown",
        "ann_file": val_ann_file,
        "data_prefix": {"img": ""},
        "test_mode": True,
        "pipeline": test_pipeline,
        # raw_meta=copy.deepcopy(raw_meta),
    },
}

test_dataloader = {
    "batch_size": 1,
    "num_workers": 2,
    "persistent_workers": False,
    "sampler": {"type": "DefaultSampler", "shuffle": False},
    "dataset": {
        "type": dataset_type,
        # metainfo=metainfo,
        "metainfo": copy.deepcopy(metainfo),
        "data_root": data_root,
        # data_mode="topdown",
        "ann_file": test_ann_file,
        "data_prefix": {"img": ""},
        "test_mode": True,
        "pipeline": test_pipeline,
        # raw_meta=copy.deepcopy(raw_meta),
    },
}

optim_wrapper = {
    "type": "OptimWrapper",
    "optimizer": {"type": "SGD", "lr": 0.0025, "momentum": 0.9, "weight_decay": 0.0001},
    "clip_grad": {"max_norm": 5.0, "norm_type": 2},
}

max_epochs = 100
val_interval = 2

param_scheduler = [
    {"type": "LinearLR", "start_factor": 0.001, "by_epoch": False, "begin": 0, "end": 500},
    {
        "type": "MultiStepLR",
        "begin": 0,
        "end": max_epochs,
        "by_epoch": True,
        "milestones": [int(max_epochs * 0.67), int(max_epochs * 0.92)],
        "gamma": 0.1,
    },
]

train_cfg = {"type": "EpochBasedTrainLoop", "max_epochs": max_epochs, "val_interval": val_interval}
val_cfg = {"type": "ValLoop"}
test_cfg = {"type": "TestLoop"}

val_evaluator = {"type": "SingleClassAPMetric", "iou_thr": 0.75}

test_evaluator = {"type": "SingleClassAPMetric", "iou_thr": 0.75}

default_hooks = {
    "checkpoint": {"type": "CheckpointHook", "interval": 1, "save_best": "single_class_ap/AP@0.75", "rule": "greater"},
}

auto_scale_lr = {"base_batch_size": 16, "enable": True}

work_dir = "runs/mouse_detector_retinanet"
