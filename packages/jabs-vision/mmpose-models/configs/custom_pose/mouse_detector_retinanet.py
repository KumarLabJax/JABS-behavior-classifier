from __future__ import annotations

import copy
import pathlib

_base_ = ["./_base_/det_runtime.py"]

num_classes = 1
data_root = str(pathlib.Path.home() / "datasets/hydra-label-cache")
# train_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_train.json"
# val_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_val.json"
# test_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_test.json"
train_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_train_coco.json"
val_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_val_coco.json"
test_ann_file = "dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_test_coco.json"

metainfo = dict(classes=("mouse",), palette=[(255, 0, 0)])

model = dict(
    type="RetinaNet",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5,
    ),
    bbox_head=dict(
        type="RetinaHead",
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    train_cfg=dict(
        assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0.0, ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=100,
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs"),
]

# dataset_type = "JsonKeypointDataset"
dataset_type = "CocoDataset"

raw_meta = dict(
    dataset_name="custom_mouse_pose",
    keypoints=[
        {"name": "tip of nose", "type": "upper", "color": [255, 85, 0]},
        {"name": "left ear", "type": "upper", "swap": "right ear", "color": [0, 255, 0]},
        {"name": "right ear", "type": "upper", "swap": "left ear", "color": [0, 255, 0]},
        {"name": "base of tail", "type": "lower", "color": [0, 0, 255]},
        {"name": "tip of tail", "type": "lower", "color": [255, 255, 0]},
    ],
    skeleton=[
        {"link": ["left ear", "tip of nose"], "color": [255, 255, 0]},
        {"link": ["right ear", "tip of nose"], "color": [0, 255, 255]},
        {"link": ["base of tail", "tip of nose"], "color": [255, 255, 255]},
        {"link": ["tip of tail", "base of tail"], "color": [255, 128, 0]},
    ],
    joint_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
    sigmas=[0.05, 0.05, 0.05, 0.05, 0.05],
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        # metainfo=metainfo,
        metainfo=copy.deepcopy(metainfo),
        data_root=data_root,
        # data_mode="topdown",
        ann_file=train_ann_file,
        data_prefix=dict(img=""),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
        # raw_meta=copy.deepcopy(raw_meta),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        # metainfo=metainfo,
        metainfo=copy.deepcopy(metainfo),
        data_root=data_root,
        # data_mode="topdown",
        ann_file=val_ann_file,
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        # raw_meta=copy.deepcopy(raw_meta),
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        # metainfo=metainfo,
        metainfo=copy.deepcopy(metainfo),
        data_root=data_root,
        # data_mode="topdown",
        ann_file=test_ann_file,
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        # raw_meta=copy.deepcopy(raw_meta),
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.0025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

max_epochs = 100
val_interval = 2

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[int(max_epochs * 0.67), int(max_epochs * 0.92)],
        gamma=0.1,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

val_evaluator = dict(type="SingleClassAPMetric", iou_thr=0.75)

test_evaluator = dict(type="SingleClassAPMetric", iou_thr=0.75)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, save_best="single_class_ap/AP@0.75", rule="greater"),
)

auto_scale_lr = dict(base_batch_size=16, enable=True)

work_dir = "runs/mouse_detector_retinanet"
