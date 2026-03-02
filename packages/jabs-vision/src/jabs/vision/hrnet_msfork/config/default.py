"""Default YACS config for Leoxiaobin HRNet models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ""
_C.LOG_DIR = ""
_C.DATA_DIR = ""
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = "pose_hrnet"
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ""
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = "gaussian"
_C.MODEL.IMAGE_SIZE = [256, 256]
_C.MODEL.HEATMAP_SIZE = [64, 64]
_C.MODEL.SIGMA = 2
_C.MODEL.EXP_DECAY_LAMBDA = 0.3
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False
_C.LOSS.POSE_LOSS_FUNC = "MSE"
_C.LOSS.BALANCED_BCE_FAIRNESS_QUOTIENT = 1.0
_C.LOSS.POSITIVE_LABEL_WEIGHT = 1
_C.LOSS.POSE_HEATMAP_WEIGHT = 1 / 2
_C.LOSS.ASSOC_EMBEDDING_WEIGHT = 1 / 1000

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ""
_C.DATASET.CVAT_XML = ""
_C.DATASET.DATASET = "mpii"
_C.DATASET.TRAIN_SET = "train"
_C.DATASET.TEST_SET = "valid"
_C.DATASET.TEST_SET_PROPORTION = 0.1
_C.DATASET.DATA_FORMAT = "jpg"
_C.DATASET.HYBRID_JOINTS_TYPE = ""
_C.DATASET.SELECT_DATA = False

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE = 1.0
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False
_C.DATASET.PROB_RANDOMIZED_OCCLUSION = 0.0
_C.DATASET.MAX_OCCLUSION_SIZE = 100
_C.DATASET.OCCLUSION_OPACITIES = [0.75, 1.0]
_C.DATASET.PROB_RANDOMIZED_CENTER = 0.0
_C.DATASET.JITTER_CENTER = 0.0
_C.DATASET.JITTER_BRIGHTNESS = 0.0
_C.DATASET.JITTER_CONTRAST = 0.0
_C.DATASET.JITTER_SATURATION = 0.0

# train
_C.TRAIN = CN()
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.OPTIMIZER = "adam"
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140
_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ""
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False
_C.TEST.USE_GT_BBOX = False
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ""
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ""

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def get_cfg_defaults() -> CN:
    """Get a clone of default config."""
    return _C.clone()


def update_config(cfg: CN, args: Any) -> None:
    """Update a config object from argparse-like args.

    Args:
        cfg: Config object to update.
        args: argparse-like object with fields used by the legacy HRNet code path.
    """
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(cfg.DATA_DIR, cfg.DATASET.ROOT)
    cfg.MODEL.PRETRAINED = os.path.join(cfg.DATA_DIR, cfg.MODEL.PRETRAINED)

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(cfg.DATA_DIR, cfg.TEST.MODEL_FILE)

    cfg.freeze()


def load_cfg_from_file(config_path: str | Path) -> CN:
    """Load a frozen config from a YAML file."""
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    return cfg


cfg = get_cfg_defaults()
