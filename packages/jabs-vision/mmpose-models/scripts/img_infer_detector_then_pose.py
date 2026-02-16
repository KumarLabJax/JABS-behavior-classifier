#!/usr/bin/env python3
"""
Run a 2-stage instance pose pipeline:
1) MMDetection detector -> bounding boxes
2) MMPose top-down pose model -> keypoints per detected instance
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from mmcv.image import imread
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

import numpy as np

import torch


def _to_numpy(value) -> np.ndarray:
    if value is None:
        value = []
    else:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()

    return np.asarray(value)


def _pad_and_clip_xyxy(bboxes: np.ndarray, img_shape: Tuple[int, int], padding: float) -> np.ndarray:
    if bboxes.size == 0:
        return bboxes.reshape((0, 4)).astype(np.float32)
    bboxes = bboxes.astype(np.float32)
    height, width = img_shape
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    box_w = np.maximum(x2 - x1, 1.0)
    box_h = np.maximum(y2 - y1, 1.0)
    cx = x1 + 0.5 * box_w
    cy = y1 + 0.5 * box_h
    half_w = 0.5 * box_w * float(padding)
    half_h = 0.5 * box_h * float(padding)
    x1 = np.clip(cx - half_w, 0, width - 1)
    y1 = np.clip(cy - half_h, 0, height - 1)
    x2 = np.clip(cx + half_w, 0, width - 1)
    y2 = np.clip(cy + half_h, 0, height - 1)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector → top-down pose inference on a single image.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--det-config", type=Path, required=True, help="MMDet config file.")
    parser.add_argument("--det-checkpoint", type=Path, required=True, help="MMDet checkpoint file.")
    parser.add_argument("--pose-config", type=Path, required=True, help="MMPose top-down config file.")
    parser.add_argument("--pose-checkpoint", type=Path, required=True, help="MMPose checkpoint file.")
    parser.add_argument("--device", default="cuda:0", help="Device string (e.g. cuda:0, cpu).")
    parser.add_argument("--det-score-thr", type=float, default=0.3, help="Detector score threshold.")
    parser.add_argument("--max-instances", type=int, default=10, help="Max detections to pass to pose model.")
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=1.25,
        help="Multiply detector bbox size by this factor before pose inference.",
    )
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Keypoint score threshold for visualization.")
    parser.add_argument("--out-file", type=Path, default=None, help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Display the rendered image (if supported).")
    return parser.parse_args()


def _extract_det_results(
    det_sample,
    frame_shape: Tuple[int, int],
    det_score_thr: float,
    max_instances: int,
    bbox_padding: float,
    allow_empty: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_instances = getattr(det_sample, "pred_instances", None)
    if pred_instances is None:
        raise RuntimeError("Unexpected MMDet output: missing `pred_instances`.")

    det_bboxes = _to_numpy(getattr(pred_instances, "bboxes", None)).reshape((-1, 4))
    det_scores = _to_numpy(getattr(pred_instances, "scores", None)).reshape((-1,))
    if det_bboxes.shape[0] != det_scores.shape[0]:
        raise RuntimeError(f"Unexpected detector outputs: bboxes={det_bboxes.shape}, scores={det_scores.shape}")

    keep = det_scores >= float(det_score_thr)
    det_bboxes = det_bboxes[keep]
    det_scores = det_scores[keep]
    if det_bboxes.shape[0] == 0:
        if not allow_empty:
            raise RuntimeError(f"No detections above --det-score-thr={det_score_thr}.")
        return det_bboxes.reshape((0, 4)).astype(np.float32), det_scores

    order = np.argsort(-det_scores)
    order = order[: max(int(max_instances), 1)]
    det_bboxes = det_bboxes[order]
    det_scores = det_scores[order]

    det_bboxes = _pad_and_clip_xyxy(det_bboxes, frame_shape, padding=float(bbox_padding))
    det_scores = det_scores.astype(np.float32, copy=False)
    return det_bboxes, det_scores


def main() -> None:
    # PyTorch 2.6 defaults weights_only=True; override for trusted local
    # checkpoints, otherwise loading mmpose model fails
    _orig_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_compat

    args = parse_args()

    img_path = str(args.image)
    img_bgr = imread(img_path)
    img_rgb = img_bgr[..., ::-1].copy()
    img_h, img_w = int(img_bgr.shape[0]), int(img_bgr.shape[1])

    det_model = init_detector(str(args.det_config), str(args.det_checkpoint), device=args.device)
    det_sample = inference_detector(det_model, img_bgr)
    det_bboxes, det_scores = _extract_det_results(
        det_sample,
        (img_h, img_w),
        args.det_score_thr,
        args.max_instances,
        args.bbox_padding,
        allow_empty=False,
    )

    pose_model = init_model(str(args.pose_config), str(args.pose_checkpoint), device=args.device)
    try:
        pose_batch = inference_topdown(pose_model, img_bgr, bboxes=det_bboxes, bbox_format="xyxy")
    except TypeError as exc:
        if "len() of unsized object" not in str(exc):
            raise
        person_results = [
            {"bbox": det_bboxes[i], "bbox_score": float(det_scores[i])} for i in range(det_bboxes.shape[0])
        ]
        pose_batch = inference_topdown(pose_model, img_bgr, bboxes=person_results, bbox_format="xyxy")

    results = merge_data_samples(pose_batch)

    visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
    visualizer.set_dataset_meta(pose_model.dataset_meta)

    out_file = str(args.out_file) if args.out_file is not None else None
    visualizer.add_datasample(
        "result",
        img_rgb,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=float(args.kpt_thr),
        show=bool(args.show),
        out_file=out_file,
    )

    if out_file:
        print(out_file)


if __name__ == "__main__":
    main()
