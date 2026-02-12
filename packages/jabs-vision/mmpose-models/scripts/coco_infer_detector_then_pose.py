#!/usr/bin/env python3
"""
Run a 2-stage instance pose pipeline over COCO images:
1) MMDetection detector -> bounding boxes per image
2) MMPose top-down pose model -> keypoints per detected instance

Outputs JSON lines to stdout (one record per image).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from mmcv.image import imread
from mmdet.apis import inference_detector, init_detector
from mmengine.registry import DefaultScope
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples


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
    parser = argparse.ArgumentParser(description="Run detector → top-down pose inference on COCO images.")
    parser.add_argument("coco_json", type=Path, help="COCO JSON file containing an images list.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root directory for image file_name entries.",
    )
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
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        required=True,
        help="Output JSON lines path (one record per image).",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on images processed.")
    return parser.parse_args()


def _resolve_image_path(coco_path: Path, image_root: Path | None, file_name: str) -> Path:
    candidate = Path(file_name)
    if candidate.is_absolute():
        return candidate
    base = image_root if image_root is not None else coco_path.parent
    return base / candidate


def _run_detector(det_model, img_bgr: np.ndarray):
    # Temporarily force MMDet scope so PackDetInputs resolves after MMPose sets its own scope.
    with DefaultScope.overwrite_default_scope("mmdet"):
        return inference_detector(det_model, img_bgr)


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


def _image_to_json(
    image_id: int,
    file_name: str,
    frame_shape: Tuple[int, int],
    det_bboxes: np.ndarray,
    det_scores: np.ndarray,
    pose_sample,
) -> str:
    instances = []
    if pose_sample is not None:
        pred_instances = getattr(pose_sample, "pred_instances", None)
        keypoints = _to_numpy(getattr(pred_instances, "keypoints", None))
        kpt_scores = _to_numpy(getattr(pred_instances, "keypoint_scores", None))
        num = min(det_bboxes.shape[0], keypoints.shape[0] if keypoints.size else det_bboxes.shape[0])
        for i in range(num):
            item = {
                "bbox": det_bboxes[i].tolist(),
                "bbox_score": float(det_scores[i]) if det_scores.size > i else float("nan"),
            }
            if keypoints.size:
                item["keypoints"] = keypoints[i].tolist()
            if kpt_scores.size:
                item["keypoint_scores"] = kpt_scores[i].tolist()
            instances.append(item)

    record = {
        "image_id": int(image_id),
        "file_name": file_name,
        "height": int(frame_shape[0]),
        "width": int(frame_shape[1]),
        "instances": instances,
    }
    return json.dumps(record, separators=(",", ":"))


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
    if not args.coco_json.exists():
        raise FileNotFoundError(f"COCO JSON not found: {args.coco_json}")

    with args.coco_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = payload.get("images") or []
    if not isinstance(images, list):
        raise ValueError("COCO JSON missing `images` list.")

    det_model = init_detector(str(args.det_config), str(args.det_checkpoint), device=args.device)
    pose_model = init_model(str(args.pose_config), str(args.pose_checkpoint), device=args.device)

    max_images = int(args.max_images) if args.max_images is not None else None

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as out_handle:
        with torch.no_grad():
            for idx, image in enumerate(images):
                if max_images is not None and idx >= max_images:
                    break
                file_name = str(image.get("file_name") or "")
                image_id = int(image.get("id") or (idx + 1))
                img_path = _resolve_image_path(args.coco_json, args.image_root, file_name)
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")

                img_bgr = imread(str(img_path))
                if img_bgr is None:
                    raise RuntimeError(f"Failed to load image: {img_path}")

                img_h, img_w = int(img_bgr.shape[0]), int(img_bgr.shape[1])

                det_sample = _run_detector(det_model, img_bgr)
                det_bboxes, det_scores = _extract_det_results(
                    det_sample,
                    (img_h, img_w),
                    args.det_score_thr,
                    args.max_instances,
                    args.bbox_padding,
                    allow_empty=True,
                )

                pose_sample = None
                if det_bboxes.shape[0] > 0:
                    try:
                        pose_batch = inference_topdown(pose_model, img_bgr, bboxes=det_bboxes, bbox_format="xyxy")
                    except TypeError as exc:
                        if "len() of unsized object" not in str(exc):
                            raise
                        person_results = [
                            {"bbox": det_bboxes[j], "bbox_score": float(det_scores[j])}
                            for j in range(det_bboxes.shape[0])
                        ]
                        pose_batch = inference_topdown(pose_model, img_bgr, bboxes=person_results, bbox_format="xyxy")
                    pose_sample = merge_data_samples(pose_batch)

                line = _image_to_json(image_id, file_name, (img_h, img_w), det_bboxes, det_scores, pose_sample)
                out_handle.write(line + "\n")


if __name__ == "__main__":
    main()
