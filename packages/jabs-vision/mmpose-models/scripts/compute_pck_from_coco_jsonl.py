#!/usr/bin/env python3
"""
Compute PCK from coco_infer_detector_then_pose.py JSONL output using COCO GT.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PCK for JSONL predictions against COCO GT.")
    parser.add_argument("pred_jsonl", type=Path, help="JSON lines from coco_infer_detector_then_pose.py.")
    parser.add_argument("coco_json", type=Path, help="COCO keypoint annotation file.")
    parser.add_argument("--pck-thr", type=float, default=0.05, help="PCK threshold (normalized distance).")
    parser.add_argument(
        "--norm",
        choices=("max", "sqrt_area"),
        default="max",
        help="Normalization for PCK (max side or sqrt bbox area).",
    )
    parser.add_argument(
        "--visible-mode",
        choices=("gt", "pred"),
        default="gt",
        help=(
            "Visibility denominator: gt = count all GT-visible points and treat unmatched GT instances as all-incorrect; "
            "pred = only points with predictions (unmatched GT instances are ignored)."
        ),
    )
    parser.add_argument(
        "--exclude-points",
        type=int,
        nargs="+",
        default=[],
        help="keypoint indexes to exclude from PCK calculation.",
    )
    parser.add_argument("--min-iou", type=float, default=0.1, help="Min IoU to match pred/gt instances.")
    return parser.parse_args()


def _xywh_to_xyxy(bbox: Iterable[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _xyxy_area(bbox: np.ndarray) -> float:
    w = max(float(bbox[2] - bbox[0]), 0.0)
    h = max(float(bbox[3] - bbox[1]), 0.0)
    return w * h


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(x2 - x1, 0.0)
    inter_h = max(y2 - y1, 0.0)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = _xyxy_area(a) + _xyxy_area(b) - inter
    return inter / union if union > 0 else 0.0


def _load_coco_gt(path: Path) -> Dict[int, List[dict]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    annotations = payload.get("annotations") or []
    by_image: Dict[int, List[dict]] = {}
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        keypoints = ann.get("keypoints")
        if not isinstance(keypoints, list) or len(keypoints) < 3:
            continue
        if "bbox" not in ann:
            continue
        by_image.setdefault(int(image_id), []).append(ann)
    return by_image


def _load_pred_jsonl(path: Path) -> Dict[int, List[dict]]:
    by_image: Dict[int, List[dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = record.get("image_id")
            if image_id is None:
                continue
            instances = record.get("instances") or []
            by_image.setdefault(int(image_id), []).extend(instances)
    return by_image


def _extract_pred_keypoints(instance: dict, num_kpts: int) -> List[Tuple[float, float] | None]:
    keypoints = instance.get("keypoints") or []
    points: List[Tuple[float, float] | None] = []
    for idx in range(num_kpts):
        if idx >= len(keypoints):
            points.append(None)
            continue
        kp = keypoints[idx]
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            points.append(None)
            continue
        x_val = float(kp[0])
        y_val = float(kp[1])
        if math.isnan(x_val) or math.isnan(y_val):
            points.append(None)
            continue
        points.append((x_val, y_val))
    return points


def _extract_gt_keypoints(ann: dict) -> Tuple[List[Tuple[float, float] | None], List[int]]:
    raw = ann.get("keypoints") or []
    points: List[Tuple[float, float] | None] = []
    vis: List[int] = []
    for i in range(0, len(raw), 3):
        x_val = float(raw[i])
        y_val = float(raw[i + 1])
        v_val = int(raw[i + 2])
        vis.append(v_val)
        if v_val <= 0 or math.isnan(x_val) or math.isnan(y_val):
            points.append(None)
        else:
            points.append((x_val, y_val))
    return points, vis


def _match_instances(
    gt_instances: List[dict],
    pred_instances: List[dict],
    min_iou: float,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[float, int, int]] = []
    gt_boxes = []
    pred_boxes = []
    for gt in gt_instances:
        gt_boxes.append(_xywh_to_xyxy(gt["bbox"]))
    for pred in pred_instances:
        if "bbox" not in pred:
            pred_boxes.append(None)
        else:
            pred_boxes.append(np.asarray(pred["bbox"], dtype=np.float32))

    for gi, gt_box in enumerate(gt_boxes):
        for pi, pred_box in enumerate(pred_boxes):
            if pred_box is None:
                continue
            iou = _iou(gt_box, pred_box)
            if iou >= min_iou:
                pairs.append((iou, gi, pi))

    pairs.sort(reverse=True, key=lambda t: t[0])
    matched_gt = set()
    matched_pred = set()
    matches: List[Tuple[int, int]] = []
    for _, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi))
    return matches


def _norm_factor(bbox_xywh: Iterable[float], norm: str) -> float:
    x, y, w, h = [float(v) for v in bbox_xywh]
    if norm == "sqrt_area":
        return math.sqrt(max(w * h, 1e-6))
    return max(w, h, 1e-6)


def main() -> None:
    args = parse_args()
    if not args.pred_jsonl.exists():
        raise FileNotFoundError(f"Pred JSONL not found: {args.pred_jsonl}")
    if not args.coco_json.exists():
        raise FileNotFoundError(f"COCO JSON not found: {args.coco_json}")

    gt_by_image = _load_coco_gt(args.coco_json)
    pred_by_image = _load_pred_jsonl(args.pred_jsonl)

    total_visible = 0
    total_correct = 0
    per_kpt_visible: List[int] = []
    per_kpt_correct: List[int] = []
    matched_instances = 0
    exclude_set = set(args.exclude_points)

    for image_id, gt_instances in gt_by_image.items():
        pred_instances = pred_by_image.get(image_id, [])
        if not gt_instances or not pred_instances:
            continue

        matches = _match_instances(gt_instances, pred_instances, args.min_iou)
        matched_gt = {gi for gi, _ in matches}
        for gi, pi in matches:
            gt = gt_instances[gi]
            pred = pred_instances[pi]
            gt_points, gt_vis = _extract_gt_keypoints(gt)
            num_kpts = len(gt_points)
            if num_kpts == 0:
                continue
            pred_points = _extract_pred_keypoints(pred, num_kpts)

            if len(per_kpt_visible) < num_kpts:
                per_kpt_visible.extend([0] * (num_kpts - len(per_kpt_visible)))
                per_kpt_correct.extend([0] * (num_kpts - len(per_kpt_correct)))

            norm = _norm_factor(gt["bbox"], args.norm)
            thr = float(args.pck_thr) * norm
            for k_idx, (gt_pt, pred_pt, v) in enumerate(zip(gt_points, pred_points, gt_vis)):
                if k_idx in exclude_set:
                    continue
                if v <= 0 or gt_pt is None:
                    continue
                if args.visible_mode == "gt":
                    per_kpt_visible[k_idx] += 1
                    total_visible += 1
                    if pred_pt is None:
                        continue
                else:
                    if pred_pt is None:
                        continue
                    per_kpt_visible[k_idx] += 1
                    total_visible += 1
                dx = gt_pt[0] - pred_pt[0]
                dy = gt_pt[1] - pred_pt[1]
                dist = math.hypot(dx, dy)
                if dist <= thr:
                    per_kpt_correct[k_idx] += 1
                    total_correct += 1
            matched_instances += 1
        if args.visible_mode == "gt":
            for gi, gt in enumerate(gt_instances):
                if gi in matched_gt:
                    continue
                gt_points, gt_vis = _extract_gt_keypoints(gt)
                num_kpts = len(gt_points)
                if num_kpts == 0:
                    continue
                if len(per_kpt_visible) < num_kpts:
                    per_kpt_visible.extend([0] * (num_kpts - len(per_kpt_visible)))
                    per_kpt_correct.extend([0] * (num_kpts - len(per_kpt_correct)))
                for k_idx, (gt_pt, v) in enumerate(zip(gt_points, gt_vis)):
                    if k_idx in exclude_set:
                        continue
                    if v <= 0 or gt_pt is None:
                        continue
                    per_kpt_visible[k_idx] += 1
                    total_visible += 1

    pck = total_correct / total_visible if total_visible else 0.0
    per_kpt = []
    for idx, (corr, vis) in enumerate(zip(per_kpt_correct, per_kpt_visible)):
        if idx in exclude_set:
            continue
        score = corr / vis if vis else 0.0
        per_kpt.append({"index": idx, "pck": score, "visible": vis})

    summary = {
        "pck": pck,
        "total_visible": total_visible,
        "total_correct": total_correct,
        "matched_instances": matched_instances,
        "per_keypoint": per_kpt,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
