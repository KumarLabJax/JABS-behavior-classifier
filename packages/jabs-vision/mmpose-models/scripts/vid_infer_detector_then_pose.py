#!/usr/bin/env python3
"""
Run a 2-stage instance pose pipeline on a video:
1) MMDetection detector -> bounding boxes per frame
2) MMPose top-down pose model -> keypoints per detected instance

Outputs JSON lines to stdout by default, or parquet/pose-v8 when the corresponding
output path flag is set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from mmcv import VideoReader
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


def _iter_frames(reader: VideoReader, max_frames: int | None) -> Iterable[Tuple[int, np.ndarray]]:
    for idx, frame in enumerate(reader):
        if max_frames is not None and idx >= max_frames:
            break
        yield idx, frame


def _run_detector_on_batch(det_model, frames: List[np.ndarray]):
    # Temporarily force MMDet scope so PackDetInputs resolves after MMPose sets its own scope.
    with DefaultScope.overwrite_default_scope("mmdet"):
        try:
            outputs = inference_detector(det_model, frames)
        except Exception:
            outputs = [inference_detector(det_model, frame) for frame in frames]
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return outputs


def _extract_det_results(
    det_sample,
    frame_shape: Tuple[int, int],
    det_score_thr: float,
    max_instances: int,
    bbox_padding: float,
    allow_empty: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        empty = det_bboxes.reshape((0, 4)).astype(np.float32)
        return empty, empty, det_scores.astype(np.float32, copy=False)

    order = np.argsort(-det_scores)
    order = order[: max(int(max_instances), 1)]
    det_bboxes = det_bboxes[order].astype(np.float32, copy=False)
    det_scores = det_scores[order]

    padded_bboxes = _pad_and_clip_xyxy(det_bboxes, frame_shape, padding=float(bbox_padding))
    det_scores = det_scores.astype(np.float32, copy=False)
    return det_bboxes, padded_bboxes, det_scores


PARQUET_KEYPOINTS = 5


POSE_V3_KEYPOINTS = 12
POSE_V3_KEYPOINT_NAME_MAP = {
    "nose": 0,
    "tip of nose": 0,
    "left ear": 1,
    "right ear": 2,
    "base neck": 3,
    "base of neck": 3,
    "left front paw": 4,
    "right front paw": 5,
    "center spine": 6,
    "left rear paw": 7,
    "left hind paw": 7,
    "right rear paw": 8,
    "right hind paw": 8,
    "base tail": 9,
    "base of tail": 9,
    "mid tail": 10,
    "middle tail": 10,
    "tip tail": 11,
    "tip of tail": 11,
}


def _normalize_keypoint_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())


def _build_pose_v3_keypoint_map(pose_model, num_keypoints: int) -> dict[int, int]:
    dataset_meta = getattr(pose_model, "dataset_meta", None) or {}
    keypoint_info = dataset_meta.get("keypoint_info") or {}
    mapping: dict[int, int] = {}
    if keypoint_info:
        try:
            items = sorted(keypoint_info.items(), key=lambda item: int(item[0]))
        except Exception:
            items = list(keypoint_info.items())
        for kpt_idx, info in items:
            try:
                src_idx = int(kpt_idx)
            except Exception:
                continue
            name = ""
            if isinstance(info, dict):
                name = info.get("name") or ""
            norm_name = _normalize_keypoint_name(name)
            if norm_name in POSE_V3_KEYPOINT_NAME_MAP:
                mapping[src_idx] = POSE_V3_KEYPOINT_NAME_MAP[norm_name]
    if not mapping:
        if num_keypoints == 5:
            mapping = {0: 0, 1: 1, 2: 2, 3: 9, 4: 11}
        elif num_keypoints == POSE_V3_KEYPOINTS:
            mapping = {i: i for i in range(POSE_V3_KEYPOINTS)}
    return mapping


def _get_pose_num_keypoints(pose_model) -> int:
    num_keypoints = int(getattr(pose_model, "num_keypoints", 0) or 0)
    if num_keypoints <= 0:
        head = getattr(pose_model, "head", None)
        num_keypoints = int(getattr(head, "out_channels", 0) or 0)
    return num_keypoints


def _extract_pose_arrays(pose_sample) -> tuple[np.ndarray, np.ndarray]:
    if pose_sample is None:
        return np.empty((0, 0, 2), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    pred_instances = getattr(pose_sample, "pred_instances", None)
    keypoints = _to_numpy(getattr(pred_instances, "keypoints", None))
    kpt_scores = _to_numpy(getattr(pred_instances, "keypoint_scores", None))
    if keypoints.ndim < 3:
        keypoints = np.empty((0, 0, 2), dtype=np.float32)
    if kpt_scores.ndim < 2:
        kpt_scores = np.empty((0, 0), dtype=np.float32)
    return keypoints, kpt_scores


class _PoseV8Buffer:
    """Stores PoseV8-style keypoints and metadata per frame for HDF5 export."""

    def __init__(self, num_frames: int, max_instances: int, keypoint_map: dict[int, int]) -> None:
        # Preallocate fixed-size arrays for fast indexed writes.
        self.points = np.full(
            (num_frames, max_instances, POSE_V3_KEYPOINTS, 2),
            np.nan,
            dtype=np.float32,
        )
        self.confidence = np.zeros(
            (num_frames, max_instances, POSE_V3_KEYPOINTS), dtype=np.float32
        )
        self.instance_count = np.zeros((num_frames,), dtype=np.int32)
        self.id_mask = np.ones((num_frames, max_instances), dtype=np.uint8)
        self.instance_embed_id = np.zeros((num_frames, max_instances), dtype=np.int32)
        self.bbox = np.full((num_frames, max_instances, 2, 2), np.nan, dtype=np.float32)
        self.keypoint_map = keypoint_map
        self.max_instances = max_instances

    def add_frame(self, frame_index: int, keypoints: np.ndarray, kpt_scores: np.ndarray) -> None:
        """Write keypoints/scores for a frame into the PoseV8 layout."""
        # Reset per-frame slots so repeated writes cannot leave stale state.
        self.points[frame_index, ...] = np.nan
        self.confidence[frame_index, ...] = 0
        self.id_mask[frame_index, ...] = 1
        self.instance_embed_id[frame_index, ...] = 0
        self.bbox[frame_index, ...] = np.nan

        if keypoints.size == 0:
            self.instance_count[frame_index] = 0
            return
        num_instances = min(self.max_instances, keypoints.shape[0])
        self.instance_count[frame_index] = num_instances
        self.id_mask[frame_index, :num_instances] = 0

        for src_idx, dst_idx in self.keypoint_map.items():
            if src_idx >= keypoints.shape[1]:
                continue

            # PoseV3 expects (y, x), so swap.
            self.points[frame_index, :num_instances, dst_idx, 0] = keypoints[
                :num_instances, src_idx, 1
            ]
            self.points[frame_index, :num_instances, dst_idx, 1] = keypoints[
                :num_instances, src_idx, 0
            ]
            if kpt_scores.size and kpt_scores.shape[1] > src_idx:
                self.confidence[frame_index, :num_instances, dst_idx] = kpt_scores[
                    :num_instances, src_idx
                ]


def _pose_v3_cost(
    points_a: np.ndarray,
    conf_a: np.ndarray,
    center_a: np.ndarray,
    points_b: np.ndarray,
    conf_b: np.ndarray,
    center_b: np.ndarray,
) -> float:
    # Only use finite coords with non-zero confidence on both sides.
    valid_a = np.isfinite(points_a).all(axis=1) & (conf_a > 0)
    valid_b = np.isfinite(points_b).all(axis=1) & (conf_b > 0)

    # Add detector center as a pseudo keypoint to guarantee finite overlap.
    points_a = np.concatenate([points_a, center_a[None, :]], axis=0)
    conf_a = np.concatenate([conf_a, np.array([1.0], dtype=conf_a.dtype)], axis=0)
    points_b = np.concatenate([points_b, center_b[None, :]], axis=0)
    conf_b = np.concatenate([conf_b, np.array([1.0], dtype=conf_b.dtype)], axis=0)
    valid_a = np.isfinite(points_a).all(axis=1) & (conf_a > 0)
    valid_b = np.isfinite(points_b).all(axis=1) & (conf_b > 0)
    valid = valid_a & valid_b
    if not np.any(valid):
        return float("inf")
    diffs = points_a[valid] - points_b[valid]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    return float(np.median(dists))


class _PoseV3Tracker:
    def __init__(self, max_gap_frames: int | None = 10) -> None:
        self.max_gap_frames = max_gap_frames
        self._next_track_id = 0
        # track_id -> (last_frame, points[K,2], confidence[K], center[2])
        self._tracks: dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _new_track_id(self) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _prune(self, frame_index: int) -> None:
        if self.max_gap_frames is None or self.max_gap_frames <= 0:
            return
        to_delete: list[int] = []
        for track_id, (last_frame, _points, _conf, _center) in self._tracks.items():
            if frame_index - last_frame > self.max_gap_frames:
                to_delete.append(track_id)
        for track_id in to_delete:
            del self._tracks[track_id]

    def assign(
        self,
        frame_index: int,
        frame_points: np.ndarray,
        frame_confidence: np.ndarray,
        frame_centers: np.ndarray,
        num_instances: int,
    ) -> np.ndarray:
        """Return assigned track ids for instances [0:num_instances] in this frame."""
        self._prune(frame_index)

        if num_instances <= 0:
            return np.empty((0,), dtype=np.int32)

        # Snapshot active tracks (may be empty).
        track_items = list(self._tracks.items())
        if not track_items:
            assigned = np.empty((num_instances,), dtype=np.int32)
            for i in range(num_instances):
                track_id = self._new_track_id()
                assigned[i] = track_id
                self._tracks[track_id] = (
                    frame_index,
                    frame_points[i].copy(),
                    frame_confidence[i].copy(),
                    frame_centers[i].copy(),
                )
            return assigned

        # Build all pairwise costs then greedily match lowest cost pairs first.
        costs: list[tuple[float, int, int]] = []
        for inst_idx in range(num_instances):
            for track_list_idx, (track_id, (last_frame, t_points, t_conf, t_center)) in enumerate(
                track_items
            ):
                _ = last_frame
                cost = _pose_v3_cost(
                    frame_points[inst_idx],
                    frame_confidence[inst_idx],
                    frame_centers[inst_idx],
                    t_points,
                    t_conf,
                    t_center,
                )
                costs.append((cost, inst_idx, track_list_idx))
        costs.sort(key=lambda item: item[0])

        assigned = np.full((num_instances,), -1, dtype=np.int32)
        used_instances: set[int] = set()
        used_tracks: set[int] = set()

        for cost, inst_idx, track_list_idx in costs:
            if inst_idx in used_instances:
                continue
            track_id = track_items[track_list_idx][0]
            if track_id in used_tracks:
                continue
            if not np.isfinite(cost):
                continue
            assigned[inst_idx] = track_id
            used_instances.add(inst_idx)
            used_tracks.add(track_id)
            self._tracks[track_id] = (
                frame_index,
                frame_points[inst_idx].copy(),
                frame_confidence[inst_idx].copy(),
                frame_centers[inst_idx].copy(),
            )

        # Create new tracks for any unassigned instances (e.g. no valid overlap).
        for inst_idx in range(num_instances):
            if assigned[inst_idx] >= 0:
                continue
            track_id = self._new_track_id()
            assigned[inst_idx] = track_id
            self._tracks[track_id] = (
                frame_index,
                frame_points[inst_idx].copy(),
                frame_confidence[inst_idx].copy(),
                frame_centers[inst_idx].copy(),
            )

        return assigned


def _pose_to_parquet_rows(
    frame_index: int,
    pose_sample,
    num_keypoints: int,
    det_bboxes: np.ndarray,
    det_scores: np.ndarray,
) -> List[dict]:
    if pose_sample is None or det_bboxes.size == 0:
        return []

    pred_instances = getattr(pose_sample, "pred_instances", None)
    keypoints = _to_numpy(getattr(pred_instances, "keypoints", None))
    if keypoints.size == 0 or keypoints.ndim < 3 or keypoints.shape[2] < 2:
        return []

    num_instances = min(det_bboxes.shape[0], keypoints.shape[0])
    rows: List[dict] = []
    for inst_idx in range(num_instances):
        row = {
            "frame_number": int(frame_index),
            "bb_left": float(det_bboxes[inst_idx, 0]),
            "bb_top": float(det_bboxes[inst_idx, 1]),
            "bb_right": float(det_bboxes[inst_idx, 2]),
            "bb_bottom": float(det_bboxes[inst_idx, 3]),
            "bb_conf": float(det_scores[inst_idx]) if det_scores.size > inst_idx else np.nan,
        }
        for kp_idx in range(num_keypoints):
            x_val = np.nan
            y_val = np.nan
            if keypoints.shape[1] > kp_idx:
                x_val = float(keypoints[inst_idx, kp_idx, 0])
                y_val = float(keypoints[inst_idx, kp_idx, 1])
            row[f"kpt_{kp_idx + 1}_x"] = x_val
            row[f"kpt_{kp_idx + 1}_y"] = y_val
        rows.append(row)

    return rows


def _process_batch(
    frame_idx: int,
    frame_batch: List[np.ndarray],
    index_batch: List[int],
    det_model,
    pose_model,
    args: argparse.Namespace,
    parquet_rows: List[dict] | None,
    pose_v8_buffer: _PoseV8Buffer | None,
    pose_v3_tracker: _PoseV3Tracker | None,
) -> None:
    det_outputs = _run_detector_on_batch(det_model, frame_batch)
    det_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i, det_sample in enumerate(det_outputs):
        img_h, img_w = int(frame_batch[i].shape[0]), int(frame_batch[i].shape[1])
        raw_det_bboxes, padded_det_bboxes, det_scores = _extract_det_results(
            det_sample,
            (img_h, img_w),
            args.det_score_thr,
            args.max_instances,
            args.bbox_padding,
            allow_empty=True,
        )
        det_results.append((raw_det_bboxes, padded_det_bboxes, det_scores))

    for i, img_bgr in enumerate(frame_batch):
        raw_det_bboxes, det_bboxes, det_scores = det_results[i]
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

        if pose_v8_buffer is not None:
            keypoints, kpt_scores = _extract_pose_arrays(pose_sample)
            pose_v8_buffer.add_frame(index_batch[i], keypoints, kpt_scores)
            if pose_v3_tracker is not None:
                frame_index = index_batch[i]
                num_instances = int(pose_v8_buffer.instance_count[frame_index])
                if num_instances > 0:
                    bboxes = det_bboxes[:num_instances]
                    centers = np.stack(
                        [(bboxes[:, 0] + bboxes[:, 2]) * 0.5, (bboxes[:, 1] + bboxes[:, 3]) * 0.5],
                        axis=1,
                    )
                else:
                    centers = np.empty((0, 2), dtype=np.float32)
                assigned = pose_v3_tracker.assign(
                    frame_index,
                    pose_v8_buffer.points[frame_index],
                    pose_v8_buffer.confidence[frame_index],
                    centers,
                    num_instances,
                )
                pose_v8_buffer.instance_embed_id[frame_index, :num_instances] = assigned + 1
                if num_instances > 0:
                    bboxes = raw_det_bboxes[:num_instances]
                    pose_v8_buffer.bbox[frame_index, :num_instances, 0, :] = bboxes[:, :2]
                    pose_v8_buffer.bbox[frame_index, :num_instances, 1, :] = bboxes[:, 2:]

        frame_shape = (img_bgr.shape[0], img_bgr.shape[1])
        if parquet_rows is not None:
            parquet_rows.extend(
                _pose_to_parquet_rows(
                    index_batch[i],
                    pose_sample,
                    PARQUET_KEYPOINTS,
                    det_bboxes,
                    det_scores,
                )
            )
        elif pose_v8_buffer is None:
            line = _frame_to_json(index_batch[i], frame_shape, det_bboxes, det_scores, pose_sample)
            print(f"Frame {frame_idx + i}: {line}")


def _frame_to_json(
    frame_index: int,
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
                "bbox_score": float(det_scores[i]),
            }
            if keypoints.size:
                item["keypoints"] = keypoints[i].tolist()
            if kpt_scores.size:
                item["keypoint_scores"] = kpt_scores[i].tolist()
            instances.append(item)

    record = {
        "frame_index": int(frame_index),
        "height": int(frame_shape[0]),
        "width": int(frame_shape[1]),
        "instances": instances,
    }
    return json.dumps(record, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector → top-down pose inference on a video.")
    parser.add_argument("video", type=Path, help="Input video path.")
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
    parser.add_argument("--batch-size", type=int, default=4, help="Number of frames per detector batch.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames processed.")
    parser.add_argument(
        "--max-tracklet-gap",
        type=int,
        default=None,
        help=
            "Max frames a tracklet can be absent before it is pruned. "
            "Use None for no pruning.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help="Write keypoints to a parquet file compatible with VideoMinuteLoaderWorker.",
    )
    output_group.add_argument(
        "--out-pose-v8",
        type=Path,
        default=None,
        help="Write keypoints to a pose_est v8 HDF5 file.",
    )
    return parser.parse_args()


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
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    batch_size = max(int(args.batch_size), 1)

    det_model = init_detector(str(args.det_config), str(args.det_checkpoint), device=args.device)
    pose_model = init_model(str(args.pose_config), str(args.pose_checkpoint), device=args.device)

    reader = VideoReader(str(args.video))

    frame_batch: List[np.ndarray] = []
    index_batch: List[int] = []
    parquet_rows: List[dict] | None = [] if args.out_parquet else None
    pose_v8_buffer: _PoseV8Buffer | None = None
    pose_v3_tracker: _PoseV3Tracker | None = None

    if args.out_pose_v8 is not None:
        try:
            total_frames = len(reader)
        except Exception:
            total_frames = getattr(reader, "frame_cnt", None)
        if total_frames is None:
            raise RuntimeError("Unable to determine total frames for pose v8 output.")
        if args.max_frames is not None:
            total_frames = min(total_frames, int(args.max_frames))
        keypoint_map = _build_pose_v3_keypoint_map(pose_model, _get_pose_num_keypoints(pose_model))
        if not keypoint_map:
            raise RuntimeError("Unable to infer keypoint mapping for pose v8 output.")
        pose_v8_buffer = _PoseV8Buffer(total_frames, int(args.max_instances), keypoint_map)
        pose_v3_tracker = _PoseV3Tracker(max_gap_frames=args.max_tracklet_gap)

    with torch.no_grad():
        for frame_idx, frame_bgr in _iter_frames(reader, args.max_frames):
            frame_batch.append(frame_bgr)
            index_batch.append(frame_idx)

            if len(frame_batch) < batch_size:
                continue

            _process_batch(
                frame_idx,
                frame_batch,
                index_batch,
                det_model,
                pose_model,
                args,
                parquet_rows,
                pose_v8_buffer,
                pose_v3_tracker,
            )

            frame_batch.clear()
            index_batch.clear()

        if frame_batch:
            _process_batch(
                frame_idx,
                frame_batch,
                index_batch,
                det_model,
                pose_model,
                args,
                parquet_rows,
                pose_v8_buffer,
                pose_v3_tracker,
            )

    if args.out_parquet is not None:
        columns = ["frame_number", "bb_left", "bb_top", "bb_right", "bb_bottom", "bb_conf"]
        for kp_idx in range(1, PARQUET_KEYPOINTS + 1):
            columns.extend([f"kpt_{kp_idx}_x", f"kpt_{kp_idx}_y"])
        df = pd.DataFrame(parquet_rows or [], columns=columns)
        df.to_parquet(args.out_parquet, index=False)

    if args.out_pose_v8 is not None and pose_v8_buffer is not None:
        args.out_pose_v8.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.out_pose_v8, "w") as pose_h5:
            pose_grp = pose_h5.create_group("poseest")
            pose_grp.attrs["version"] = np.array([8, 0, 0], dtype=np.int32)
            pose_grp.create_dataset("points", data=pose_v8_buffer.points)
            pose_grp.create_dataset("confidence", data=pose_v8_buffer.confidence)
            pose_grp.create_dataset("id_mask", data=pose_v8_buffer.id_mask)
            pose_grp.create_dataset("instance_embed_id", data=pose_v8_buffer.instance_embed_id)
            bbox_ds = pose_grp.create_dataset("bbox", data=pose_v8_buffer.bbox)
            bbox_ds.attrs["bboxes_generated"] = True


if __name__ == "__main__":
    main()
