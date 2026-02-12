#!/usr/bin/env python3
"""
Render pose keypoints from scripts/vid_infer_detector_then_pose.py output onto a video.

Input pose data can be JSON lines (stdout capture), parquet produced by --out-parquet,
or pose HDF5 (v3/v8) produced by vid_infer_detector_then_pose.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from distinctipy import distinctipy
import h5py


SKELETON_SEGMENTS = [
    [1, 0, 2],  # left ear -> nose -> right ear
    [0, 3, 4],  # nose -> base tail -> tip tail
]


def _parse_color(value: str) -> Tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Color must be in B,G,R format like 0,255,0")
    try:
        bgr = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color must be integers like 0,255,0") from exc
    if any(p < 0 or p > 255 for p in bgr):
        raise argparse.ArgumentTypeError("Color values must be in 0-255 range")
    return bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render pose keypoints from vid_infer_detector_then_pose.py output onto a video."
    )
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument("pose", type=Path, help="Pose output (JSON lines, parquet, or pose HDF5).")
    parser.add_argument("out_video", type=Path, help="Rendered output video path.")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Keypoint score threshold.")
    parser.add_argument("--kpt-radius", type=int, default=3, help="Keypoint circle radius (pixels).")
    parser.add_argument("--kpt-color", type=_parse_color, default="0,255,0", help="Keypoint color B,G,R.")
    parser.add_argument("--draw-bbox", action="store_true", help="Draw bounding boxes if present.")
    parser.add_argument("--bbox-color", type=_parse_color, default="255,0,0", help="BBox color B,G,R.")
    parser.add_argument("--line-thickness", type=int, default=1, help="Line thickness for bboxes and skeleton lines.")
    parser.add_argument("--no-skel", action="store_true", help="Disable skeleton rendering.")
    parser.add_argument(
        "--exclude-points",
        type=int,
        nargs="+",
        default=[],
        help="keypoint indexes to exclude from rendering.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames rendered.")
    return parser.parse_args()


def _strip_frame_prefix(line: str) -> str:
    if line.startswith("Frame "):
        _, payload = line.split(":", 1)
        return payload.strip()
    return line


def _load_pose_jsonl(path: Path) -> Dict[int, List[dict]]:
    frames: Dict[int, List[dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            line = _strip_frame_prefix(line)
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame_index = record.get("frame_index")
            if frame_index is None:
                continue
            instances = record.get("instances") or []
            frames.setdefault(int(frame_index), []).extend(instances)
    return frames


def _parquet_keypoint_columns(columns: Iterable[str]) -> List[Tuple[str, str]]:
    pairs = []
    for name in columns:
        if name.startswith("kpt_") and name.endswith("_x"):
            base = name[:-2]
            y_name = f"{base}_y"
            pairs.append((name, y_name))
    def _sort_key(pair: Tuple[str, str]) -> int:
        stem = pair[0].split("_")
        if len(stem) >= 2 and stem[1].isdigit():
            return int(stem[1])
        return 0
    return sorted(pairs, key=_sort_key)


def _load_pose_parquet(path: Path) -> Dict[int, List[dict]]:
    import pandas as pd

    df = pd.read_parquet(path)
    frame_col = "frame_number" if "frame_number" in df.columns else "frame_index"
    if frame_col not in df.columns:
        raise ValueError(f"Parquet missing frame index column (expected frame_number or frame_index).")
    has_bbox = all(col in df.columns for col in ("bb_left", "bb_top", "bb_right", "bb_bottom"))
    pairs = _parquet_keypoint_columns(df.columns)
    frames: Dict[int, List[dict]] = {}
    for _, row in df.iterrows():
        frame_index = int(row[frame_col])
        instance: dict = {}
        if has_bbox:
            left = row["bb_left"]
            top = row["bb_top"]
            right = row["bb_right"]
            bottom = row["bb_bottom"]
            if not any(np.isnan(val) for val in (left, top, right, bottom)):
                instance["bbox"] = [float(left), float(top), float(right), float(bottom)]
        keypoints = []
        for x_col, y_col in pairs:
            x_val = row[x_col]
            y_val = row[y_col]
            if np.isnan(x_val) or np.isnan(y_val):
                keypoints.append([float("nan"), float("nan")])
            else:
                keypoints.append([float(x_val), float(y_val)])
        instance["keypoints"] = keypoints
        frames.setdefault(frame_index, []).append(instance)
    return frames


def _load_pose_h5(path: Path) -> Dict[int, List[dict]]:

    frames: Dict[int, List[dict]] = {}
    with h5py.File(path, "r") as pose_h5:
        if "poseest" not in pose_h5:
            raise ValueError("Pose file missing 'poseest' group.")
        pose_grp = pose_h5["poseest"]
        if "points" not in pose_grp or "confidence" not in pose_grp:
            raise ValueError("Pose file missing required datasets (points, confidence).")
        points = pose_grp["points"]
        confidence = pose_grp["confidence"]
        num_frames = int(points.shape[0])
        # Legacy v3 path.
        if "instance_count" in pose_grp:
            instance_count = pose_grp["instance_count"]
            for frame_index in range(num_frames):
                num_instances = int(instance_count[frame_index])
                if num_instances <= 0:
                    continue
                instances: List[dict] = []
                frame_points = points[frame_index, :num_instances]
                # Pose stores keypoints as (y, x); convert to (x, y) for rendering.
                frame_points = np.flip(frame_points, axis=-1)
                frame_conf = confidence[frame_index, :num_instances]
                for inst_idx in range(num_instances):
                    keypoints = frame_points[inst_idx].tolist()
                    keypoint_scores = frame_conf[inst_idx].tolist()
                    instances.append(
                        {
                            "keypoints": keypoints,
                            "keypoint_scores": keypoint_scores,
                        }
                    )
                frames[frame_index] = instances
            return frames

        # v4+ path (including v8): valid slots indicated by id_mask == 0.
        if "id_mask" not in pose_grp:
            raise ValueError("Pose file missing identity indexing dataset (instance_count or id_mask).")

        id_mask = pose_grp["id_mask"]
        instance_embed_id = pose_grp.get("instance_embed_id")
        bbox_ds = pose_grp.get("bbox")
        bbox_available = (
            bbox_ds is not None and bool(bbox_ds.attrs.get("bboxes_generated", False))
        )

        for frame_index in range(num_frames):
            valid_slots = np.where(id_mask[frame_index] == 0)[0]
            if valid_slots.size == 0:
                continue

            instances: List[dict] = []
            frame_points = np.flip(points[frame_index, valid_slots], axis=-1)
            frame_conf = confidence[frame_index, valid_slots]

            for local_idx, slot_idx in enumerate(valid_slots.tolist()):
                instance = {
                    "keypoints": frame_points[local_idx].tolist(),
                    "keypoint_scores": frame_conf[local_idx].tolist(),
                }

                if instance_embed_id is not None:
                    embed_id = int(instance_embed_id[frame_index, slot_idx])
                    if embed_id > 0:
                        instance["instance_embed_id"] = embed_id

                if bbox_available:
                    bbox = bbox_ds[frame_index, slot_idx]
                    if np.isfinite(bbox).all():
                        instance["bbox"] = [
                            float(bbox[0, 0]),
                            float(bbox[0, 1]),
                            float(bbox[1, 0]),
                            float(bbox[1, 1]),
                        ]

                instances.append(instance)

            frames[frame_index] = instances

    return frames


def _load_pose(path: Path) -> Dict[int, List[dict]]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return _load_pose_parquet(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return _load_pose_h5(path)
    return _load_pose_jsonl(path)


def _iter_keypoints(instance: dict, kpt_thr: float) -> Iterable[Tuple[int, float, float]]:
    keypoints = instance.get("keypoints") or []
    scores = instance.get("keypoint_scores")
    for idx, kp in enumerate(keypoints):
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            continue
        x_val = float(kp[0])
        y_val = float(kp[1])
        score = None
        if scores is not None and idx < len(scores):
            score = float(scores[idx])
        elif len(kp) >= 3:
            score = float(kp[2])
        if score is not None and score < kpt_thr:
            continue
        if np.isnan(x_val) or np.isnan(y_val):
            continue
        yield idx, x_val, y_val


def _instance_keypoints(instance: dict, kpt_thr: float) -> List[Tuple[float, float] | None]:
    keypoints = instance.get("keypoints") or []
    scores = instance.get("keypoint_scores")
    points: List[Tuple[float, float] | None] = []
    for idx, kp in enumerate(keypoints):
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            points.append(None)
            continue
        x_val = float(kp[0])
        y_val = float(kp[1])
        score = None
        if scores is not None and idx < len(scores):
            score = float(scores[idx])
        elif len(kp) >= 3:
            score = float(kp[2])
        if score is not None and score < kpt_thr:
            points.append(None)
            continue
        if np.isnan(x_val) or np.isnan(y_val):
            points.append(None)
            continue
        points.append((x_val, y_val))
    return points


def _avg_keypoint_distance(
    prev_points: List[Tuple[float, float] | None],
    curr_points: List[Tuple[float, float] | None],
) -> float:
    total = 0.0
    count = 0
    for p_prev, p_curr in zip(prev_points, curr_points):
        if p_prev is None or p_curr is None:
            continue
        dx = p_prev[0] - p_curr[0]
        dy = p_prev[1] - p_curr[1]
        total += float(np.hypot(dx, dy))
        count += 1
    if count == 0:
        return float("inf")
    return total / float(count)


def _match_instances(
    prev_instances: List[dict],
    curr_instances: List[dict],
    kpt_thr: float,
) -> Tuple[Dict[int, int], List[int]]:
    prev_points = [_instance_keypoints(inst, kpt_thr) for inst in prev_instances]
    curr_points = [_instance_keypoints(inst, kpt_thr) for inst in curr_instances]
    pairs: List[Tuple[float, int, int]] = []
    for i, p_prev in enumerate(prev_points):
        for j, p_curr in enumerate(curr_points):
            dist = _avg_keypoint_distance(p_prev, p_curr)
            if np.isfinite(dist):
                pairs.append((dist, i, j))
    pairs.sort(key=lambda item: item[0])
    matched_prev = set()
    matched_curr = set()
    mapping: Dict[int, int] = {}
    for _, i, j in pairs:
        if i in matched_prev or j in matched_curr:
            continue
        mapping[j] = i
        matched_prev.add(i)
        matched_curr.add(j)
    unmatched_curr = [idx for idx in range(len(curr_instances)) if idx not in matched_curr]
    return mapping, unmatched_curr


def _draw_instances(
    frame: np.ndarray,
    instances: List[dict],
    instance_colors: List[Tuple[int, int, int]],
    kpt_thr: float,
    kpt_radius: int,
    draw_bbox: bool,
    bbox_color: Tuple[int, int, int],
    line_thickness: int,
    draw_skeleton: bool,
    exclude_points: List[int],
) -> None:
    exclude_set = set(exclude_points)
    for inst, kpt_color in zip(instances, instance_colors):
        if draw_bbox and "bbox" in inst:
            bbox = inst.get("bbox") or []
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = (int(b) for b in bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, line_thickness)
        if draw_skeleton:
            points = _instance_keypoints(inst, kpt_thr)
            for segment in SKELETON_SEGMENTS:
                for start_idx, end_idx in zip(segment, segment[1:]):
                    if start_idx >= len(points) or end_idx >= len(points):
                        continue
                    if start_idx in exclude_set or end_idx in exclude_set:
                        continue
                    start = points[start_idx]
                    end = points[end_idx]
                    if start is None or end is None:
                        continue
                    cv2.line(
                        frame,
                        (int(start[0]), int(start[1])),
                        (int(end[0]), int(end[1])),
                        kpt_color,
                        line_thickness,
                    )
        for idx, x_val, y_val in _iter_keypoints(inst, kpt_thr):
            if idx in exclude_set:
                continue
            cv2.circle(frame, (int(x_val), int(y_val)), kpt_radius, kpt_color, -1)


def _distinct_bgr(existing: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    existing_rgb = [(c[2] / 255.0, c[1] / 255.0, c[0] / 255.0) for c in existing]
    color = distinctipy.get_colors(1, exclude_colors=existing_rgb)[0]
    return (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))


def main() -> None:
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.pose.exists():
        raise FileNotFoundError(f"Pose file not found: {args.pose}")

    pose_by_frame = _load_pose(args.pose)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {args.out_video}")

    frame_idx = 0
    tracks: List[dict] = []
    track_colors: List[Tuple[int, int, int]] = []
    next_track_id = 0
    max_frames = args.max_frames
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        instances = pose_by_frame.get(frame_idx, [])
        if instances:
            if tracks:
                mapping, unmatched = _match_instances(
                    [t["instance"] for t in tracks],
                    instances,
                    float(args.kpt_thr),
                )
            else:
                mapping, unmatched = ({}, list(range(len(instances))))

            new_tracks: List[dict] = []
            instance_colors: List[Tuple[int, int, int]] = [args.kpt_color] * len(instances)

            for curr_idx, prev_idx in mapping.items():
                track = tracks[prev_idx]
                new_tracks.append(
                    {
                        "id": track["id"],
                        "instance": instances[curr_idx],
                        "color": track["color"],
                    }
                )
                instance_colors[curr_idx] = track["color"]

            for curr_idx in unmatched:
                track_id = next_track_id
                next_track_id += 1
                if len(track_colors) <= track_id:
                    track_colors.append(_distinct_bgr(track_colors))
                color = track_colors[track_id]
                new_tracks.append({"id": track_id, "instance": instances[curr_idx], "color": color})
                instance_colors[curr_idx] = color

            tracks = new_tracks

            _draw_instances(
                frame,
                instances,
                instance_colors,
                float(args.kpt_thr),
                int(args.kpt_radius),
                bool(args.draw_bbox),
                args.bbox_color,
                int(args.line_thickness),
                not bool(args.no_skel),
                args.exclude_points,
            )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(args.out_video)


if __name__ == "__main__":
    main()
