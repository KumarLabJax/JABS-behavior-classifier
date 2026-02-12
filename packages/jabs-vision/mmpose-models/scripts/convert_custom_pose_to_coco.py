#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_RAW_META: Dict = dict(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dax3 custom pose annotations into COCO keypoint format."
    )
    parser.add_argument("input_json", help="Path to the custom annotation JSON file.")
    parser.add_argument("output_json", help="Destination path for the COCO JSON file.")
    parser.add_argument(
        "--meta",
        help=(
            "Optional JSON file that provides `keypoints`/`skeleton` metadata. "
            "If omitted the script looks for a `meta` block in the source file "
            "and finally falls back to an internal default pose definition."
        ),
    )
    parser.add_argument(
        "--category-name",
        default="mouse",
        help="COCO category name to emit (default: %(default)s).",
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=1,
        help="COCO category id to emit (default: %(default)s).",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        metavar="PREFIX",
        help="Strip this prefix from every `image_file`. Can be specified multiple times.",
    )
    parser.add_argument(
        "--image-prefix",
        default="",
        help="Optional path prefix to prepend to every `file_name` entry.",
    )
    return parser.parse_args()


def read_payload(path: str) -> Dict | List:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Dict) -> None:
    target = Path(path)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def extract_samples(payload) -> List[Dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
        if not isinstance(samples, list):
            raise TypeError("`samples` entry must be a list.")
        return samples
    raise TypeError("Input JSON must be a list of samples or a dict with a `samples` list.")


def load_meta(args_meta: str | None, payload) -> Dict | None:
    if args_meta:
        with open(args_meta, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if isinstance(payload, dict):
        meta = payload.get("meta")
        if meta:
            return meta
    return DEFAULT_RAW_META


def infer_keypoint_order(samples: Sequence[Dict]) -> List[str]:
    ordered = OrderedDict()
    for sample in samples:
        for kp in sample.get("keypoints", []):
            name = kp.get("label")
            if name and name not in ordered:
                ordered[name] = None
    return list(ordered.keys())


def derive_keypoints_and_skeleton(meta: Dict | None, samples: Sequence[Dict]) -> Tuple[List[str], List[List[int]]]:
    if meta and "keypoints" in meta:
        names = [entry["name"] for entry in meta["keypoints"] if "name" in entry]
    else:
        names = infer_keypoint_order(samples)
    if not names:
        raise ValueError("Unable to determine keypoint ordering. Provide a meta file with `keypoints`.")
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    skeleton_pairs: List[List[int]] = []
    if meta:
        for limb in meta.get("skeleton", []):
            link = limb.get("link")
            if not isinstance(link, Sequence) or len(link) != 2:
                continue
            if link[0] not in name_to_idx or link[1] not in name_to_idx:
                continue
            skeleton_pairs.append(
                [name_to_idx[link[0]] + 1, name_to_idx[link[1]] + 1]
            )
    return names, skeleton_pairs


def normalize_image_path(image_file: str, strip_prefixes: Iterable[str], prepend: str) -> str:
    normalized = image_file or ""
    for prefix in strip_prefixes:
        if not prefix:
            continue
        if normalized.lower().startswith(prefix.lower()):
            normalized = normalized[len(prefix):]
    if prepend:
        normalized = str(Path(prepend) / normalized)
    return normalized


def group_keypoints(keypoints: Sequence[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, Dict[str, float]]] = {}
    fallback_idx = 0
    for kp in keypoints or []:
        label = kp.get("label")
        if label is None:
            continue
        instance_id = kp.get("individual_uuid")
        if not instance_id:
            instance_id = f"group_{fallback_idx}"
            fallback_idx += 1
        grouped.setdefault(instance_id, {})
        grouped[instance_id][label] = kp
    return grouped


def build_bbox_map(sample: Dict) -> Tuple[Dict[str, List[float]], List[float] | None]:
    mapping: Dict[str, List[float]] = {}
    fallback: List[float] | None = None
    direct_bbox = sample.get("bbox")
    if isinstance(direct_bbox, Sequence) and len(direct_bbox) == 4:
        fallback = [float(val) for val in direct_bbox]
    for bbox in sample.get("bboxes", []) or []:
        if not isinstance(bbox, dict):
            continue
        parts = [bbox.get("left"), bbox.get("top"), bbox.get("width"), bbox.get("height")]
        if any(part is None for part in parts):
            continue
        coords = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
        uuid = bbox.get("individual_uuid")
        if uuid:
            mapping[uuid] = coords
        if fallback is None:
            fallback = coords
    return mapping, fallback


def build_segmentation_map(sample: Dict) -> Dict[str | None, List[List[float]]]:
    seg_map: Dict[str | None, List[List[float]]] = defaultdict(list)
    for seg in sample.get("segmentations", []) or []:
        vertices = seg.get("vertices") or []
        flattened: List[float] = []
        for pair in vertices:
            if "x" not in pair or "y" not in pair:
                continue
            flattened.extend([float(pair["x"]), float(pair["y"])])
        if len(flattened) < 6:
            continue
        seg_map[seg.get("individual_uuid")].append(flattened)
    return seg_map


def polygon_area(flattened: Sequence[float]) -> float:
    if len(flattened) < 6:
        return 0.0
    coords = list(zip(flattened[0::2], flattened[1::2]))
    area = 0.0
    for idx in range(len(coords)):
        x1, y1 = coords[idx]
        x2, y2 = coords[(idx + 1) % len(coords)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def confidence_to_visibility(value) -> int:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "high":
            return 2
        if lowered == "medium":
            return 1
        if lowered == "low":
            return 1
    if isinstance(value, (int, float)):
        if value <= 0:
            return 0
        if value >= 2:
            return 2
        return 1
    return 2


def encode_keypoints(kp_map: Dict[str, Dict], order: Sequence[str]) -> Tuple[List[float], int]:
    encoded: List[float] = []
    visible = 0
    for name in order:
        entry = kp_map.get(name)
        if entry is None:
            encoded.extend([0.0, 0.0, 0])
            continue
        x = float(entry["x"])
        y = float(entry["y"])
        v = confidence_to_visibility(entry.get("confidence"))
        encoded.extend([x, y, v])
        if v > 0:
            visible += 1
    return encoded, visible


def get_image_dims(sample: Dict) -> Tuple[int, int]:
    frame_shape = sample.get("frame_shape")
    if isinstance(frame_shape, Sequence) and len(frame_shape) == 2:
        return int(round(float(frame_shape[0]))), int(round(float(frame_shape[1])))
    width = sample.get("width")
    height = sample.get("height")
    if width is not None and height is not None:
        return int(round(float(width))), int(round(float(height)))
    raise ValueError("Sample is missing `frame_shape` or explicit `width`/`height`.")


def resolve_segments(
    uuid: str,
    seg_map: Dict[str | None, List[List[float]]],
    grouped_keypoints: Dict[str, Dict],
) -> List[List[float]]:
    if uuid in seg_map:
        return seg_map[uuid]
    if None in seg_map and len(seg_map[None]) == 1 and len(grouped_keypoints) == 1:
        return seg_map[None]
    return []


def convert_samples_to_coco(
    samples: Sequence[Dict],
    keypoint_names: Sequence[str],
    skeleton_pairs: Sequence[Sequence[int]],
    args: argparse.Namespace,
) -> Dict:
    images: List[Dict] = []
    annotations: List[Dict] = []
    annotation_id = 1
    skipped_samples = 0
    skipped_instances = 0

    for image_id, sample in enumerate(samples, start=1):
        try:
            width, height = get_image_dims(sample)
        except ValueError:
            skipped_samples += 1
            continue
        file_name = normalize_image_path(sample.get("image_file", ""), args.strip_prefix, args.image_prefix)
        images.append(
            dict(
                id=image_id,
                file_name=file_name,
                width=width,
                height=height,
            )
        )

        grouped = group_keypoints(sample.get("keypoints", []))
        if not grouped:
            skipped_samples += 1
            continue
        bbox_map, fallback_bbox = build_bbox_map(sample)
        seg_map = build_segmentation_map(sample)
        for uuid, kp_map in grouped.items():
            bbox = bbox_map.get(uuid) or fallback_bbox
            if bbox is None:
                skipped_instances += 1
                continue
            x, y, w, h = [float(val) for val in bbox]
            if w <= 0 or h <= 0:
                skipped_instances += 1
                continue
            keypoints, num_kps = encode_keypoints(kp_map, keypoint_names)
            if num_kps == 0:
                continue
            segments = resolve_segments(uuid, seg_map, grouped)
            seg_area = sum(polygon_area(poly) for poly in segments)
            area = seg_area if seg_area > 0 else w * h
            annotations.append(
                dict(
                    id=annotation_id,
                    image_id=image_id,
                    category_id=args.category_id,
                    bbox=[x, y, w, h],
                    area=area,
                    iscrowd=int(sample.get("iscrowd", 0)),
                    keypoints=keypoints,
                    num_keypoints=num_kps,
                    segmentation=segments,
                )
            )
            annotation_id += 1

    if skipped_samples:
        print(f"[convert] Skipped {skipped_samples} images without usable keypoints or dimensions.")
    if skipped_instances:
        print(f"[convert] Skipped {skipped_instances} instances without bounding boxes.")

    categories = [
        dict(
            id=args.category_id,
            name=args.category_name,
            supercategory=args.category_name,
            keypoints=list(keypoint_names),
            skeleton=[list(pair) for pair in skeleton_pairs],
        )
    ]
    return dict(images=images, annotations=annotations, categories=categories)


def main() -> None:
    args = parse_args()
    payload = read_payload(args.input_json)
    samples = extract_samples(payload)
    meta = load_meta(args.meta, payload)
    keypoint_names, skeleton_pairs = derive_keypoints_and_skeleton(meta, samples)
    coco = convert_samples_to_coco(samples, keypoint_names, skeleton_pairs, args)
    write_json(args.output_json, coco)
    print(
        f"Wrote {len(coco['images'])} images and {len(coco['annotations'])} annotations to {args.output_json}"
    )


if __name__ == "__main__":
    main()
