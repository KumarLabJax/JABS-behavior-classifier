from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from mmengine.config import Config

from mmpose.registry import DATASETS, VISUALIZERS


def choose_indices(population: int, count: int) -> List[int]:
    """Return up to `count` unique random indices for a dataset."""

    if population <= 0:
        return []
    if count >= population:
        return list(range(population))
    return random.sample(range(population), count)


def prepare_visualizer_image(inputs):
    """Convert tensor inputs to HWC numpy arrays for visualization."""

    if isinstance(inputs, torch.Tensor):
        array = inputs.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        return array
    return inputs


def log_keypoint_stats(split_name: str, idx: int, data_sample) -> None:
    """Print keypoint tensor stats for a dataset sample."""

    if not hasattr(data_sample, "gt_instances"):
        print(f"[visualize_samples] {split_name}[{idx}] missing gt_instances")
        return
    instances = data_sample.gt_instances
    kpts = instances.get("transformed_keypoints", instances.keypoints)
    if kpts is None:
        print(f"[visualize_samples] {split_name}[{idx}] has no keypoints tensor")
        return
    vis = getattr(instances, "keypoints_visible", None)
    counts = int((np.asarray(vis) > 0).sum()) if vis is not None else 0
    msg = f"[visualize_samples] {split_name}[{idx}] keypoints shape={tuple(kpts.shape)}"
    if vis is not None:
        msg += f", visible pts={counts}"
    else:
        msg += ", keypoints_visible missing"
    print(msg)
    print("Keypoints:")
    print(kpts)
    print("Bounding boxes:")
    print(instances.bboxes)


def overlay_occluded_keypoints(image_path: Path, data_sample, radius: int = 5) -> None:
    """Annotate occluded keypoints on the saved visualization image."""

    instances = getattr(data_sample, "gt_instances", None)
    if instances is None or not hasattr(instances, "keypoints_visible"):
        return
    keypoints = instances.get("transformed_keypoints", instances.keypoints)
    visibility = instances.keypoints_visible
    if keypoints is None or visibility is None:
        return
    occluded_points: List[Tuple[float, float]] = []
    for person_kpts, person_vis in zip(keypoints, visibility):
        mask = np.asarray(person_vis) <= 0
        if not np.any(mask):
            continue
        coords = np.asarray(person_kpts)[mask]
        for coord in coords:
            if len(coord) >= 2 and np.isfinite(coord[0]) and np.isfinite(coord[1]):
                occluded_points.append((float(coord[0]), float(coord[1])))
    if not occluded_points:
        return
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    outline_color = (255, 255, 0)
    for x, y in occluded_points:
        bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bbox, outline=outline_color, width=2)
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=outline_color, width=1)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=outline_color, width=1)
    image.save(image_path)


def visualize_dataset_samples(cfg: Config, num_samples: int) -> None:
    """Create sample visualizations for each dataset split."""

    if num_samples <= 0:
        return

    visualizer = VISUALIZERS.build(cfg.visualizer)
    out_dir = Path(cfg.work_dir) / "sample_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train": DATASETS.build(deepcopy(cfg.train_dataloader["dataset"])),
        "val": DATASETS.build(deepcopy(cfg.val_dataloader["dataset"])),
        "test": DATASETS.build(deepcopy(cfg.test_dataloader["dataset"])),
    }

    for split_name, dataset in datasets.items():
        if len(dataset) == 0:
            continue
        visualizer.set_dataset_meta(dataset.metainfo)
        indices = choose_indices(len(dataset), num_samples)
        for idx in indices:
            sample = dataset[idx]
            inputs = prepare_visualizer_image(sample["inputs"])
            data_sample = sample["data_samples"]
            if isinstance(data_sample, list):
                if not data_sample:
                    continue
                data_sample = data_sample[0]
            log_keypoint_stats(split_name, idx, data_sample)
            save_path = out_dir / f"{split_name}_{idx}.png"
            visualizer.add_datasample(
                f"{split_name}_{idx}",
                inputs,
                data_sample=data_sample,
                draw_gt=True,
                draw_pred=False,
                show=False,
                out_file=str(save_path),
                kpt_thr=-1,
            )
            overlay_occluded_keypoints(save_path, data_sample)
