from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.evaluator.metric import _to_cpu

try:
    from mmdet.registry import METRICS
except ImportError:  # pragma: no cover
    METRICS = None


def _instance_field(instances, key: str):
    if instances is None:
        return None
    if hasattr(instances, key):
        return getattr(instances, key)
    if isinstance(instances, dict):
        return instances.get(key)
    try:
        return instances[key]
    except (KeyError, TypeError, IndexError):
        return None


def _safe_get_boxes(instances, key: str) -> np.ndarray:
    data = _instance_field(instances, key)
    if data is None:
        return np.empty((0, 4), dtype=np.float32)
    if hasattr(data, "numpy"):
        return np.asarray(data.numpy(), dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def _safe_get_scores(instances, key: str = "scores") -> np.ndarray:
    data = _instance_field(instances, key)
    if data is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(data, "numpy"):
        return np.asarray(data.numpy(), dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(ix2 - ix1, 0.0)
    inter_h = np.maximum(iy2 - iy1, 0.0)
    inter = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0 or precisions.size == 0:
        return 0.0
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    changing_points = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum(
        (recalls[changing_points + 1] - recalls[changing_points]) * precisions[changing_points + 1]
    )
    return float(ap)


if METRICS is not None:

    @METRICS.register_module()
    class SingleClassAPMetric(BaseMetric):
        """Compute AP for a single detection class at a fixed IoU threshold."""

        default_prefix = "single_class_ap"

        def __init__(self, iou_thr: float = 0.5, prefix: str | None = None) -> None:
            super().__init__(collect_device="cpu", prefix=prefix)
            self.iou_thr = float(iou_thr)

        def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
            """Process a batch of data samples.

            Args:
                data_batch: A batch of input data.
                data_samples: A sequence of data samples containing ground truth
                    and predicted instances.
            """
            samples = _to_cpu(data_samples)
            for idx, sample in enumerate(samples):
                gt_instances = sample.get("gt_instances", {})
                pred_instances = sample.get("pred_instances", {})
                record = {
                    "image_id": sample.get("img_id", idx),
                    "gt_bboxes": _safe_get_boxes(gt_instances, "bboxes"),
                    "pred_bboxes": _safe_get_boxes(pred_instances, "bboxes"),
                    "pred_scores": _safe_get_scores(pred_instances),
                }
                self.results.append(record)

        def compute_metrics(self, results: list) -> dict:
            """Compute AP metrics from detection results.

            Args:
                results: A list of detection results containing gt_bboxes,
                    pred_bboxes, and pred_scores for each image.

            Returns:
                A dictionary containing AP, precision, and recall metrics.
            """
            total_gts = int(sum(res["gt_bboxes"].shape[0] for res in results))
            if total_gts == 0:
                return {"AP": 0.0, "precision": 0.0, "recall": 0.0}

            gt_cache: dict[int, dict[str, np.ndarray]] = {}
            for res in results:
                img_id = res["image_id"]
                boxes = res["gt_bboxes"]
                if img_id in gt_cache:
                    cache = gt_cache[img_id]
                    cache["boxes"] = np.concatenate([cache["boxes"], boxes], axis=0)
                    cache["matched"] = np.concatenate(
                        [cache["matched"], np.zeros(boxes.shape[0], dtype=bool)], axis=0
                    )
                else:
                    gt_cache[img_id] = {
                        "boxes": boxes,
                        "matched": np.zeros(boxes.shape[0], dtype=bool),
                    }

            detections = []
            for res in results:
                img_id = res["image_id"]
                boxes = res["pred_bboxes"]
                scores = res["pred_scores"]
                for box, score in zip(boxes, scores, strict=False):
                    detections.append((img_id, float(score), box))
            detections.sort(key=lambda item: item[1], reverse=True)

            tps: list[int] = []
            fps: list[int] = []
            for img_id, _, box in detections:
                cache = gt_cache.get(img_id)
                if cache is None or cache["boxes"].size == 0:
                    fps.append(1)
                    tps.append(0)
                    continue

                ious = _compute_iou(box, cache["boxes"])
                if ious.size == 0:
                    fps.append(1)
                    tps.append(0)
                    continue
                best_idx = int(ious.argmax())
                best_iou = float(ious[best_idx])
                if best_iou >= self.iou_thr and not cache["matched"][best_idx]:
                    cache["matched"][best_idx] = True
                    tps.append(1)
                    fps.append(0)
                else:
                    fps.append(1)
                    tps.append(0)

            if not tps:
                return {"AP": 0.0, "precision": 0.0, "recall": 0.0}

            tp_cum = np.cumsum(tps)
            fp_cum = np.cumsum(fps)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-10)
            recalls = tp_cum / max(total_gts, 1)
            ap = _average_precision(recalls, precisions)
            return {
                f"AP@{self.iou_thr:.2f}": ap,
                "precision": float(precisions[-1]),
                "recall": float(recalls[-1]),
            }
