"""Whole-video screening and 50k-style outlier detection for `jabs.bale`.

Scans every non-excluded video in a JABS project using only compact prediction HDF5s
(`jabs/predictions/*.h5`) and annotation JSONs (`jabs/annotations/*.json`) — zero Parquet I/O —
to compute whole-video behavior frames, `% time`, bout counts, longest bout duration,
zero-behavior status, low-confidence `Behavior` (`E3a`), low-confidence `Not-Behavior`
near-misses (`E3b`), postprocessing fragility (`E4`), and multi-class chiral asymmetry (`E1`).
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierMode
from jabs.io.annotations import read_document
from jabs.project.track_labels import TrackLabels

from .bout_extractor import (
    extract_video_group_id,
    resolve_project_video_name,
    strip_known_video_suffix,
)
from .prediction_extractor import _contiguous_true_runs, _lookup_identity_arr

if TYPE_CHECKING:
    from jabs.project import Project

logger = logging.getLogger(__name__)

VideoMetricSource = Literal["predictions", "labels", "predictions_unlabeled"]
VideoScreeningPreset = Literal[
    "all",
    "any_outlier",
    "high_frames",
    "high_bouts",
    "long_bout",
    "low_conf_pos",
    "low_conf_none",
    "fragile",
    "zero_all",
    "zero_high_conf",
    "unlabeled_with_behavior",
    "labeled_only",
]

_DEFAULT_CAGE_RE = re.compile(r"(cage_\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class VideoFocalEvent:
    """Focal `(identity, start_frame, end_frame, reason)` jump anchor inside a video."""

    identity: int
    start_frame: int
    end_frame: int
    reason: str


@dataclass
class VideoSummaryRecord:
    """Whole-video behavior, bout, confidence, and 50k-style outlier summary for one video."""

    video: str
    cage: str = ""
    n_animals: int = 0
    max_track_frames: int = 0
    total_frames: int = 0
    valid_frames: int = 0
    pose_valid_frac: float = 1.0

    # Whole-video prediction statistics
    has_predictions: bool = False
    pred_pos_frames: int = 0
    pred_pos_pct: float = 0.0
    pred_bout_count: int = 0
    pred_bouts_per_min: float = 0.0
    pred_max_bout_frames: int = 0
    pred_mean_bout_frames: float = 0.0
    pred_mean_prob: float = float("nan")
    pred_min_bout_prob: float = float("nan")
    pred_mean_none_prob: float = float("nan")
    pred_low_conf_pos_bouts: int = 0
    pred_low_conf_pos_frames: int = 0
    pred_low_conf_none_spans: int = 0
    pred_low_conf_none_frames: int = 0
    pred_min_none_prob: float = float("nan")
    pred_postproc_fragility: float = 0.0
    pred_chiral_asymmetry: float = float("nan")
    pred_class_switch_count: int = 0
    unlabeled_pred_frames: int = 0
    unlabeled_pred_bouts: int = 0
    unlabeled_pred_max_bout_frames: int = 0
    pred_label_disagree_frames: int = 0
    pred_label_disagree_rate: float = 0.0

    # Whole-video human-label statistics
    has_labels: bool = False
    label_pos_frames: int = 0
    label_neg_frames: int = 0
    label_pos_pct: float = 0.0
    label_pos_bouts: int = 0
    label_neg_bouts: int = 0
    label_max_bout_frames: int = 0
    label_dual_class_tracks: int = 0

    # 50k-style cohort outlier screening tags (recomputed cleanly per active metric source)
    outlier_tags: list[str] = field(default_factory=list)
    primary_bucket: str = "Normal"
    outlier_score: float = 0.0

    # Jump anchors keyed by preset/reason ("default", "longest", "low_conf_pos", "low_conf_none", "label")
    focal_events: dict[str, VideoFocalEvent] = field(default_factory=dict)

    # Lightweight per-track timeline spans for fast whole-video overview plots in GUI
    # identity -> {"pred_pos": [(s, e), ...], "low_conf_pos": [...], "low_conf_none": [...],
    #              "label_pos": [...], "label_neg": [...]}
    track_spans: dict[int, dict[str, list[tuple[int, int]]]] = field(default_factory=dict)
    # Multi-class per-track predicted spans keyed by class name for Set1 timeline coloring:
    # identity -> {class_name: [(s, e), ...]}
    pred_by_class: dict[int, dict[str, list[tuple[int, int]]]] = field(default_factory=dict)

    def focal_event_for(
        self,
        preset: VideoScreeningPreset = "all",
        source: VideoMetricSource = "predictions",
    ) -> VideoFocalEvent:
        """Return the most relevant `(identity, start, end, reason)` anchor for `preset`."""
        if source == "labels":
            if "label" in self.focal_events:
                return self.focal_events["label"]
            return VideoFocalEvent(0, 0, 0, "Video start (no human labels)")

        key_order: list[str] = []
        if preset == "low_conf_none":
            key_order = ["low_conf_none", "low_conf_pos", "longest", "default"]
        elif preset == "low_conf_pos":
            key_order = ["low_conf_pos", "low_conf_none", "longest", "default"]
        elif preset == "long_bout":
            key_order = ["longest", "default"]
        elif preset in ("zero_all", "zero_high_conf"):
            key_order = ["low_conf_none", "label", "default"]
        elif preset == "labeled_only":
            key_order = ["label", "default"]
        else:
            key_order = ["default", "low_conf_pos", "low_conf_none", "longest", "label"]

        for k in key_order:
            ev = self.focal_events.get(k)
            if ev is not None:
                return ev
        return VideoFocalEvent(0, 0, 0, "Video start (frame 0)")

    def active_frames(self, source: VideoMetricSource) -> int:
        """Return whole-video behavior frame count for `source`."""
        if source == "labels":
            return int(self.label_pos_frames)
        if source == "predictions_unlabeled":
            return int(self.unlabeled_pred_frames)
        return int(self.pred_pos_frames)

    def active_pct(self, source: VideoMetricSource) -> float:
        """Return whole-video behavior percentage of valid frames for `source`."""
        if source == "labels":
            return float(self.label_pos_pct)
        if source == "predictions_unlabeled":
            if self.valid_frames > 0:
                return 100.0 * float(self.unlabeled_pred_frames) / float(self.valid_frames)
            return 0.0
        return float(self.pred_pos_pct)

    def active_bouts(self, source: VideoMetricSource) -> int:
        """Return whole-video behavior bout count for `source`."""
        if source == "labels":
            return int(self.label_pos_bouts)
        if source == "predictions_unlabeled":
            return int(self.unlabeled_pred_bouts)
        return int(self.pred_bout_count)

    def active_max_bout(self, source: VideoMetricSource) -> int:
        """Return longest single bout duration (frames) for `source`."""
        if source == "labels":
            return int(self.label_max_bout_frames)
        if source == "predictions_unlabeled":
            return int(self.unlabeled_pred_max_bout_frames)
        return int(self.pred_max_bout_frames)


@dataclass
class VideoScreenerReport:
    """Complete whole-video screening summary across a JABS project."""

    project_dir: str
    behavior: str
    is_multiclass: bool
    include_all_multiclass: bool
    use_postprocessed: bool
    records: list[VideoSummaryRecord]
    has_any_predictions: bool
    has_any_labels: bool
    elapsed_ms: float = 0.0
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_dataframe(self, source: VideoMetricSource = "predictions") -> pd.DataFrame:
        """Export whole-video summary metrics and 50k screening buckets as a DataFrame."""
        rows: list[dict[str, Any]] = []
        for rec in self.records:
            rows.append(
                {
                    "video": rec.video,
                    "cage": rec.cage,
                    "n_animals": rec.n_animals,
                    "valid_frames": rec.valid_frames,
                    "pose_valid_frac": round(float(rec.pose_valid_frac), 4),
                    "has_predictions": rec.has_predictions,
                    "pred_pos_frames": rec.pred_pos_frames,
                    "pred_pos_pct": round(float(rec.pred_pos_pct), 3),
                    "pred_bout_count": rec.pred_bout_count,
                    "pred_bouts_per_min": round(float(rec.pred_bouts_per_min), 3),
                    "pred_max_bout_frames": rec.pred_max_bout_frames,
                    "pred_mean_bout_frames": round(float(rec.pred_mean_bout_frames), 2),
                    "pred_mean_prob": (
                        round(float(rec.pred_mean_prob), 4)
                        if math.isfinite(rec.pred_mean_prob)
                        else float("nan")
                    ),
                    "pred_min_bout_prob": (
                        round(float(rec.pred_min_bout_prob), 4)
                        if math.isfinite(rec.pred_min_bout_prob)
                        else float("nan")
                    ),
                    "pred_mean_none_prob": (
                        round(float(rec.pred_mean_none_prob), 4)
                        if math.isfinite(rec.pred_mean_none_prob)
                        else float("nan")
                    ),
                    "pred_low_conf_pos_bouts": rec.pred_low_conf_pos_bouts,
                    "pred_low_conf_pos_frames": rec.pred_low_conf_pos_frames,
                    "pred_low_conf_none_spans": rec.pred_low_conf_none_spans,
                    "pred_low_conf_none_frames": rec.pred_low_conf_none_frames,
                    "pred_min_none_prob": (
                        round(float(rec.pred_min_none_prob), 4)
                        if math.isfinite(rec.pred_min_none_prob)
                        else float("nan")
                    ),
                    "pred_postproc_fragility": round(float(rec.pred_postproc_fragility), 4),
                    "pred_chiral_asymmetry": (
                        round(float(rec.pred_chiral_asymmetry), 4)
                        if math.isfinite(rec.pred_chiral_asymmetry)
                        else float("nan")
                    ),
                    "pred_class_switch_count": rec.pred_class_switch_count,
                    "has_labels": rec.has_labels,
                    "label_pos_frames": rec.label_pos_frames,
                    "label_neg_frames": rec.label_neg_frames,
                    "label_pos_pct": round(float(rec.label_pos_pct), 3),
                    "label_pos_bouts": rec.label_pos_bouts,
                    "label_neg_bouts": rec.label_neg_bouts,
                    "label_max_bout_frames": rec.label_max_bout_frames,
                    "label_dual_class_tracks": rec.label_dual_class_tracks,
                    "unlabeled_pred_frames": rec.unlabeled_pred_frames,
                    "unlabeled_pred_bouts": rec.unlabeled_pred_bouts,
                    "pred_label_disagree_rate": round(float(rec.pred_label_disagree_rate), 4),
                    "active_source": source,
                    "primary_bucket": rec.primary_bucket,
                    "outlier_tags": ", ".join(rec.outlier_tags),
                    "outlier_score": round(float(rec.outlier_score), 4),
                }
            )
        return pd.DataFrame(rows)


def _load_video_annotation_details(
    project: Project,
    behavior: str | None,
    is_multiclass: bool,
    include_all_multiclass: bool,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Return per-video, per-identity human label spans and frame counts.

    Output layout:
        `details[video_name][identity] = {
            "pos_spans": [(s, e), ...],
            "neg_spans": [(s, e), ...],
            "num_frames": int,
        }`
    """
    project_dir = Path(project.project_paths.project_dir)
    anno_dir = project_dir / "jabs" / "annotations"
    sm = project.settings_manager
    behavior_names = list(sm.behavior_names)
    eff_beh = behavior or (behavior_names[0] if behavior_names else "")

    details: dict[str, dict[int, dict[str, Any]]] = {}

    def _merge_label_dict(video_name: str, label_dict: Any, num_frames: int) -> None:
        if not isinstance(label_dict, dict):
            return
        vid_map = details.setdefault(video_name, {})
        for id_str, beh_map in label_dict.items():
            try:
                ident = int(id_str)
            except (TypeError, ValueError):
                continue
            if not isinstance(beh_map, dict):
                continue
            entry = vid_map.setdefault(
                ident,
                {"pos_spans": [], "neg_spans": [], "num_frames": max(0, int(num_frames))},
            )
            if num_frames > entry["num_frames"]:
                entry["num_frames"] = int(num_frames)
            if is_multiclass:
                pos_keys = (
                    [b for b in behavior_names if b != MULTICLASS_NONE_BEHAVIOR]
                    if (include_all_multiclass or not eff_beh)
                    else [eff_beh]
                )
                for b_key in pos_keys:
                    for blk in beh_map.get(b_key, []) or []:
                        if isinstance(blk, dict) and blk.get("present", False):
                            s, e = int(blk["start"]), int(blk["end"])
                            if e >= s:
                                entry["pos_spans"].append((s, e))
                # Explicit "None" or other classes serve as negative frames for single-target view
                neg_keys = [MULTICLASS_NONE_BEHAVIOR]
                if not include_all_multiclass and eff_beh:
                    neg_keys.extend(b for b in behavior_names if b != eff_beh)
                for n_key in neg_keys:
                    for blk in beh_map.get(n_key, []) or []:
                        if isinstance(blk, dict) and blk.get("present", False):
                            s, e = int(blk["start"]), int(blk["end"])
                            if e >= s:
                                entry["neg_spans"].append((s, e))
            else:
                if eff_beh and eff_beh in beh_map:
                    for blk in beh_map.get(eff_beh, []) or []:
                        if not isinstance(blk, dict):
                            continue
                        s, e = int(blk["start"]), int(blk["end"])
                        if e < s:
                            continue
                        if blk.get("present", False):
                            entry["pos_spans"].append((s, e))
                        else:
                            entry["neg_spans"].append((s, e))

    seen_json_videos: set[str] = set()
    if anno_dir.is_dir():
        for anno_path in sorted(anno_dir.glob("*.json")):
            # Fast-skip excluded videos before parsing JSON on disk
            cand_name = resolve_project_video_name(project, anno_path.name)
            if sm.is_video_excluded(cand_name):
                continue
            doc = read_document(anno_path)
            if not doc:
                continue
            video_name = resolve_project_video_name(project, anno_path.name, doc=doc)
            if sm.is_video_excluded(video_name):
                continue
            seen_json_videos.add(video_name)
            n_frames = int(doc.get("num_frames") or 0)
            lbl_dict = doc.get("unfragmented_labels") or doc.get("labels") or {}
            _merge_label_dict(video_name, lbl_dict, n_frames)

    # Also inspect live/synthetic VideoManager labels if JSON files were not written
    try:
        vm = getattr(project, "video_manager", None)
        load_vl = getattr(vm, "load_video_labels", None) if vm is not None else None
        raw_videos = getattr(vm, "videos", None) if vm is not None else None
        if callable(load_vl) and isinstance(raw_videos, list | tuple | set):
            from jabs.project.video_manager import VideoManager

            is_real_vm = isinstance(vm, VideoManager)
            for v in raw_videos:
                v_name = Path(str(v)).name
                if is_real_vm and v_name in seen_json_videos:
                    continue
                try:
                    vl = load_vl(v_name)
                except Exception:
                    continue
                if vl is None or not hasattr(vl, "iter_identity_behavior_labels"):
                    continue
                live_dict: dict[str, dict[str, list[dict[str, Any]]]] = {}
                max_n = 0
                try:
                    for id_str, beh_key, tl in vl.iter_identity_behavior_labels():
                        blocks = tl.get_blocks()
                        if blocks:
                            live_dict.setdefault(str(id_str), {})[str(beh_key)] = blocks
                            max_n = max(max_n, int(getattr(tl, "num_frames", 0) or 0))
                except Exception:
                    continue
                if live_dict:
                    _merge_label_dict(v_name, live_dict, max_n)
    except Exception:
        pass

    # Deduplicate & sort spans cleanly
    for vid_map in details.values():
        for entry in vid_map.values():
            entry["pos_spans"] = sorted(set(entry["pos_spans"]))
            entry["neg_spans"] = sorted(set(entry["neg_spans"]))
    return details


def _spans_to_mask(spans: list[tuple[int, int]], n_frames: int) -> np.ndarray:
    mask = np.zeros(max(0, int(n_frames)), dtype=bool)
    for s, e in spans:
        lo = max(0, int(s))
        hi = min(n_frames, int(e) + 1)
        if hi > lo:
            mask[lo:hi] = True
    return mask


def _dilate_bool_mask(mask: np.ndarray, pad: int) -> np.ndarray:
    """Dilate a 1D boolean mask by `pad` frames on each side to exclude bout-edge shoulders."""
    if pad <= 0 or mask.size == 0 or not np.any(mask):
        return mask
    kernel = np.ones(2 * pad + 1, dtype=np.int32)
    return np.convolve(mask.astype(np.int32, copy=False), kernel, mode="same") > 0


def _binary_prob_confidence(
    raw_pred: np.ndarray,
    prob_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve JABS binary `.h5` confidence array into `(p_pos, conf_pred, margin, prob_finite)`.

    Handles BOTH:
    - Real JABS `.h5` schema (`classify.py:277`), where `prob_arr[t]` stores the
      confidence of the predicted class (`>= 0.50` on all valid frames, so on `raw_pred == 0`
      frames `prob_arr[t]` is already `P(Not-Behavior)`), AND
    - Raw positive-probability test/export arrays where `prob_arr[t]` is `P(Behavior) in [0, 1]`
      (so on `raw_pred == 0` frames `prob_arr[t] < 0.50`).

    Non-finite (`NaN`/`Inf`) probability entries are flagged in `prob_finite` (`False`) and
    kept as `NaN` rather than coerced to `0.50`, so missing probabilities never fire fake
    `E3a`/`E3b` near-miss alerts.
    """
    p_raw = np.asarray(prob_arr, dtype=np.float32)
    prob_finite = np.isfinite(p_raw)
    p = np.where(prob_finite, np.clip(p_raw, 0.0, 1.0), np.nan).astype(np.float32)
    is_pos_raw = raw_pred == TrackLabels.Label.BEHAVIOR.value
    is_neg_raw = raw_pred == TrackLabels.Label.NOT_BEHAVIOR.value

    # On raw_pred == 0 frames: if p < 0.50, it's P(Beh) -> P(None) = 1 - p; else already P(None)
    conf_on_neg = np.where(prob_finite & (p < 0.5), 1.0 - p, p)
    # On raw_pred == 1 frames: P(Beh) is p
    p_pos = np.where(
        prob_finite,
        np.where(is_pos_raw, p, np.where(is_neg_raw, 1.0 - conf_on_neg, 0.0)),
        np.nan,
    ).astype(np.float32)
    conf_pred = np.where(
        prob_finite,
        np.where(is_pos_raw, p, np.where(is_neg_raw, conf_on_neg, 0.0)),
        np.nan,
    ).astype(np.float32)
    margin = np.where(prob_finite, np.abs(2.0 * p_pos - 1.0), np.nan).astype(np.float32)
    return p_pos, conf_pred, margin, prob_finite


def classify_video_outliers(
    records: list[VideoSummaryRecord],
    *,
    source: VideoMetricSource = "predictions",
    high_percentile: float = 90.0,
) -> dict[str, float]:
    """Assign 50k-style outlier bucket tags (`E1`-`E5`) and composite scores across `records`.

    Mutates `rec.outlier_tags`, `rec.primary_bucket`, and `rec.outlier_score` in place and
    returns the cohort thresholds used (e.g., `frames_p90`, `bouts_p90`, `max_bout_p90`).
    """
    if not records:
        return {"n_videos": 0.0, "n_active_videos": 0.0}

    frames_vals = np.array([rec.active_frames(source) for rec in records], dtype=np.float64)
    pct_vals = np.array([rec.active_pct(source) for rec in records], dtype=np.float64)
    bout_vals = np.array([rec.active_bouts(source) for rec in records], dtype=np.float64)
    max_dur_vals = np.array([rec.active_max_bout(source) for rec in records], dtype=np.float64)

    active_mask = frames_vals > 0
    n_active = int(np.count_nonzero(active_mask))
    eff_pct = high_percentile if n_active >= 8 else (80.0 if n_active >= 4 else 75.0)

    if n_active >= 3:
        frames_thr = max(1.0, float(np.percentile(frames_vals[active_mask], eff_pct)))
        pct_thr = max(0.01, float(np.percentile(pct_vals[active_mask], eff_pct)))
        bouts_thr = max(2.0, float(np.percentile(bout_vals[active_mask], eff_pct)))
        max_dur_thr = max(15.0, float(np.percentile(max_dur_vals[active_mask], eff_pct)))
    elif n_active > 0:
        frames_thr = max(1.0, float(np.max(frames_vals[active_mask])))
        pct_thr = max(0.01, float(np.max(pct_vals[active_mask])))
        bouts_thr = max(2.0, float(np.max(bout_vals[active_mask])))
        max_dur_thr = max(15.0, float(np.max(max_dur_vals[active_mask])))
    else:
        frames_thr = 1.0
        pct_thr = 1.0
        bouts_thr = 2.0
        max_dur_thr = 15.0

    # Normalize ranks for composite outlier score
    f_max = max(1.0, float(np.max(frames_vals)))
    b_max = max(1.0, float(np.max(bout_vals)))
    d_max = max(1.0, float(np.max(max_dur_vals)))

    # Adaptive weak-confidence tail for projects whose postprocessing already filters `P < 0.80`
    finite_min_probs = [
        float(r.pred_min_bout_prob)
        for r in records
        if r.has_predictions and r.pred_bout_count > 0 and math.isfinite(r.pred_min_bout_prob)
    ]
    e3a_explicit_recs = [
        r for r in records if r.pred_low_conf_pos_bouts > 0 or r.pred_class_switch_count >= 2
    ]
    any_explicit_e3a = bool(e3a_explicit_recs)
    adaptive_e3a_thr = (
        min(0.88, float(np.percentile(finite_min_probs, 10.0)))
        if (not any_explicit_e3a and len(finite_min_probs) >= 5)
        else 0.72
    )
    # Cohort-tail gating on large cohorts (>=25 videos) if >25% fire explicit E3a/E3b so the
    # `50k Outliers` spotlight stays focused on the genuinely anomalous top tail while raw
    # counts (`pred_low_conf_pos_bouts`, `pred_low_conf_none_spans`) remain intact for sorting.
    n_tot = len(records)
    gate_e3a_tail = n_tot >= 25 and (len(e3a_explicit_recs) / n_tot) > 0.25
    e3a_frames_thr = (
        float(np.percentile([r.pred_low_conf_pos_frames for r in e3a_explicit_recs], 75.0))
        if gate_e3a_tail
        else 0.0
    )
    e3b_explicit_recs = [r for r in records if r.pred_low_conf_none_spans > 0]
    gate_e3b_tail = n_tot >= 25 and (len(e3b_explicit_recs) / n_tot) > 0.25
    e3b_frames_thr = (
        float(np.percentile([r.pred_low_conf_none_frames for r in e3b_explicit_recs], 75.0))
        if gate_e3b_tail
        else 0.0
    )

    for idx, rec in enumerate(records):
        tags: list[str] = []
        f_val = float(frames_vals[idx])
        p_val = float(pct_vals[idx])
        b_val = float(bout_vals[idx])
        d_val = float(max_dur_vals[idx])
        use_pred = source != "labels"

        has_e3b_hit = (
            use_pred
            and rec.pred_low_conf_none_spans > 0
            and (
                (not gate_e3b_tail)
                or rec.pred_low_conf_none_frames >= max(30.0, e3b_frames_thr)
                or (math.isfinite(rec.pred_min_none_prob) and rec.pred_min_none_prob <= 0.58)
            )
        )

        if f_val <= 0 and b_val <= 0:
            # Under `predictions_unlabeled`, a fully-labeled video with predicted behavior is NOT
            # a zero-behavior control — leave it as Normal rather than tagging E5_zero_control.
            is_all_labeled_with_behavior = source == "predictions_unlabeled" and (
                rec.pred_pos_frames > 0 or rec.pred_bout_count > 0
            )
            if not is_all_labeled_with_behavior:
                if use_pred and rec.has_predictions and has_e3b_hit:
                    tags.append("E3b_low_conf_none")
                    tags.append("E5_zero_control")
                else:
                    tags.append("E5_zero_control")
        else:
            if (f_val >= frames_thr or p_val >= pct_thr) and n_active >= 2:
                tags.append("E1_high_frames")
            if b_val >= bouts_thr and n_active >= 2:
                tags.append("E1_high_bouts")
            if (
                use_pred
                and math.isfinite(rec.pred_chiral_asymmetry)
                and rec.pred_chiral_asymmetry >= 0.70
                and rec.pred_bout_count >= 4
            ):
                tags.append("E1_chiral_asym")
            if d_val >= max_dur_thr and n_active >= 2:
                tags.append("E2_long_bout")
            if use_pred and (
                (
                    rec.pred_low_conf_pos_bouts > 0
                    and (
                        (not gate_e3a_tail)
                        or rec.pred_low_conf_pos_frames >= max(30.0, e3a_frames_thr)
                        or (
                            math.isfinite(rec.pred_min_bout_prob)
                            and rec.pred_min_bout_prob <= 0.62
                        )
                    )
                )
                or rec.pred_class_switch_count >= 2
                or (
                    not any_explicit_e3a
                    and rec.pred_bout_count > 0
                    and math.isfinite(rec.pred_min_bout_prob)
                    and rec.pred_min_bout_prob <= adaptive_e3a_thr
                )
            ):
                tags.append("E3a_low_conf_pos")
            if has_e3b_hit:
                tags.append("E3b_low_conf_none")

        if use_pred:
            if rec.pred_postproc_fragility >= 0.35:
                tags.append("E4_fragile")
            if rec.pred_label_disagree_rate >= 0.35:
                tags.append("E4_pred_label_disagree")
            if rec.has_predictions and rec.pose_valid_frac < 0.75:
                tags.append("E4_low_pose_valid")

        has_e4 = any(t.startswith("E4_") for t in tags)
        rec.outlier_tags = tags

        # Assign clean human-readable primary bucket label (prioritizing whole-video E1/E2 extremes)
        if "E3b_low_conf_none" in tags and "E5_zero_control" in tags:
            rec.primary_bucket = "E5 Zero (+ Near-Miss)"
        elif has_e4 and "E5_zero_control" in tags:
            rec.primary_bucket = "E4 Fragile / Disagree"
        elif "E5_zero_control" in tags:
            if (
                use_pred
                and rec.has_predictions
                and math.isfinite(rec.pred_mean_none_prob)
                and rec.pred_mean_none_prob >= 0.90
            ):
                rec.primary_bucket = "E5 Zero (High-Conf)"
            else:
                rec.primary_bucket = "E5 Zero Behavior"
        elif "E1_high_frames" in tags and "E1_high_bouts" in tags:
            rec.primary_bucket = "E1 High Frames & Bouts"
        elif "E1_high_frames" in tags:
            rec.primary_bucket = "E1 High Frames"
        elif "E1_high_bouts" in tags:
            rec.primary_bucket = "E1 High Bouts"
        elif "E1_chiral_asym" in tags:
            rec.primary_bucket = "E1 Chiral Asymmetry"
        elif "E2_long_bout" in tags:
            rec.primary_bucket = "E2 Long Bout"
        elif "E3a_low_conf_pos" in tags and "E3b_low_conf_none" in tags:
            rec.primary_bucket = "E3a+E3b Uncertain"
        elif "E3a_low_conf_pos" in tags:
            rec.primary_bucket = "E3a Low-Conf Beh"
        elif "E3b_low_conf_none" in tags:
            rec.primary_bucket = "E3b Near-Miss None"
        elif has_e4:
            rec.primary_bucket = "E4 Fragile / Disagree"
        else:
            rec.primary_bucket = "Normal"

        # Composite 0..1 outlier priority score (higher = stronger screening hit)
        score = 0.0
        if "E3a_low_conf_pos" in tags:
            weak_p = (
                1.0 - float(rec.pred_min_bout_prob)
                if math.isfinite(rec.pred_min_bout_prob)
                else 0.35
            )
            score = max(score, min(1.0, 0.55 + 0.90 * weak_p))
        if "E3b_low_conf_none" in tags:
            weak_n = (
                1.0 - float(rec.pred_min_none_prob)
                if math.isfinite(rec.pred_min_none_prob)
                else 0.35
            )
            score = max(score, min(1.0, 0.55 + 0.90 * weak_n))
        if has_e4:
            e4_sev = max(
                rec.pred_postproc_fragility,
                rec.pred_label_disagree_rate,
                max(0.0, 1.0 - rec.pose_valid_frac) if rec.has_predictions else 0.0,
            )
            score = max(score, min(1.0, 0.50 + 0.50 * e4_sev))
        if "E1_high_frames" in tags or "E1_high_bouts" in tags or "E1_chiral_asym" in tags:
            e1_s = max(f_val / f_max, b_val / b_max)
            score = max(score, 0.55 + 0.40 * e1_s)
        if "E2_long_bout" in tags:
            score = max(score, 0.50 + 0.42 * (d_val / d_max))
        if score == 0.0 and "E5_zero_control" not in tags:
            score = 0.25 * max(f_val / f_max, b_val / b_max)
        rec.outlier_score = float(min(1.0, max(0.0, score)))

    return {
        "n_videos": float(len(records)),
        "frames_high_thr": frames_thr,
        "pct_high_thr": pct_thr,
        "bouts_high_thr": bouts_thr,
        "max_bout_high_thr": max_dur_thr,
        "n_active_videos": float(n_active),
    }


def filter_video_records(
    records: list[VideoSummaryRecord],
    *,
    preset: VideoScreeningPreset = "all",
    source: VideoMetricSource = "predictions",
    unlabeled_only: bool = False,
    text_query: str = "",
) -> list[VideoSummaryRecord]:
    """Filter `records` by 50k screening preset, unlabeled toggle, and text search."""
    q = text_query.strip().lower()
    out: list[VideoSummaryRecord] = []
    use_pred = source != "labels"
    has_any_e1_frames = any("E1_high_frames" in r.outlier_tags for r in records)
    has_any_e1_bouts = any("E1_high_bouts" in r.outlier_tags for r in records)

    for rec in records:
        if unlabeled_only and rec.has_labels:
            continue
        if q:
            hay = " ".join(
                (
                    rec.video.lower(),
                    rec.cage.lower(),
                    rec.primary_bucket.lower(),
                    " ".join(t.lower() for t in rec.outlier_tags),
                )
            )
            if q not in hay:
                continue

        tags = set(rec.outlier_tags)
        f_val = rec.active_frames(source)
        b_val = rec.active_bouts(source)
        is_zero = f_val <= 0 and b_val <= 0
        if source == "predictions_unlabeled" and (
            rec.pred_pos_frames > 0 or rec.pred_bout_count > 0
        ):
            is_zero = False

        if preset == "all":
            keep = True
        elif preset == "any_outlier":
            keep = bool(
                tags
                & {
                    "E1_high_frames",
                    "E1_high_bouts",
                    "E1_chiral_asym",
                    "E2_long_bout",
                    "E3a_low_conf_pos",
                    "E3b_low_conf_none",
                    "E4_fragile",
                    "E4_pred_label_disagree",
                    "E4_low_pose_valid",
                }
            )
        elif preset == "high_frames":
            keep = "E1_high_frames" in tags or (f_val > 0 and not has_any_e1_frames)
        elif preset == "high_bouts":
            keep = "E1_high_bouts" in tags or (b_val > 0 and not has_any_e1_bouts)
        elif preset == "long_bout":
            keep = "E2_long_bout" in tags
        elif preset == "low_conf_pos":
            keep = use_pred and ("E3a_low_conf_pos" in tags or rec.pred_low_conf_pos_bouts > 0)
        elif preset == "low_conf_none":
            keep = use_pred and ("E3b_low_conf_none" in tags or rec.pred_low_conf_none_spans > 0)
        elif preset == "fragile":
            keep = use_pred and any(t.startswith("E4_") for t in tags)
        elif preset == "zero_all":
            keep = is_zero
        elif preset == "zero_high_conf":
            keep = (
                use_pred
                and is_zero
                and rec.has_predictions
                and rec.pred_low_conf_none_spans == 0
                and not any(t.startswith("E4_") for t in tags)
                and math.isfinite(rec.pred_mean_none_prob)
                and rec.pred_mean_none_prob >= 0.90
            )
        elif preset == "unlabeled_with_behavior":
            keep = (not rec.has_labels) and (rec.pred_pos_frames > 0 or rec.pred_bout_count > 0)
        elif preset == "labeled_only":
            keep = rec.has_labels
        else:
            keep = True

        if keep:
            out.append(rec)
    return out


def summarize_project_videos(
    project: Project,
    behavior: str | None = None,
    *,
    include_all_multiclass: bool = False,
    use_postprocessed: bool = True,
    min_bout_frames: int = 2,
    min_near_miss_frames: int = 12,
    beh_low_max: float = 0.72,
    none_low_max: float = 0.65,
    margin_low_max: float = 0.16,
    shoulder_pad_frames: int = 6,
    source: VideoMetricSource = "predictions",
    cancel_event: threading.Event | None = None,
) -> VideoScreenerReport:
    """Compute whole-video behavior counts and 50k-style outlier metrics for every project video.

    Reads only `jabs/annotations/*.json` and `jabs/predictions/*.h5` (zero Parquet I/O).
    Every non-excluded project video appears in `report.records`, including videos with
    zero behavior frames / zero bouts.
    """
    t0 = time.perf_counter()
    project_dir = Path(project.project_paths.project_dir)
    sm = project.settings_manager
    is_multiclass = sm.classifier_mode == ClassifierMode.MULTICLASS
    behavior_names = list(sm.behavior_names)
    active_behavior = behavior or (behavior_names[0] if behavior_names else "")
    eff_all_mc = bool(is_multiclass and (include_all_multiclass or behavior is None))

    # Optional project cage-grouping regex for clean cage tagging
    raw_cage_re = getattr(sm, "cv_grouping_regex", "") or ""
    cage_regex_str = str(raw_cage_re) if raw_cage_re else None

    # 1. Build O(1) video-name index once & gather all non-excluded project videos
    all_videos_ordered: list[str] = []
    seen_stems: set[str] = set()

    def _add_video(v_name: str) -> None:
        if not v_name or sm.is_video_excluded(v_name):
            return
        stem = strip_known_video_suffix(v_name)
        if stem in seen_stems:
            return
        seen_stems.add(stem)
        all_videos_ordered.append(v_name)

    try:
        for v in getattr(project.video_manager, "videos", []) or []:
            _add_video(Path(str(v)).name)
    except Exception:
        pass

    anno_details = _load_video_annotation_details(
        project,
        active_behavior if not eff_all_mc else None,
        is_multiclass,
        eff_all_mc,
    )
    for v_name in sorted(anno_details.keys()):
        _add_video(v_name)

    pred_dir = project_dir / "jabs" / "predictions"
    pred_stems_present: set[str] = set()
    if pred_dir.is_dir():
        for p_h5 in sorted(pred_dir.glob("*.h5")):
            v_name = resolve_project_video_name(project, p_h5.name)
            _add_video(v_name)
            pred_stems_present.add(strip_known_video_suffix(v_name))

    all_videos_ordered.sort()

    records: list[VideoSummaryRecord] = []
    has_any_preds = False
    has_any_labels = False

    for video_name in all_videos_ordered:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Video screener cancelled")
        rec = VideoSummaryRecord(
            video=video_name,
            cage=extract_video_group_id(video_name, cage_regex_str),
        )
        vid_stem = strip_known_video_suffix(video_name)
        vid_anno = anno_details.get(video_name, {})

        # Track-level human label accumulation
        label_identities: set[int] = set()
        best_label_span: tuple[int, int, int, int] | None = None  # (dur, ident, s, e)
        label_track_total_frames = 0

        for ident, info in sorted(vid_anno.items()):
            pos_spans: list[tuple[int, int]] = info["pos_spans"]
            neg_spans: list[tuple[int, int]] = info["neg_spans"]
            n_f_ann = int(info.get("num_frames") or 0)
            if pos_spans or neg_spans:
                label_identities.add(ident)
                rec.has_labels = True
                has_any_labels = True
            pos_f = sum(e - s + 1 for s, e in pos_spans)
            neg_f = sum(e - s + 1 for s, e in neg_spans)
            rec.label_pos_frames += pos_f
            rec.label_neg_frames += neg_f
            rec.label_pos_bouts += len(pos_spans)
            rec.label_neg_bouts += len(neg_spans)
            if pos_f >= 10 and neg_f >= 10:
                rec.label_dual_class_tracks += 1
            if pos_spans:
                longest_s, longest_e = max(pos_spans, key=lambda se: se[1] - se[0] + 1)
                dur_l = longest_e - longest_s + 1
                if dur_l > rec.label_max_bout_frames:
                    rec.label_max_bout_frames = dur_l
                if best_label_span is None or dur_l > best_label_span[0]:
                    best_label_span = (dur_l, ident, longest_s, longest_e)
            elif neg_spans and best_label_span is None:
                s0, e0 = neg_spans[0]
                best_label_span = (0, ident, s0, e0)

            max_span_end = max([e + 1 for _, e in pos_spans] + [e + 1 for _, e in neg_spans] + [0])
            tr_len = max(n_f_ann, max_span_end)
            if tr_len > rec.max_track_frames:
                rec.max_track_frames = tr_len
            label_track_total_frames += tr_len

            if pos_spans or neg_spans:
                ts = rec.track_spans.setdefault(
                    ident,
                    {
                        "pred_pos": [],
                        "low_conf_pos": [],
                        "low_conf_none": [],
                        "label_pos": [],
                        "label_neg": [],
                    },
                )
                ts["label_pos"].extend(pos_spans)
                ts["label_neg"].extend(neg_spans)

        if best_label_span is not None:
            _, b_id, b_s, b_e = best_label_span
            rec.focal_events["label"] = VideoFocalEvent(
                b_id,
                b_s,
                b_e,
                f"Longest labeled bout ({b_e - b_s + 1}f)",
            )

        # Prediction scanning (if a prediction file exists or PredictionManager returns data)
        pred_identities: set[int] = set()
        pose_valid_total = 0
        pos_prob_sum = 0.0
        pos_prob_count = 0
        fallback_pos_prob_sum = 0.0
        fallback_pos_prob_count = 0
        none_prob_sum = 0.0
        none_prob_count = 0
        raw_or_post_pos_union = 0
        raw_vs_post_diff = 0
        labeled_overlap_frames = 0
        disagree_frames = 0
        bout_lengths: list[int] = []
        class_bout_counts: dict[int, int] = {}

        best_longest: tuple[int, int, int, int] | None = None  # (dur, ident, s, e)
        best_low_pos: tuple[float, int, int, int] | None = None  # (mean_p, ident, s, e)
        best_low_none: tuple[float, int, int, int] | None = None  # (mean_p_none, ident, s, e)

        try_pred = (not pred_dir.is_dir()) or (vid_stem in pred_stems_present)
        if try_pred:
            if is_multiclass:
                preds_dict: dict[Any, Any] = {}
                probs_dict: dict[Any, Any] = {}
                post_dict: dict[Any, Any] = {}
                class_names: list[str] = []
                try:
                    preds_dict, probs_dict, post_dict, class_names = (
                        project.prediction_manager.load_multiclass_predictions(video_name)
                    )
                except Exception:
                    preds_dict = {}

                if preds_dict and class_names:
                    none_idx = (
                        class_names.index(MULTICLASS_NONE_BEHAVIOR)
                        if MULTICLASS_NONE_BEHAVIOR in class_names
                        else 0
                    )
                    target_indices: list[int] = []
                    if eff_all_mc:
                        target_indices = [
                            i for i, c in enumerate(class_names) if c != MULTICLASS_NONE_BEHAVIOR
                        ]
                    elif active_behavior in class_names:
                        target_indices = [class_names.index(active_behavior)]

                    if target_indices:
                        target_set = set(target_indices)
                        for raw_ident, raw_arr_obj in preds_dict.items():
                            try:
                                ident = int(raw_ident)
                            except (TypeError, ValueError):
                                continue
                            raw_arr = np.asarray(raw_arr_obj, dtype=np.int32)
                            prob_mat = _lookup_identity_arr(probs_dict, ident)
                            if prob_mat is None or prob_mat.ndim != 2:
                                continue
                            post_arr_obj = _lookup_identity_arr(post_dict, ident)
                            post_arr = (
                                np.asarray(post_arr_obj, dtype=np.int32)
                                if (use_postprocessed and post_arr_obj is not None)
                                else None
                            )
                            active_arr = post_arr if post_arr is not None else raw_arr
                            n_f = min(raw_arr.shape[0], prob_mat.shape[0], active_arr.shape[0])
                            if n_f <= 0:
                                continue

                            rec.has_predictions = True
                            has_any_preds = True
                            pred_identities.add(ident)
                            rec.total_frames += n_f
                            if n_f > rec.max_track_frames:
                                rec.max_track_frames = n_f

                            raw_sub = raw_arr[:n_f]
                            act_sub = active_arr[:n_f]
                            pm_sub = np.asarray(prob_mat[:n_f], dtype=np.float32)

                            # Preserve GapInterpolationStage-filled postprocessed frames (`act_valid`)
                            # while keeping raw pose validity (`raw_valid`) for pose_valid_frac & probs!
                            raw_valid = raw_sub >= 0
                            act_valid = act_sub >= 0
                            pose_valid_total += int(np.count_nonzero(raw_valid))
                            rec.valid_frames += int(np.count_nonzero(act_valid))

                            if eff_all_mc:
                                raw_pos = raw_valid & (raw_sub != none_idx)
                                act_pos = act_valid & (act_sub != none_idx)
                            else:
                                c_idx = target_indices[0]
                                raw_pos = raw_valid & (raw_sub == c_idx)
                                act_pos = act_valid & (act_sub == c_idx)
                            raw_none = raw_valid & (raw_sub == none_idx)
                            act_none = act_valid & (act_sub == none_idx)

                            if post_arr is not None:
                                union_m = (raw_pos | act_pos) & raw_valid
                                raw_or_post_pos_union += int(np.count_nonzero(union_m))
                                raw_vs_post_diff += int(
                                    np.count_nonzero((raw_pos ^ act_pos) & raw_valid)
                                )

                            # Full-matrix top-2 margin & None confidence (NaN-safe!)
                            n_cls = pm_sub.shape[1]
                            row_finite = np.all(np.isfinite(pm_sub), axis=1)
                            prob_valid = raw_valid & row_finite
                            pm_safe = np.where(row_finite[:, None], pm_sub, 0.0)
                            if n_cls >= 2:
                                sorted_p = np.sort(pm_safe, axis=1)
                                margin_arr = sorted_p[:, -1] - sorted_p[:, -2]
                            else:
                                margin_arr = np.ones(n_f, dtype=np.float32)
                            p_none_arr = (
                                pm_safe[:, none_idx]
                                if none_idx < n_cls
                                else np.zeros(n_f, dtype=np.float32)
                            )

                            # Human label overlap / disagreement
                            tr_ann = vid_anno.get(ident)
                            ann_pos_mask = (
                                _spans_to_mask(tr_ann["pos_spans"], n_f)
                                if tr_ann
                                else np.zeros(n_f, dtype=bool)
                            )
                            ann_neg_mask = (
                                _spans_to_mask(tr_ann["neg_spans"], n_f)
                                if tr_ann
                                else np.zeros(n_f, dtype=bool)
                            )
                            ann_any = ann_pos_mask | ann_neg_mask
                            if np.any(ann_any & act_valid):
                                ov = ann_any & act_valid
                                labeled_overlap_frames += int(np.count_nonzero(ov))
                                disagree_frames += int(
                                    np.count_nonzero(
                                        (ann_pos_mask & ~act_pos & act_valid)
                                        | (ann_neg_mask & act_pos & act_valid)
                                    )
                                )

                            rec.pred_pos_frames += int(np.count_nonzero(act_pos))
                            rec.unlabeled_pred_frames += int(np.count_nonzero(act_pos & ~ann_any))

                            # Mean None prob across valid finite None frames
                            none_prob_mask = raw_none & act_none & prob_valid
                            if np.any(none_prob_mask):
                                none_prob_sum += float(np.sum(p_none_arr[none_prob_mask]))
                                none_prob_count += int(np.count_nonzero(none_prob_mask))

                            # Vectorized prefix sums for O(1) per-bout probability & margin means
                            pv_i = prob_valid.astype(np.int64)
                            cs_pv = np.empty(n_f + 1, dtype=np.int64)
                            cs_pv[0] = 0
                            np.cumsum(pv_i, out=cs_pv[1:])

                            cs_m = np.empty(n_f + 1, dtype=np.float64)
                            cs_m[0] = 0.0
                            np.cumsum(np.where(prob_valid, margin_arr, 0.0), out=cs_m[1:])

                            cs_ann = np.empty(n_f + 1, dtype=np.int64)
                            cs_ann[0] = 0
                            np.cumsum(ann_any.astype(np.int64), out=cs_ann[1:])

                            # Per-target-class bouts (preserves distinct Left/Right bouts & Set1 colors!)
                            ts = rec.track_spans.setdefault(
                                ident,
                                {
                                    "pred_pos": [],
                                    "low_conf_pos": [],
                                    "low_conf_none": [],
                                    "label_pos": [],
                                    "label_neg": [],
                                },
                            )
                            track_bouts_for_switch: list[tuple[int, int, int]] = []
                            for c_idx in target_indices:
                                if c_idx >= n_cls:
                                    continue
                                c_name = class_names[c_idx]
                                c_mask = act_valid & (act_sub == c_idx)
                                runs_c = _contiguous_true_runs(c_mask)
                                if not runs_c:
                                    continue
                                c_probs = pm_safe[:, c_idx]
                                if np.any(c_mask & prob_valid):
                                    cm_pv = c_mask & prob_valid
                                    fallback_pos_prob_sum += float(np.sum(c_probs[cm_pv]))
                                    fallback_pos_prob_count += int(np.count_nonzero(cm_pv))
                                cs_pc = np.empty(n_f + 1, dtype=np.float64)
                                cs_pc[0] = 0.0
                                np.cumsum(np.where(prob_valid, c_probs, 0.0), out=cs_pc[1:])
                                cls_spans = rec.pred_by_class.setdefault(ident, {}).setdefault(
                                    c_name, []
                                )
                                for s, e in runs_c:
                                    dur = e - s + 1
                                    if dur < min_bout_frames:
                                        continue
                                    rec.pred_bout_count += 1
                                    class_bout_counts[c_idx] = class_bout_counts.get(c_idx, 0) + 1
                                    bout_lengths.append(dur)
                                    ts["pred_pos"].append((s, e))
                                    cls_spans.append((s, e))
                                    track_bouts_for_switch.append((s, e, c_idx))
                                    if (cs_ann[e + 1] - cs_ann[s]) < dur:
                                        rec.unlabeled_pred_bouts += 1
                                        if dur > rec.unlabeled_pred_max_bout_frames:
                                            rec.unlabeled_pred_max_bout_frames = dur

                                    if best_longest is None or dur > best_longest[0]:
                                        best_longest = (dur, ident, s, e)

                                    n_ok = int(cs_pv[e + 1] - cs_pv[s])
                                    if n_ok > 0:
                                        p_sum = float(cs_pc[e + 1] - cs_pc[s])
                                        m_sum = float(cs_m[e + 1] - cs_m[s])
                                        pos_prob_sum += p_sum
                                        pos_prob_count += n_ok
                                        mean_p = p_sum / float(n_ok)
                                        mean_m = m_sum / float(n_ok)
                                        if (
                                            not math.isfinite(rec.pred_min_bout_prob)
                                            or mean_p < rec.pred_min_bout_prob
                                        ):
                                            rec.pred_min_bout_prob = mean_p
                                        if mean_p <= beh_low_max or mean_m < margin_low_max:
                                            rec.pred_low_conf_pos_bouts += 1
                                            rec.pred_low_conf_pos_frames += dur
                                            ts["low_conf_pos"].append((s, e))
                                            if best_low_pos is None or mean_p < best_low_pos[0]:
                                                best_low_pos = (mean_p, ident, s, e)

                            # Rapid multi-class switching (e.g. Turn_Left <-> Turn_Right within <=15f)
                            if len(track_bouts_for_switch) >= 2:
                                track_bouts_for_switch.sort(key=lambda x: x[0])
                                for prev_b, next_b in pairwise(track_bouts_for_switch):
                                    if prev_b[2] != next_b[2] and (next_b[0] - prev_b[1]) <= 15:
                                        rec.pred_class_switch_count += 1

                            # E3b: Genuine raw-None near-miss spans (excluding postproc-suppressed raw
                            # positives and ±pad shoulders around any raw or postprocessed positive bout)
                            near_pos_shoulder = _dilate_bool_mask(
                                raw_pos | act_pos, shoulder_pad_frames
                            )
                            base_none_e3b = raw_none & act_none & prob_valid & ~near_pos_shoulder
                            if eff_all_mc:
                                low_none_mask = base_none_e3b & (
                                    (p_none_arr <= none_low_max) | (margin_arr < margin_low_max)
                                )
                            else:
                                c_idx = target_indices[0]
                                runner_p = (
                                    pm_safe[:, c_idx]
                                    if c_idx < n_cls
                                    else np.zeros(n_f, dtype=np.float32)
                                )
                                low_none_mask = (
                                    base_none_e3b
                                    & (
                                        (p_none_arr <= none_low_max)
                                        | ((p_none_arr - runner_p) < margin_low_max)
                                    )
                                    & (runner_p >= 0.25)
                                )
                            runs_nm = _contiguous_true_runs(low_none_mask)
                            if runs_nm:
                                cs_pn = np.empty(n_f + 1, dtype=np.float64)
                                cs_pn[0] = 0.0
                                np.cumsum(p_none_arr, out=cs_pn[1:])
                                for s, e in runs_nm:
                                    dur = e - s + 1
                                    if dur < min_near_miss_frames:
                                        continue
                                    mean_pn = float(cs_pn[e + 1] - cs_pn[s]) / float(dur)
                                    rec.pred_low_conf_none_spans += 1
                                    rec.pred_low_conf_none_frames += dur
                                    ts["low_conf_none"].append((s, e))
                                    if (
                                        not math.isfinite(rec.pred_min_none_prob)
                                        or mean_pn < rec.pred_min_none_prob
                                    ):
                                        rec.pred_min_none_prob = mean_pn
                                    if best_low_none is None or mean_pn < best_low_none[0]:
                                        best_low_none = (mean_pn, ident, s, e)

                        # Chiral asymmetry if >=2 positive classes had bouts (e.g. Turn_Left / Turn_Right)
                        if len(target_set) >= 2:
                            counts_sorted = sorted(
                                (class_bout_counts.get(ci, 0) for ci in target_indices),
                                reverse=True,
                            )
                            c_top = counts_sorted[0]
                            c_sec = counts_sorted[1]
                            if (c_top + c_sec) >= 4:
                                rec.pred_chiral_asymmetry = float(abs(c_top - c_sec)) / float(
                                    c_top + c_sec
                                )
            elif active_behavior:
                preds_dict_b: dict[Any, Any] = {}
                probs_dict_b: dict[Any, Any] = {}
                post_dict_b: dict[Any, Any] = {}
                try:
                    preds_dict_b, probs_dict_b, post_dict_b = (
                        project.prediction_manager.load_predictions(video_name, active_behavior)
                    )
                except Exception:
                    preds_dict_b = {}

                if preds_dict_b:
                    beh_val = TrackLabels.Label.BEHAVIOR.value
                    not_beh_val = TrackLabels.Label.NOT_BEHAVIOR.value

                    for raw_ident, raw_arr_obj in preds_dict_b.items():
                        try:
                            ident = int(raw_ident)
                        except (TypeError, ValueError):
                            continue
                        raw_arr = np.asarray(raw_arr_obj, dtype=np.int32)
                        prob_arr_obj = _lookup_identity_arr(probs_dict_b, ident)
                        if prob_arr_obj is None:
                            continue
                        prob_arr = np.asarray(prob_arr_obj, dtype=np.float32)
                        post_arr_obj = _lookup_identity_arr(post_dict_b, ident)
                        post_arr = (
                            np.asarray(post_arr_obj, dtype=np.int32)
                            if (use_postprocessed and post_arr_obj is not None)
                            else None
                        )
                        active_arr = post_arr if post_arr is not None else raw_arr
                        n_f = min(raw_arr.shape[0], prob_arr.shape[0], active_arr.shape[0])
                        if n_f <= 0:
                            continue

                        rec.has_predictions = True
                        has_any_preds = True
                        pred_identities.add(ident)
                        rec.total_frames += n_f
                        if n_f > rec.max_track_frames:
                            rec.max_track_frames = n_f

                        raw_sub = raw_arr[:n_f]
                        act_sub = active_arr[:n_f]
                        p_pos_arr, conf_pred_arr, margin_arr, prob_finite = (
                            _binary_prob_confidence(raw_sub, prob_arr[:n_f])
                        )

                        # Preserve GapInterpolationStage-filled postprocessed frames (`act_valid`)
                        # while restricting pose_valid_frac & probability sums to `raw_valid & prob_finite`.
                        raw_valid = raw_sub >= 0
                        act_valid = act_sub >= 0
                        prob_valid = raw_valid & prob_finite
                        pose_valid_total += int(np.count_nonzero(raw_valid))
                        rec.valid_frames += int(np.count_nonzero(act_valid))

                        raw_pos = raw_valid & (raw_sub == beh_val)
                        act_pos = act_valid & (act_sub == beh_val)
                        raw_neg = raw_valid & (raw_sub == not_beh_val)
                        act_neg = act_valid & (act_sub == not_beh_val)

                        if post_arr is not None:
                            union_m = (raw_pos | act_pos) & raw_valid
                            raw_or_post_pos_union += int(np.count_nonzero(union_m))
                            raw_vs_post_diff += int(
                                np.count_nonzero((raw_pos ^ act_pos) & raw_valid)
                            )

                        tr_ann = vid_anno.get(ident)
                        ann_pos_mask = (
                            _spans_to_mask(tr_ann["pos_spans"], n_f)
                            if tr_ann
                            else np.zeros(n_f, dtype=bool)
                        )
                        ann_neg_mask = (
                            _spans_to_mask(tr_ann["neg_spans"], n_f)
                            if tr_ann
                            else np.zeros(n_f, dtype=bool)
                        )
                        ann_any = ann_pos_mask | ann_neg_mask
                        if np.any(ann_any & act_valid):
                            ov = ann_any & act_valid
                            labeled_overlap_frames += int(np.count_nonzero(ov))
                            disagree_frames += int(
                                np.count_nonzero(
                                    (ann_pos_mask & ~act_pos & act_valid)
                                    | (ann_neg_mask & act_pos & act_valid)
                                )
                            )

                        rec.pred_pos_frames += int(np.count_nonzero(act_pos))
                        rec.unlabeled_pred_frames += int(np.count_nonzero(act_pos & ~ann_any))

                        p_pos_safe = np.where(prob_valid, p_pos_arr, 0.0)
                        margin_safe = np.where(prob_valid, margin_arr, 0.0)

                        if np.any(act_pos & prob_valid):
                            ap_pv = act_pos & prob_valid
                            fallback_pos_prob_sum += float(np.sum(p_pos_safe[ap_pv]))
                            fallback_pos_prob_count += int(np.count_nonzero(ap_pv))

                        none_prob_mask = raw_neg & act_neg & prob_valid
                        if np.any(none_prob_mask):
                            p_none_on_neg = 1.0 - p_pos_safe[none_prob_mask]
                            none_prob_sum += float(np.sum(p_none_on_neg))
                            none_prob_count += int(np.count_nonzero(none_prob_mask))

                        ts = rec.track_spans.setdefault(
                            ident,
                            {
                                "pred_pos": [],
                                "low_conf_pos": [],
                                "low_conf_none": [],
                                "label_pos": [],
                                "label_neg": [],
                            },
                        )

                        runs_b = _contiguous_true_runs(act_pos)
                        if runs_b:
                            cs_pv = np.empty(n_f + 1, dtype=np.int64)
                            cs_pv[0] = 0
                            np.cumsum(prob_valid.astype(np.int64), out=cs_pv[1:])

                            cs_p = np.empty(n_f + 1, dtype=np.float64)
                            cs_p[0] = 0.0
                            np.cumsum(p_pos_safe, out=cs_p[1:])

                            cs_m = np.empty(n_f + 1, dtype=np.float64)
                            cs_m[0] = 0.0
                            np.cumsum(margin_safe, out=cs_m[1:])

                            cs_ann = np.empty(n_f + 1, dtype=np.int64)
                            cs_ann[0] = 0
                            np.cumsum(ann_any.astype(np.int64), out=cs_ann[1:])

                            for s, e in runs_b:
                                dur = e - s + 1
                                if dur < min_bout_frames:
                                    continue
                                rec.pred_bout_count += 1
                                bout_lengths.append(dur)
                                ts["pred_pos"].append((s, e))
                                if (cs_ann[e + 1] - cs_ann[s]) < dur:
                                    rec.unlabeled_pred_bouts += 1
                                    if dur > rec.unlabeled_pred_max_bout_frames:
                                        rec.unlabeled_pred_max_bout_frames = dur

                                if best_longest is None or dur > best_longest[0]:
                                    best_longest = (dur, ident, s, e)

                                n_ok = int(cs_pv[e + 1] - cs_pv[s])
                                if n_ok > 0:
                                    p_sum = float(cs_p[e + 1] - cs_p[s])
                                    m_sum = float(cs_m[e + 1] - cs_m[s])
                                    pos_prob_sum += p_sum
                                    pos_prob_count += n_ok
                                    mean_p = p_sum / float(n_ok)
                                    mean_m = m_sum / float(n_ok)
                                    if (
                                        not math.isfinite(rec.pred_min_bout_prob)
                                        or mean_p < rec.pred_min_bout_prob
                                    ):
                                        rec.pred_min_bout_prob = mean_p
                                    if mean_p <= beh_low_max or mean_m < margin_low_max:
                                        rec.pred_low_conf_pos_bouts += 1
                                        rec.pred_low_conf_pos_frames += dur
                                        ts["low_conf_pos"].append((s, e))
                                        if best_low_pos is None or mean_p < best_low_pos[0]:
                                            best_low_pos = (mean_p, ident, s, e)

                        # E3b: Genuine raw Not-Behavior near-miss spans (excluding postproc-suppressed
                        # raw positives, NaN probabilities, and ±pad shoulders around positive bouts)
                        near_pos_shoulder = _dilate_bool_mask(
                            raw_pos | act_pos, shoulder_pad_frames
                        )
                        p_none_arr = np.where(
                            prob_valid,
                            np.where(raw_sub == not_beh_val, conf_pred_arr, 1.0 - p_pos_arr),
                            1.0,
                        )
                        low_none_mask = (
                            raw_neg
                            & act_neg
                            & prob_valid
                            & ~near_pos_shoulder
                            & ((p_none_arr <= none_low_max) | (margin_safe < margin_low_max))
                        )
                        runs_nm = _contiguous_true_runs(low_none_mask)
                        if runs_nm:
                            cs_pn = np.empty(n_f + 1, dtype=np.float64)
                            cs_pn[0] = 0.0
                            np.cumsum(p_none_arr, out=cs_pn[1:])
                            for s, e in runs_nm:
                                dur = e - s + 1
                                if dur < min_near_miss_frames:
                                    continue
                                mean_pn = float(cs_pn[e + 1] - cs_pn[s]) / float(dur)
                                rec.pred_low_conf_none_spans += 1
                                rec.pred_low_conf_none_frames += dur
                                ts["low_conf_none"].append((s, e))
                                if (
                                    not math.isfinite(rec.pred_min_none_prob)
                                    or mean_pn < rec.pred_min_none_prob
                                ):
                                    rec.pred_min_none_prob = mean_pn
                                if best_low_none is None or mean_pn < best_low_none[0]:
                                    best_low_none = (mean_pn, ident, s, e)

        # Finalize derived ratios & focal events for this video
        all_idents = pred_identities | label_identities
        rec.n_animals = len(all_idents)
        if not rec.has_predictions:
            rec.total_frames = label_track_total_frames
            rec.valid_frames = label_track_total_frames
            rec.pose_valid_frac = 1.0
        elif rec.total_frames > 0:
            rec.pose_valid_frac = float(pose_valid_total) / float(rec.total_frames)

        denom_f = rec.valid_frames if rec.valid_frames > 0 else rec.total_frames
        if denom_f > 0:
            rec.pred_pos_pct = 100.0 * float(rec.pred_pos_frames) / float(denom_f)
            rec.pred_bouts_per_min = float(rec.pred_bout_count) * 1800.0 / float(denom_f)
            rec.label_pos_pct = 100.0 * float(rec.label_pos_frames) / float(denom_f)

        if bout_lengths:
            rec.pred_max_bout_frames = max(bout_lengths)
            rec.pred_mean_bout_frames = float(np.mean(bout_lengths))
        if pos_prob_count > 0:
            rec.pred_mean_prob = pos_prob_sum / float(pos_prob_count)
        elif fallback_pos_prob_count > 0:
            rec.pred_mean_prob = fallback_pos_prob_sum / float(fallback_pos_prob_count)
        if none_prob_count > 0:
            rec.pred_mean_none_prob = none_prob_sum / float(none_prob_count)
        if raw_or_post_pos_union >= 10:
            rec.pred_postproc_fragility = float(raw_vs_post_diff) / float(raw_or_post_pos_union)
        if labeled_overlap_frames >= 10:
            rec.pred_label_disagree_frames = disagree_frames
            rec.pred_label_disagree_rate = float(disagree_frames) / float(labeled_overlap_frames)

        if best_longest is not None:
            dur, b_id, b_s, b_e = best_longest
            rec.focal_events["longest"] = VideoFocalEvent(
                b_id, b_s, b_e, f"Longest predicted bout ({dur}f)"
            )
        if best_low_pos is not None:
            mp, b_id, b_s, b_e = best_low_pos
            rec.focal_events["low_conf_pos"] = VideoFocalEvent(
                b_id, b_s, b_e, f"Lowest-conf Behavior bout (P={mp:.2f}, {b_e - b_s + 1}f)"
            )
        if best_low_none is not None:
            mpn, b_id, b_s, b_e = best_low_none
            rec.focal_events["low_conf_none"] = VideoFocalEvent(
                b_id,
                b_s,
                b_e,
                f"Near-miss Not-Behavior span (P(None)={mpn:.2f}, {b_e - b_s + 1}f)",
            )

        # Default jump anchor: prioritize low-conf Behavior -> Near-miss None -> Longest bout -> Label -> Frame 0
        for pref_k in ("low_conf_pos", "low_conf_none", "longest", "label"):
            if pref_k in rec.focal_events:
                rec.focal_events["default"] = rec.focal_events[pref_k]
                break
        if "default" not in rec.focal_events:
            first_id = min(all_idents) if all_idents else 0
            rec.focal_events["default"] = VideoFocalEvent(
                first_id, 0, 0, "Video start (0 behavior bouts)"
            )

        records.append(rec)

    eff_source: VideoMetricSource = source
    if eff_source != "labels" and not has_any_preds and has_any_labels:
        eff_source = "labels"
    thresholds = classify_video_outliers(records, source=eff_source)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return VideoScreenerReport(
        project_dir=str(project_dir.resolve()),
        behavior=active_behavior,
        is_multiclass=is_multiclass,
        include_all_multiclass=eff_all_mc,
        use_postprocessed=use_postprocessed,
        records=records,
        has_any_predictions=has_any_preds,
        has_any_labels=has_any_labels,
        elapsed_ms=elapsed_ms,
        thresholds=thresholds,
    )
