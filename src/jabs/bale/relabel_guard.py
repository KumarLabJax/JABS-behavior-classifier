"""Cross-validation group guard (`Dual-Class CV rule` & `ADR-0003`), 1-click relabeling, and undo stack."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from jabs.bale.bout_extractor import (
    resolve_behavior_settings,
    strip_known_video_suffix,
)
from jabs.bale.models import BoutRecord, CVGroupImpact
from jabs.bale.trajectory_and_export import load_annotation_pose_proxy
from jabs.classifier import Classifier, MultiClassClassifier
from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import (
    DEFAULT_CV_GROUPING_STRATEGY,
    ClassifierMode,
    CrossValidationGroupingStrategy,
)
from jabs.feature_extraction import DEFAULT_WINDOW_SIZE
from jabs.project import Project, TrackLabels, VideoLabels

logger = logging.getLogger(__name__)

EditAction = Literal["flip", "trim", "split", "delete", "promote_prediction"]

MIN_DUAL_CLASS_FRAMES_PER_TRACK = Classifier.LABEL_THRESHOLD
MIN_DUAL_CLASS_CV_TRACKS = 15


@dataclass
class BoutEditSnapshot:
    """Scoped undo/redo snapshot for a single bout edit (and optional paired bout).

    Restores ONLY `touched_spans` on undo/redo so manual edits made on other frames of the
    same `(video, identity)` track in the main JABS timeline are never clobbered.
    """

    description: str
    video: str
    identity: int
    blocks_by_behavior_before: dict[str, list[dict[str, Any]]]
    blocks_by_behavior_after: dict[str, list[dict[str, Any]]]
    touched_spans: list[tuple[int, int]] = field(default_factory=list)
    span_slices_before: dict[str, list[np.ndarray]] = field(default_factory=dict)
    span_slices_after: dict[str, list[np.ndarray]] = field(default_factory=dict)
    paired_bout: BoutRecord | None = None


class BaleUndoStack:
    """In-memory undo/redo stack for label mutations performed inside BALE."""

    def __init__(self, max_depth: int = 100) -> None:
        self._undo_stack: list[BoutEditSnapshot] = []
        self._redo_stack: list[BoutEditSnapshot] = []
        self._max_depth = int(max_depth)

    def push(self, snapshot: BoutEditSnapshot) -> None:
        """Record a completed edit and clear the redo stack."""
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._max_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    @property
    def can_undo(self) -> bool:
        """Return True if an edit is available on the undo stack."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """Return True if an undone edit is available on the redo stack."""
        return len(self._redo_stack) > 0

    def undo(
        self,
        project: Project,
        on_video_labels_saved: Callable[[str, VideoLabels], None] | None = None,
    ) -> BoutEditSnapshot | None:
        """Revert the most recent edit on its exact touched spans and persist to disk."""
        if not self._undo_stack:
            return None
        snap = self._undo_stack.pop()
        _restore_snapshot_state(
            project,
            snap.video,
            snap.identity,
            touched_spans=snap.touched_spans,
            span_slices=snap.span_slices_before,
            fallback_blocks=snap.blocks_by_behavior_before,
            on_video_labels_saved=on_video_labels_saved,
        )
        self._redo_stack.append(snap)
        return snap

    def redo(
        self,
        project: Project,
        on_video_labels_saved: Callable[[str, VideoLabels], None] | None = None,
    ) -> BoutEditSnapshot | None:
        """Re-apply the most recently undone edit on its exact touched spans and persist."""
        if not self._redo_stack:
            return None
        snap = self._redo_stack.pop()
        _restore_snapshot_state(
            project,
            snap.video,
            snap.identity,
            touched_spans=snap.touched_spans,
            span_slices=snap.span_slices_after,
            fallback_blocks=snap.blocks_by_behavior_after,
            on_video_labels_saved=on_video_labels_saved,
        )
        self._undo_stack.append(snap)
        return snap


def _apply_label_array_slice(
    tl: TrackLabels, span_s: int, span_e: int, values: np.ndarray
) -> None:
    """Restore exact per-frame `TrackLabels` values over `[span_s, span_e]` without touching outside frames."""
    tl.clear_labels(span_s, span_e)
    n = len(values)
    if n == 0:
        return
    pos_val = int(TrackLabels.Label.BEHAVIOR.value)
    neg_val = int(TrackLabels.Label.NOT_BEHAVIOR.value)
    i = 0
    while i < n:
        v = int(values[i])
        j = i
        while j + 1 < n and int(values[j + 1]) == v:
            j += 1
        if v == pos_val:
            tl.label_behavior(span_s + i, span_s + j)
        elif v == neg_val:
            tl.label_not_behavior(span_s + i, span_s + j)
        i = j + 1


def _record_session_label_diff(
    project: Project,
    video: str,
    identity: int,
    tracked_behaviors: list[str],
    touched_spans: list[tuple[int, int]],
    span_slices_before: dict[str, list[np.ndarray]],
    span_slices_after: dict[str, list[np.ndarray]],
) -> None:
    """Log contiguous created/deleted label spans to `project.session_tracker`."""
    tracker = getattr(project, "session_tracker", None)
    if tracker is None:
        return
    label_created_fn = getattr(tracker, "label_created", None)
    label_deleted_fn = getattr(tracker, "label_deleted", None)
    if not callable(label_created_fn) or not callable(label_deleted_fn):
        return

    video_path = Path(video)
    ident_int = int(identity)
    none_val = int(TrackLabels.Label.NONE.value)
    pos_val = int(TrackLabels.Label.BEHAVIOR.value)
    neg_val = int(TrackLabels.Label.NOT_BEHAVIOR.value)

    for beh_name in tracked_behaviors:
        before_list = span_slices_before.get(beh_name, [])
        after_list = span_slices_after.get(beh_name, [])
        for (sp_s, _sp_e), before_arr, after_arr in zip(
            touched_spans, before_list, after_list, strict=False
        ):
            n = min(len(before_arr), len(after_arr))
            i = 0
            while i < n:
                b_val = int(before_arr[i])
                a_val = int(after_arr[i])
                if a_val == b_val:
                    i += 1
                    continue
                j = i
                while (
                    j + 1 < n
                    and int(after_arr[j + 1]) == a_val
                    and int(before_arr[j + 1]) != a_val
                ):
                    j += 1
                if a_val == none_val:
                    label_deleted_fn(video_path, ident_int, beh_name, int(sp_s + i), int(sp_s + j))
                elif a_val == pos_val:
                    label_created_fn(
                        video_path, ident_int, beh_name, True, int(sp_s + i), int(sp_s + j)
                    )
                elif a_val == neg_val:
                    label_created_fn(
                        video_path, ident_int, beh_name, False, int(sp_s + i), int(sp_s + j)
                    )
                i = j + 1


def _restore_snapshot_state(
    project: Project,
    video: str,
    identity: int,
    *,
    touched_spans: list[tuple[int, int]],
    span_slices: dict[str, list[np.ndarray]],
    fallback_blocks: dict[str, list[dict[str, Any]]],
    on_video_labels_saved: Callable[[str, VideoLabels], None] | None = None,
) -> None:
    """Restore scoped `[span_s, span_e]` slices (or full blocks fallback) for `(video, identity)`."""
    vm = project.video_manager
    pose = load_annotation_pose_proxy(project, video)
    video_labels = vm.load_video_labels(video, pose)
    if video_labels is None:
        video_labels = VideoLabels(video, pose.num_frames)

    ident_str = str(identity)
    eff_beh_names = (
        list(span_slices.keys())
        if (touched_spans and span_slices)
        else list(fallback_blocks.keys())
    )
    eff_spans = (
        touched_spans
        if (touched_spans and span_slices)
        else [(0, max(0, video_labels.num_frames - 1))]
    )
    current_slices_before: dict[str, list[np.ndarray]] = {
        beh_name: [
            np.asarray(
                video_labels.get_track_labels(ident_str, beh_name).get_labels()[sp_s : sp_e + 1]
            ).copy()
            for sp_s, sp_e in eff_spans
        ]
        for beh_name in eff_beh_names
    }

    if touched_spans and span_slices:
        for beh_name, slices in span_slices.items():
            tl = video_labels.get_track_labels(ident_str, beh_name)
            for (sp_s, sp_e), arr_slice in zip(touched_spans, slices, strict=False):
                _apply_label_array_slice(tl, int(sp_s), int(sp_e), arr_slice)
    else:
        for beh_name, blocks in fallback_blocks.items():
            tl = video_labels.get_track_labels(ident_str, beh_name)
            tl.clear_labels(0, max(0, video_labels.num_frames - 1))
            for b in blocks:
                s = int(b["start"])
                e = int(b["end"])
                if b.get("present", True):
                    tl.label_behavior(s, e)
                else:
                    tl.label_not_behavior(s, e)

    project.save_annotations(video_labels, pose)
    restored_slices_after: dict[str, list[np.ndarray]] = {
        beh_name: [
            np.asarray(
                video_labels.get_track_labels(ident_str, beh_name).get_labels()[sp_s : sp_e + 1]
            ).copy()
            for sp_s, sp_e in eff_spans
        ]
        for beh_name in eff_beh_names
    }
    _record_session_label_diff(
        project,
        video,
        identity,
        eff_beh_names,
        eff_spans,
        current_slices_before,
        restored_slices_after,
    )
    if on_video_labels_saved is not None:
        on_video_labels_saved(video, video_labels)


def _project_cv_grouping(
    project: Project,
) -> tuple[CrossValidationGroupingStrategy, str | None]:
    """Return `(cv_grouping_strategy, cv_grouping_regex)` from `project.settings_manager`."""
    sm = getattr(project, "settings_manager", None)
    raw_strategy = getattr(sm, "cv_grouping_strategy", DEFAULT_CV_GROUPING_STRATEGY)
    if isinstance(raw_strategy, CrossValidationGroupingStrategy):
        strategy = raw_strategy
    elif isinstance(raw_strategy, str):
        try:
            strategy = CrossValidationGroupingStrategy(raw_strategy)
        except ValueError:
            strategy = DEFAULT_CV_GROUPING_STRATEGY
    else:
        strategy = DEFAULT_CV_GROUPING_STRATEGY
    raw_regex = getattr(sm, "cv_grouping_regex", None)
    regex = raw_regex if isinstance(raw_regex, str) else None
    return strategy, regex


def _track_counts_to_jabs_counts(
    track_counts: dict[tuple[str, int], tuple[int, int]],
) -> dict[str, dict[int, dict[str, tuple[int, int]]]]:
    """Adapt BALE's `(video, identity) -> (pos_f, neg_f)` dict to `Classifier`'s count shape."""
    all_counts: dict[str, dict[int, dict[str, tuple[int, int]]]] = {}
    for (video, ident), (pos_f, neg_f) in track_counts.items():
        all_counts.setdefault(video, {})[int(ident)] = {
            "fragmented_frame_counts": (int(pos_f), int(neg_f)),
        }
    return all_counts


def summarize_cv_track_counts(
    track_counts: dict[tuple[str, int], tuple[int, int]],
    *,
    cv_grouping_strategy: CrossValidationGroupingStrategy = DEFAULT_CV_GROUPING_STRATEGY,
    cv_grouping_regex: str | None = None,
    min_groups: int = 1,
) -> tuple[int, bool, bool]:
    """Compute `(dual_groups, jabs_threshold_met, dual_class_cv_met)` matching JABS's CV rule."""
    all_counts = _track_counts_to_jabs_counts(track_counts)
    dual_groups = Classifier.count_label_threshold(
        all_counts,
        cv_grouping_strategy=cv_grouping_strategy,
        cv_grouping_regex=cv_grouping_regex,
    )
    jabs_met = Classifier.label_threshold_met(
        all_counts,
        min_groups=min_groups,
        cv_grouping_strategy=cv_grouping_strategy,
        cv_grouping_regex=cv_grouping_regex,
    )
    dual_class_cv_met = dual_groups >= MIN_DUAL_CLASS_CV_TRACKS
    return dual_groups, jabs_met, dual_class_cv_met


def collect_multiclass_counts_by_behavior(
    project: Project,
) -> dict[str, dict[str, dict[int, dict[str, tuple[int, int]]]]]:
    """Collect `{behavior: {video: {identity: counts}}}` across all non-excluded videos once."""
    sm = project.settings_manager
    all_behaviors = [MULTICLASS_NONE_BEHAVIOR, *list(sm.behavior_names)]
    return {
        b: {
            v: {
                int(ident): dict(cdict)
                for ident, cdict in id_map.items()
                if isinstance(cdict, dict)
            }
            for v, id_map in project.counts(b).items()
            if not sm.is_video_excluded(v) and isinstance(id_map, dict)
        }
        for b in all_behaviors
    }


def refresh_multiclass_video_counts(
    project: Project,
    counts_by_behavior: dict[str, dict[str, dict[int, dict[str, tuple[int, int]]]]],
    touched_videos: set[str] | list[str],
) -> bool:
    """Reload per-behavior fragmented label counts for ONLY `touched_videos`."""
    sm = project.settings_manager
    load_counts_fn = getattr(project, "load_counts", None)
    if not callable(load_counts_fn):
        return False

    all_behaviors = [MULTICLASS_NONE_BEHAVIOR, *list(sm.behavior_names)]
    updated_any = False
    for vid in touched_videos:
        if sm.is_video_excluded(vid):
            for b in all_behaviors:
                counts_by_behavior.setdefault(b, {}).pop(vid, None)
            updated_any = True
            continue
        for b in all_behaviors:
            beh_map = counts_by_behavior.setdefault(b, {})
            try:
                vid_counts = load_counts_fn(vid, b)
            except Exception:
                logger.warning(
                    "Failed to reload multi-class counts for video=%s behavior=%s",
                    vid,
                    b,
                    exc_info=True,
                )
                return False
            if not isinstance(vid_counts, dict):
                return False
            updated_any = True
            cleaned = {
                int(ident): dict(cdict)
                for ident, cdict in vid_counts.items()
                if isinstance(cdict, dict)
            }
            if cleaned:
                beh_map[vid] = cleaned
            else:
                beh_map.pop(vid, None)
    return updated_any


def summarize_multiclass_counts_by_behavior(
    project: Project,
    behavior: str,
    counts_by_behavior: dict[str, dict[str, dict[int, dict[str, tuple[int, int]]]]],
) -> tuple[int, bool, bool, dict[tuple[str, int], tuple[int, int]]]:
    """Summarize cached multi-class per-behavior counts in memory with zero disk I/O."""
    sm = project.settings_manager
    strategy, regex = _project_cv_grouping(project)
    all_behaviors = [MULTICLASS_NONE_BEHAVIOR, *list(sm.behavior_names)]
    filtered_counts = {
        b: {
            v: id_map
            for v, id_map in counts_by_behavior.get(b, {}).items()
            if not sm.is_video_excluded(v)
        }
        for b in all_behaviors
    }

    track_counts: dict[tuple[str, int], tuple[int, int]] = {}
    for video in project.video_manager.videos:
        if sm.is_video_excluded(video):
            continue
        identities: set[int] = set()
        for b in all_behaviors:
            identities.update(filtered_counts[b].get(video, {}).keys())
        for ident in sorted(identities):
            pos_f = (
                filtered_counts.get(behavior, {})
                .get(video, {})
                .get(ident, {})
                .get("fragmented_frame_counts", (0, 0))[0]
            )
            other_f = sum(
                filtered_counts[b]
                .get(video, {})
                .get(ident, {})
                .get("fragmented_frame_counts", (0, 0))[0]
                for b in all_behaviors
                if b != behavior
            )
            if pos_f > 0 or other_f > 0:
                track_counts[(video, int(ident))] = (int(pos_f), int(other_f))

    dual_groups = MultiClassClassifier.count_label_threshold(
        counts_by_behavior=filtered_counts,
        behavior_names=all_behaviors,
        cv_grouping_strategy=strategy,
        cv_grouping_regex=regex,
    )
    jabs_met = MultiClassClassifier.label_threshold_met(
        counts_by_behavior=filtered_counts,
        behavior_names=all_behaviors,
        min_groups=1,
        cv_grouping_strategy=strategy,
        cv_grouping_regex=regex,
    )
    dual_class_cv_met = dual_groups >= MIN_DUAL_CLASS_CV_TRACKS
    return dual_groups, jabs_met, dual_class_cv_met, track_counts


def count_dual_class_cv_groups(
    project: Project,
    behavior: str,
    *,
    precomputed_mc_counts_by_behavior: (
        dict[str, dict[str, dict[int, dict[str, tuple[int, int]]]]] | None
    ) = None,
) -> tuple[int, bool, bool, dict[tuple[str, int], tuple[int, int]]]:
    """Count CV groups for `behavior` using JABS's own `count_label_threshold` / `label_threshold_met`."""
    sm = project.settings_manager
    is_multiclass = sm.classifier_mode == ClassifierMode.MULTICLASS
    strategy, regex = _project_cv_grouping(project)
    track_counts: dict[tuple[str, int], tuple[int, int]] = {}

    if not is_multiclass:
        raw_counts = project.counts(behavior)
        for video, id_map in raw_counts.items():
            if sm.is_video_excluded(video):
                continue
            for ident, cdict in id_map.items():
                pos_f, neg_f = cdict.get("fragmented_frame_counts", (0, 0))
                if pos_f > 0 or neg_f > 0:
                    track_counts[(video, int(ident))] = (int(pos_f), int(neg_f))
        dual_groups, jabs_met, dual_class_cv_met = summarize_cv_track_counts(
            track_counts,
            cv_grouping_strategy=strategy,
            cv_grouping_regex=regex,
        )
        return dual_groups, jabs_met, dual_class_cv_met, track_counts

    mc_counts = (
        precomputed_mc_counts_by_behavior
        if precomputed_mc_counts_by_behavior is not None
        else collect_multiclass_counts_by_behavior(project)
    )
    return summarize_multiclass_counts_by_behavior(project, behavior, mc_counts)


def _get_window_size(project: Project, behavior: str | None) -> int:
    """Resolve classifier window size for binary or multi-class projects."""
    beh_cfg = resolve_behavior_settings(project, behavior)
    raw_ws = beh_cfg.get("window_size")
    if isinstance(raw_ws, int | float) and not isinstance(raw_ws, bool) and int(raw_ws) > 0:
        return int(raw_ws)
    sm = getattr(project, "settings_manager", None)
    ps = getattr(sm, "project_settings", None) if sm is not None else None
    defaults = ps.get("defaults") if isinstance(ps, dict) else getattr(sm, "defaults", None)
    if isinstance(defaults, dict):
        raw_def = defaults.get("window_size")
        if isinstance(raw_def, int | float) and not isinstance(raw_def, bool) and int(raw_def) > 0:
            return int(raw_def)
    return int(DEFAULT_WINDOW_SIZE)


def find_dual_class_pairing_candidate(
    project: Project,
    video: str,
    identity: int,
    behavior: str,
    needed_positive: bool,
    exclude_span: tuple[int, int] | None = None,
    min_duration: int = Classifier.LABEL_THRESHOLD,
    max_duration: int = 60,
) -> BoutRecord | None:
    """Find a high-confidence opposite-class span on the same `(video, identity)` track."""
    sm = project.settings_manager
    is_multiclass = sm.classifier_mode == ClassifierMode.MULTICLASS
    window_size = _get_window_size(project, behavior)
    min_gap = window_size + 5

    pred_file = project.project_paths.prediction_dir / f"{strip_known_video_suffix(video)}.h5"
    if not pred_file.exists():
        return None

    pm = project.prediction_manager
    probs: np.ndarray | None = None
    invalid_pred_mask: np.ndarray | None = None
    try:
        if is_multiclass:
            _, mc_probs, _, class_names = pm.load_multiclass_predictions(video)
            if class_names and behavior in class_names:
                c_idx = class_names.index(behavior)
                arr = mc_probs.get(identity)
                if arr is None:
                    arr = mc_probs.get(str(identity))  # type: ignore[arg-type]
                if arr is not None and arr.ndim == 2 and c_idx < arr.shape[1]:
                    probs = np.asarray(arr[:, c_idx], dtype=np.float32)
        else:
            bin_preds, bin_probs, _ = pm.load_predictions(video, behavior)
            arr = bin_probs.get(identity)
            if arr is None:
                arr = bin_probs.get(str(identity))  # type: ignore[arg-type]
            pred_raw = bin_preds.get(identity) if isinstance(bin_preds, dict) else None
            if pred_raw is None and isinstance(bin_preds, dict):
                pred_raw = bin_preds.get(str(identity))  # type: ignore[arg-type]
            if arr is not None:
                raw_p = np.asarray(arr, dtype=np.float32)
                if pred_raw is not None:
                    pred_arr = np.asarray(pred_raw)
                    if pred_arr.shape == raw_p.shape:
                        invalid_pred_mask = pred_arr == TrackLabels.Label.NONE.value
                        neg_conf_mask = (pred_arr == TrackLabels.Label.NOT_BEHAVIOR.value) & (
                            raw_p >= 0.5
                        )
                        probs = np.where(neg_conf_mask, 1.0 - raw_p, raw_p).astype(np.float32)
                    else:
                        probs = raw_p
                else:
                    probs = raw_p
    except Exception:
        logger.warning(
            "Failed to load predictions while searching dual-class pairing candidate for video=%s id=%s behavior=%s",
            video,
            identity,
            behavior,
            exc_info=True,
        )
        return None

    if probs is None or len(probs) == 0:
        return None

    num_frames = len(probs)
    forbidden = np.zeros(num_frames, dtype=bool)
    if invalid_pred_mask is not None and len(invalid_pred_mask) == num_frames:
        forbidden |= invalid_pred_mask
    if exclude_span is not None:
        s0 = max(0, int(exclude_span[0]) - min_gap)
        e0 = min(num_frames, int(exclude_span[1]) + min_gap + 1)
        forbidden[s0:e0] = True

    pose = load_annotation_pose_proxy(project, video)
    video_labels = project.video_manager.load_video_labels(video, pose)
    if video_labels is not None:
        for _, tl in video_labels.iter_behavior_labels(str(identity)):
            arr = tl.get_labels()
            labeled_idx = np.flatnonzero(arr != TrackLabels.Label.NONE)
            if len(labeled_idx) > 0:
                for b in tl.get_blocks():
                    s0 = max(0, int(b["start"]) - min_gap)
                    e0 = min(num_frames, int(b["end"]) + min_gap + 1)
                    forbidden[s0:e0] = True

    cand_mask = ((probs >= 0.80) if needed_positive else (probs <= 0.20)) & (~forbidden)

    padded = np.concatenate(([False], cand_mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1) - 1

    best_rec: BoutRecord | None = None
    best_conf = -1.0

    for s, e in zip(starts.tolist(), ends.tolist(), strict=False):
        dur = e - s + 1
        if dur < min_duration:
            continue
        if dur > max_duration:
            mid = (s + e) // 2
            s = max(s, mid - max_duration // 2)
            e = min(e, s + max_duration - 1)
            dur = e - s + 1
        mean_p = float(np.mean(probs[s : e + 1]))
        conf = mean_p if needed_positive else (1.0 - mean_p)
        if conf > best_conf:
            best_conf = conf
            class_name = (
                (behavior if needed_positive else MULTICLASS_NONE_BEHAVIOR)
                if is_multiclass
                else (behavior if needed_positive else f"Not-{behavior}")
            )
            best_rec = BoutRecord(
                bout_id=-1,
                source="predicted",
                video=video,
                identity=int(identity),
                behavior=behavior if not is_multiclass else class_name,
                class_name=class_name,
                is_positive=bool(needed_positive),
                start_frame=int(s),
                end_frame=int(e),
                duration_frames=int(dur),
                valid_pose_fraction=1.0,
                mean_pred_prob=mean_p,
                postproc_fragility=0.0,
            )

    return best_rec


def simulate_track_count_delta(
    pos_before: int,
    neg_before: int,
    *,
    action: EditAction,
    start_frame: int,
    end_frame: int,
    current_is_positive: bool = True,
    target_is_positive: bool | None = None,
    trim_bounds: tuple[int, int] | None = None,
    source_is_predicted: bool = False,
) -> tuple[int, int]:
    """Compute `(pos_after, neg_after)` for a single `(video, identity)` edit."""
    dur = max(0, int(end_frame) - int(start_frame) + 1)
    pos_after, neg_after = pos_before, neg_before
    if action == "delete":
        if current_is_positive:
            pos_after = max(0, pos_before - dur)
        else:
            neg_after = max(0, neg_before - dur)
    elif action == "flip":
        eff_new_pos = (
            (not current_is_positive) if target_is_positive is None else bool(target_is_positive)
        )
        if eff_new_pos != current_is_positive:
            if current_is_positive:
                pos_after = max(0, pos_before - dur)
                neg_after = neg_before + dur
            else:
                neg_after = max(0, neg_before - dur)
                pos_after = pos_before + dur
    elif action == "trim" and trim_bounds is not None:
        new_dur = max(0, int(trim_bounds[1]) - int(trim_bounds[0]) + 1)
        delta = max(0, dur - new_dur)
        if current_is_positive:
            pos_after = max(0, pos_before - delta)
        else:
            neg_after = max(0, neg_before - delta)
    elif action == "split":
        gap_frames = min(3, max(0, dur - 2))
        if source_is_predicted:
            to_pos = True if target_is_positive is None else bool(target_is_positive)
            kept = max(0, dur - gap_frames)
            if to_pos:
                pos_after = pos_before + kept
            else:
                neg_after = neg_before + kept
        else:
            if current_is_positive:
                pos_after = max(0, pos_before - gap_frames)
            else:
                neg_after = max(0, neg_before - gap_frames)
    elif action == "promote_prediction":
        to_pos = bool(target_is_positive) if target_is_positive is not None else True
        if to_pos:
            pos_after = pos_before + dur
        else:
            neg_after = neg_before + dur
    return pos_after, neg_after


def evaluate_cv_group_impact(
    project: Project,
    behavior: str,
    *,
    video: str,
    identity: int,
    action: EditAction,
    start_frame: int,
    end_frame: int,
    current_is_positive: bool = True,
    target_is_positive: bool | None = None,
    target_behavior: str | None = None,
    trim_bounds: tuple[int, int] | None = None,
    source_is_predicted: bool = False,
    precomputed_track_counts: dict[tuple[str, int], tuple[int, int]] | None = None,
    precomputed_mc_counts_by_behavior: (
        dict[str, dict[str, dict[int, dict[str, tuple[int, int]]]]] | None
    ) = None,
    find_pairing_candidate: bool = True,
) -> CVGroupImpact:
    """Evaluate how a proposed bout edit affects JABS CV group invariants (`Dual-Class CV rule`)."""
    sm = project.settings_manager
    strategy, regex = _project_cv_grouping(project)
    if sm.classifier_mode == ClassifierMode.MULTICLASS:
        all_behaviors = [MULTICLASS_NONE_BEHAVIOR, *list(sm.behavior_names)]
        counts_by_behavior = (
            {
                b: {
                    v: id_map
                    for v, id_map in precomputed_mc_counts_by_behavior.get(b, {}).items()
                    if not sm.is_video_excluded(v)
                }
                for b in all_behaviors
            }
            if precomputed_mc_counts_by_behavior is not None
            else collect_multiclass_counts_by_behavior(project)
        )
        before_groups = MultiClassClassifier.count_label_threshold(
            counts_by_behavior=counts_by_behavior,
            behavior_names=all_behaviors,
            cv_grouping_strategy=strategy,
            cv_grouping_regex=regex,
        )
        jabs_before = MultiClassClassifier.label_threshold_met(
            counts_by_behavior=counts_by_behavior,
            behavior_names=all_behaviors,
            min_groups=1,
            cv_grouping_strategy=strategy,
            cv_grouping_regex=regex,
        )
        adr_before = before_groups >= MIN_DUAL_CLASS_CV_TRACKS

        # Shallow-copy outer behavior dicts and only clone `video` entries touched by this edit.
        sim_counts_by_beh: dict[str, dict[str, dict[int, dict[str, tuple[int, int]]]]] = {
            b: dict(counts_by_behavior.get(b, {})) for b in all_behaviors
        }
        ident_int = int(identity)
        dur = max(0, int(end_frame) - int(start_frame) + 1)
        src_beh = behavior if current_is_positive else MULTICLASS_NONE_BEHAVIOR

        def _delta_mc(beh_name: str, delta: int) -> None:
            if beh_name not in sim_counts_by_beh or sm.is_video_excluded(video) or delta == 0:
                return
            v_map = dict(sim_counts_by_beh[beh_name].get(video, {}))
            sim_counts_by_beh[beh_name][video] = v_map
            entry = dict(v_map.get(ident_int, {"fragmented_frame_counts": (0, 0)}))
            old_p, old_n = entry.get("fragmented_frame_counts", (0, 0))
            entry["fragmented_frame_counts"] = (max(0, int(old_p) + delta), int(old_n))
            v_map[ident_int] = entry

        if action == "delete":
            _delta_mc(src_beh, -dur)
        elif action == "flip":
            eff_new_pos = (
                (not current_is_positive)
                if target_is_positive is None
                else bool(target_is_positive)
            )
            dest_beh = (target_behavior or behavior) if eff_new_pos else MULTICLASS_NONE_BEHAVIOR
            if dest_beh != src_beh:
                _delta_mc(src_beh, -dur)
                _delta_mc(dest_beh, dur)
        elif action == "trim" and trim_bounds is not None:
            new_dur = max(0, int(trim_bounds[1]) - int(trim_bounds[0]) + 1)
            _delta_mc(src_beh, -max(0, dur - new_dur))
        elif action == "split":
            gap_frames = min(3, max(0, dur - 2))
            if source_is_predicted:
                eff_new_pos = True if target_is_positive is None else bool(target_is_positive)
                dest_beh = (
                    (target_behavior or behavior) if eff_new_pos else MULTICLASS_NONE_BEHAVIOR
                )
                _delta_mc(dest_beh, max(0, dur - gap_frames))
            else:
                _delta_mc(src_beh, -gap_frames)
        elif action == "promote_prediction":
            eff_new_pos = bool(target_is_positive) if target_is_positive is not None else True
            dest_beh = (target_behavior or behavior) if eff_new_pos else MULTICLASS_NONE_BEHAVIOR
            _delta_mc(dest_beh, dur)

        after_groups = MultiClassClassifier.count_label_threshold(
            counts_by_behavior=sim_counts_by_beh,
            behavior_names=all_behaviors,
            cv_grouping_strategy=strategy,
            cv_grouping_regex=regex,
        )
        jabs_after = MultiClassClassifier.label_threshold_met(
            counts_by_behavior=sim_counts_by_beh,
            behavior_names=all_behaviors,
            min_groups=1,
            cv_grouping_strategy=strategy,
            cv_grouping_regex=regex,
        )
        adr_after = after_groups >= MIN_DUAL_CLASS_CV_TRACKS
        pos_after = (
            sim_counts_by_beh.get(behavior, {})
            .get(video, {})
            .get(ident_int, {})
            .get("fragmented_frame_counts", (0, 0))[0]
        )
        neg_after = sum(
            sim_counts_by_beh[b]
            .get(video, {})
            .get(ident_int, {})
            .get("fragmented_frame_counts", (0, 0))[0]
            for b in all_behaviors
            if b != behavior
        )
    else:
        if precomputed_track_counts is not None:
            track_counts = precomputed_track_counts
            before_groups, jabs_before, adr_before = summarize_cv_track_counts(
                track_counts,
                cv_grouping_strategy=strategy,
                cv_grouping_regex=regex,
            )
        else:
            before_groups, jabs_before, adr_before, track_counts = count_dual_class_cv_groups(
                project, behavior
            )
        key = (video, int(identity))
        pos_before, neg_before = track_counts.get(key, (0, 0))
        pos_after, neg_after = simulate_track_count_delta(
            pos_before,
            neg_before,
            action=action,
            start_frame=start_frame,
            end_frame=end_frame,
            current_is_positive=current_is_positive,
            target_is_positive=target_is_positive,
            trim_bounds=trim_bounds,
            source_is_predicted=source_is_predicted,
        )

        sim_counts = dict(track_counts)
        sim_counts[key] = (pos_after, neg_after)
        after_groups, jabs_after, adr_after = summarize_cv_track_counts(
            sim_counts,
            cv_grouping_strategy=strategy,
            cv_grouping_regex=regex,
        )

    warning_msg: str | None = None
    suggested_pair: BoutRecord | None = None
    safe = True

    if jabs_before and not jabs_after:
        safe = False
        warning_msg = (
            f"Edit drops valid CV groups ({before_groups} -> {after_groups}) or class frames "
            f"below JABS minimum ({Classifier.LABEL_THRESHOLD})."
        )
    elif adr_before and not adr_after:
        safe = False
        warning_msg = (
            f"Edit reduces dual-class animal tracks from {before_groups} to {after_groups} "
            f"(below Dual-Class CV rule minimum of {MIN_DUAL_CLASS_CV_TRACKS})."
        )
    elif action == "promote_prediction":
        if (
            pos_after >= MIN_DUAL_CLASS_FRAMES_PER_TRACK
            and neg_after < MIN_DUAL_CLASS_FRAMES_PER_TRACK
        ):
            if find_pairing_candidate:
                suggested_pair = find_dual_class_pairing_candidate(
                    project,
                    video,
                    identity,
                    behavior,
                    needed_positive=False,
                    exclude_span=(start_frame, end_frame),
                )
            warning_msg = (
                f"Track ({video}, id={identity}) has 0 Not-{behavior} frames; "
                "pair with a negative bout on this track so it forms a valid CV group (Dual-Class CV)."
            )
        elif (
            neg_after >= MIN_DUAL_CLASS_FRAMES_PER_TRACK
            and pos_after < MIN_DUAL_CLASS_FRAMES_PER_TRACK
        ):
            if find_pairing_candidate:
                suggested_pair = find_dual_class_pairing_candidate(
                    project,
                    video,
                    identity,
                    behavior,
                    needed_positive=True,
                    exclude_span=(start_frame, end_frame),
                )
            warning_msg = (
                f"Track ({video}, id={identity}) has 0 {behavior} frames; "
                "pair with a positive bout on this track so it forms a valid CV group (Dual-Class CV)."
            )

    return CVGroupImpact(
        before_groups=before_groups,
        after_groups=after_groups,
        jabs_threshold_met_before=jabs_before,
        jabs_threshold_met_after=jabs_after,
        dual_class_cv_met_before=adr_before,
        dual_class_cv_met_after=adr_after,
        safe_to_apply=safe,
        warning_message=warning_msg,
        suggested_pairing_bout=suggested_pair,
    )


def apply_bout_edit(
    project: Project,
    behavior: str,
    *,
    video: str,
    identity: int,
    action: EditAction,
    start_frame: int,
    end_frame: int,
    current_is_positive: bool = True,
    target_is_positive: bool | None = None,
    target_behavior: str | None = None,
    trim_bounds: tuple[int, int] | None = None,
    split_frame: int | None = None,
    source_is_predicted: bool = False,
    pair_with_opposite: bool = False,
    precomputed_track_counts: dict[tuple[str, int], tuple[int, int]] | None = None,
    precomputed_impact: CVGroupImpact | None = None,
    pose_est: Any | None = None,
    undo_stack: BaleUndoStack | None = None,
    on_video_labels_saved: Callable[[str, VideoLabels], None] | None = None,
) -> BoutEditSnapshot:
    """Apply a bout relabel/trim/split/delete/promote operation to JABS annotations."""
    sm = project.settings_manager
    is_multiclass = sm.classifier_mode == ClassifierMode.MULTICLASS
    vm = project.video_manager
    pose = pose_est if pose_est is not None else load_annotation_pose_proxy(project, video)
    video_labels = vm.load_video_labels(video, pose)
    if video_labels is None:
        video_labels = VideoLabels(video, pose.num_frames)

    ident_str = str(identity)
    tracked_behaviors = (
        [MULTICLASS_NONE_BEHAVIOR, *list(sm.behavior_names)]
        if is_multiclass
        else list(sm.behavior_names)
    )
    if behavior not in tracked_behaviors:
        tracked_behaviors.append(behavior)
    if target_behavior and target_behavior not in tracked_behaviors:
        tracked_behaviors.append(target_behavior)

    before_blocks: dict[str, list[dict[str, Any]]] = {
        b: [dict(blk) for blk in video_labels.get_track_labels(ident_str, b).get_blocks()]
        for b in tracked_behaviors
    }

    s = max(0, int(start_frame))
    e = min(video_labels.num_frames - 1, int(end_frame))
    eff_target_beh = target_behavior or behavior

    # Determine optional paired bout upfront so we capture its exact pre-edit slice too
    paired_bout: BoutRecord | None = None
    if action == "promote_prediction" and pair_with_opposite:
        if precomputed_impact is not None:
            paired_bout = precomputed_impact.suggested_pairing_bout
        else:
            new_pos_preview = True if target_is_positive is None else bool(target_is_positive)
            impact = evaluate_cv_group_impact(
                project,
                eff_target_beh,
                video=video,
                identity=identity,
                action="promote_prediction",
                start_frame=s,
                end_frame=e,
                target_is_positive=new_pos_preview,
                precomputed_track_counts=precomputed_track_counts,
            )
            if impact.suggested_pairing_bout is not None:
                paired_bout = impact.suggested_pairing_bout

    touched_spans: list[tuple[int, int]] = [(s, e)]
    if paired_bout is not None:
        ps = max(0, int(paired_bout.start_frame))
        pe = min(video_labels.num_frames - 1, int(paired_bout.end_frame))
        touched_spans.append((ps, pe))

    span_slices_before: dict[str, list[np.ndarray]] = {
        b: [
            np.asarray(
                video_labels.get_track_labels(ident_str, b).get_labels()[sp_s : sp_e + 1]
            ).copy()
            for sp_s, sp_e in touched_spans
        ]
        for b in tracked_behaviors
    }

    def _set_span(beh: str, span_s: int, span_e: int, is_pos: bool) -> None:
        if is_multiclass:
            dest_track = beh if is_pos else MULTICLASS_NONE_BEHAVIOR
            for other_b in tracked_behaviors:
                tl_other = video_labels.get_track_labels(ident_str, other_b)
                if other_b == dest_track:
                    tl_other.label_behavior(span_s, span_e)
                else:
                    tl_other.clear_labels(span_s, span_e)
        else:
            tl = video_labels.get_track_labels(ident_str, beh)
            if is_pos:
                tl.label_behavior(span_s, span_e)
            else:
                tl.label_not_behavior(span_s, span_e)

    def _clear_span(beh: str, span_s: int, span_e: int) -> None:
        if is_multiclass:
            for other_b in tracked_behaviors:
                video_labels.get_track_labels(ident_str, other_b).clear_labels(span_s, span_e)
        else:
            video_labels.get_track_labels(ident_str, beh).clear_labels(span_s, span_e)

    if action == "delete":
        _clear_span(behavior, s, e)
    elif action == "flip":
        new_pos = (
            (not current_is_positive) if target_is_positive is None else bool(target_is_positive)
        )
        _set_span(eff_target_beh, s, e, new_pos)
    elif action == "trim":
        if trim_bounds is None:
            raise ValueError("trim_bounds=(new_start, new_end) is required for action='trim'")
        ns = max(s, int(trim_bounds[0]))
        ne = min(e, int(trim_bounds[1]))
        _clear_span(behavior, s, e)
        if ns <= ne:
            _set_span(eff_target_beh, ns, ne, current_is_positive)
    elif action == "split":
        sf = int(split_frame) if split_frame is not None else (s + e) // 2
        gap_s = max(s + 1, sf - 1)
        gap_e = min(e - 1, sf + 1)
        if source_is_predicted:
            new_pos = True if target_is_positive is None else bool(target_is_positive)
            if s <= gap_s - 1:
                _set_span(eff_target_beh, s, gap_s - 1, new_pos)
            if gap_e + 1 <= e:
                _set_span(eff_target_beh, gap_e + 1, e, new_pos)
        if gap_s <= gap_e:
            _clear_span(eff_target_beh if source_is_predicted else behavior, gap_s, gap_e)
    elif action == "promote_prediction":
        new_pos = True if target_is_positive is None else bool(target_is_positive)
        _set_span(eff_target_beh, s, e, new_pos)
        if paired_bout is not None:
            ps = max(0, int(paired_bout.start_frame))
            pe = min(video_labels.num_frames - 1, int(paired_bout.end_frame))
            _set_span(eff_target_beh, ps, pe, paired_bout.is_positive)
    else:
        raise ValueError(f"Unsupported bout edit action: {action!r}")

    project.save_annotations(video_labels, pose)
    if on_video_labels_saved is not None:
        on_video_labels_saved(video, video_labels)

    span_slices_after: dict[str, list[np.ndarray]] = {
        b: [
            np.asarray(
                video_labels.get_track_labels(ident_str, b).get_labels()[sp_s : sp_e + 1]
            ).copy()
            for sp_s, sp_e in touched_spans
        ]
        for b in tracked_behaviors
    }
    _record_session_label_diff(
        project,
        video,
        identity,
        tracked_behaviors,
        touched_spans,
        span_slices_before,
        span_slices_after,
    )

    after_blocks: dict[str, list[dict[str, Any]]] = {
        b: [dict(blk) for blk in video_labels.get_track_labels(ident_str, b).get_blocks()]
        for b in tracked_behaviors
    }

    snapshot = BoutEditSnapshot(
        description=f"{action} on {video} id={identity} [{s}-{e}]",
        video=video,
        identity=identity,
        blocks_by_behavior_before=before_blocks,
        blocks_by_behavior_after=after_blocks,
        touched_spans=touched_spans,
        span_slices_before=span_slices_before,
        span_slices_after=span_slices_after,
        paired_bout=paired_bout,
    )
    if undo_stack is not None:
        undo_stack.push(snapshot)
    return snapshot
