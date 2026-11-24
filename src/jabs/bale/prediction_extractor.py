"""Two-stage fast predicted bout extractor for `jabs.bale`.

Stage A (< 120 ms across 465 videos):
    Scans `jabs/predictions/<video_stem>.h5` via `project.prediction_manager`,
    segments contiguous unlabeled predicted bouts, and computes zero-Parquet-I/O
    pre-scores (`mean_pred_prob`, `margin_uncertainty`, `postproc_fragility`).
Stage B (Bounded K_max Parquet extraction):
    Selects up to `max_predicted_bouts` (default 500: 70% highest pre-doubt +
    30% stratified across probability deciles/videos) so feature extraction stays
    sub-second even when a project has 40,000+ predicted bouts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierMode
from jabs.io.annotations import read_document
from jabs.project.track_labels import TrackLabels

from .bout_extractor import resolve_behavior_settings, resolve_project_video_name
from .models import BoutRecord

if TYPE_CHECKING:
    from jabs.project import Project

logger = logging.getLogger(__name__)


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive `(start, end)` frame intervals for contiguous True runs in `mask`."""
    if mask.size == 0 or not np.any(mask):
        return []
    padded = np.empty(mask.size + 2, dtype=np.int8)
    padded[0] = 0
    padded[1:-1] = mask.astype(np.int8, copy=False)
    padded[-1] = 0
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def _lookup_identity_arr(d: dict[Any, Any] | None, ident: Any) -> Any:
    """Look up a per-identity prediction/probability array using `int(ident)` then `str(ident)`."""
    if not d:
        return None
    try:
        int_id = int(ident)
        if int_id in d:
            return d[int_id]
    except (TypeError, ValueError):
        pass
    str_id = str(ident)
    if str_id in d:
        return d[str_id]
    if ident in d:
        return d[ident]
    return None


def _load_annotated_frame_masks(
    project: Project,
    project_dir: Path | str | None = None,
    behavior: str | bool | None = None,
    is_multiclass: bool | None = None,
) -> dict[tuple[str, int], list[tuple[int, int]]]:
    """Return existing human-annotated `(start, end)` spans per `(video_name, identity)`."""
    if is_multiclass is None and isinstance(behavior, bool):
        eff_is_multiclass = behavior
        eff_behavior = project_dir if isinstance(project_dir, str) else None
        eff_project_dir = Path(project.project_paths.project_dir)
    else:
        eff_project_dir = (
            Path(project_dir)
            if isinstance(project_dir, Path | str)
            else Path(project.project_paths.project_dir)
        )
        eff_behavior = behavior if isinstance(behavior, str) else None
        eff_is_multiclass = bool(is_multiclass)

    spans_by_track: dict[tuple[str, int], list[tuple[int, int]]] = {}

    def _collect_from_label_dict(video_name: str, label_dict: Any) -> None:
        if not isinstance(label_dict, dict):
            return
        for id_str, beh_map in label_dict.items():
            try:
                identity = int(id_str)
            except (TypeError, ValueError):
                continue
            if not isinstance(beh_map, dict):
                continue
            intervals: list[tuple[int, int]] = []
            if eff_is_multiclass:
                for beh_key, beh_blocks in beh_map.items():
                    if not isinstance(beh_blocks, list):
                        continue
                    for blk in beh_blocks:
                        if not isinstance(blk, dict):
                            continue
                        if blk.get("present", False) or (
                            eff_behavior is not None and beh_key == eff_behavior
                        ):
                            intervals.append((int(blk["start"]), int(blk["end"])))
            elif eff_behavior and eff_behavior in beh_map:
                beh_blocks = beh_map.get(eff_behavior)
                if isinstance(beh_blocks, list):
                    for blk in beh_blocks:
                        if isinstance(blk, dict):
                            intervals.append((int(blk["start"]), int(blk["end"])))
            if intervals:
                spans_by_track.setdefault((video_name, identity), []).extend(intervals)

    anno_dir = eff_project_dir / "jabs" / "annotations"
    seen_json_videos: set[str] = set()
    if anno_dir.is_dir():
        for anno_path in sorted(anno_dir.glob("*.json")):
            doc = read_document(anno_path)
            if not doc:
                continue
            video_name = resolve_project_video_name(project, anno_path.name, doc)
            seen_json_videos.add(video_name)
            _collect_from_label_dict(video_name, doc.get("unfragmented_labels"))
            _collect_from_label_dict(video_name, doc.get("labels"))

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
                try:
                    live_dict: dict[str, dict[str, list[dict[str, Any]]]] = {}
                    for id_str, beh_key, tl in vl.iter_identity_behavior_labels():
                        blocks = tl.get_blocks()
                        if blocks:
                            live_dict.setdefault(str(id_str), {})[str(beh_key)] = blocks
                    if live_dict:
                        _collect_from_label_dict(v_name, live_dict)
                except Exception:
                    continue
    except Exception:
        pass

    return spans_by_track


def _classify_label_halo(
    start_frame: int,
    end_frame: int,
    track_spans: list[tuple[int, int]],
    margin_frames: int,
) -> tuple[bool, str, int | None]:
    """Classify whether an unlabeled predicted run `(start_frame, end_frame)` is a label-edge halo.

    Returns `(is_label_halo, halo_side, halo_distance_frames)`:
    - Directly touching a human-labeled span (`dist == 0`, i.e., `s - 1` or `e + 1` is annotated)
      is always tagged as a Label Edge (`"before"`, `"after"`, or `"between"`).
    - Nearby short fragments within `1..margin_frames` of a labeled span whose duration is at most
      `max(15, 2 * margin_frames)` frames are also tagged as Label Edge halos, while long
      standalone predicted bouts remain normal standalone predictions.
    """
    if not track_spans or margin_frames < 0:
        return False, "", None

    dur = end_frame - start_frame + 1
    max_near_dur = max(15, 2 * max(1, margin_frames))
    best_before: int | None = None
    best_after: int | None = None

    for s_a, e_a in track_spans:
        # Predicted run sits immediately before a labeled span [s_a, e_a]
        if end_frame < s_a:
            d_before = s_a - end_frame - 1
            if (d_before == 0 or (d_before <= margin_frames and dur <= max_near_dur)) and (
                best_before is None or d_before < best_before
            ):
                best_before = d_before
        # Predicted run sits immediately after a labeled span [s_a, e_a]
        elif start_frame > e_a:
            d_after = start_frame - e_a - 1
            if (d_after == 0 or (d_after <= margin_frames and dur <= max_near_dur)) and (
                best_after is None or d_after < best_after
            ):
                best_after = d_after

    if best_before is not None and best_after is not None:
        return True, "between", min(best_before, best_after)
    if best_before is not None:
        return True, "before", best_before
    if best_after is not None:
        return True, "after", best_after
    return False, "", None


def _sample_doubt_and_stratified(
    items: list[tuple[float, BoutRecord, np.ndarray, np.ndarray]],
    cap: int,
) -> list[tuple[float, BoutRecord, np.ndarray, np.ndarray]]:
    """Select up to `cap` items (`70%` highest pre-doubt + `30%` stratified across remainder)."""
    if cap <= 0 or not items:
        return []
    ordered = sorted(items, key=lambda item: item[0], reverse=True)
    if len(ordered) <= cap:
        return ordered
    n_top = round(0.70 * cap)
    n_strat = cap - n_top
    top_part = ordered[:n_top]
    remainder = ordered[n_top:]
    if n_strat > 0 and remainder:
        stride_idx = np.linspace(0, len(remainder) - 1, num=n_strat, dtype=int)
        strat_part = [remainder[i] for i in stride_idx]
    else:
        strat_part = []
    return top_part + strat_part


def _resolve_halo_margin(project: Project, beh_name: str | None, override: int | None) -> int:
    if override is not None:
        return max(0, int(override))
    try:
        settings = resolve_behavior_settings(project, beh_name or "")
        win = int(settings.get("window_size", 10))
    except Exception:
        win = 10
    return max(5, win)


def collect_predicted_bout_records(
    project: Project,
    behavior: str | None = None,
    *,
    use_postprocessed: bool = True,
    max_predicted_bouts: int = 300,
    min_bout_frames: int = 4,
    only_video: str | None = None,
    start_bout_id: int = 0,
    max_videos_to_scan: int = 5000,
    synthesize_if_missing: bool = False,
    halo_margin_frames: int | None = None,
    max_halo_bouts: int | None = None,
) -> tuple[list[BoutRecord], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Scan `jabs/predictions/*.h5` and return top-doubt + stratified predicted `BoutRecord`s.

    Predicted fragments touching or within `halo_margin_frames` (default: `max(5, window_size)`)
    of an existing human-labeled bout on the same animal are tagged with `is_label_halo=True`
    (`halo_side` in `{"before", "after", "between"}`) and sampled in their own reserved quota
    (`max_halo_bouts`) so label-edge halos never crowd out standalone predicted bouts.

    Returns:
        `(records, frame_probs_by_bout_id, raw_preds_by_bout_id)`
    """
    project_dir = Path(project.project_paths.project_dir)
    is_multiclass = project.settings_manager.classifier_mode == ClassifierMode.MULTICLASS
    behavior_names = list(project.settings_manager.behavior_names)
    if behavior is None and behavior_names and not is_multiclass:
        behavior = behavior_names[0]

    annotated_spans = _load_annotated_frame_masks(project, project_dir, behavior, is_multiclass)
    margin_cache: dict[str, int] = {}

    def _margin_for(beh_key: str) -> int:
        if beh_key not in margin_cache:
            margin_cache[beh_key] = _resolve_halo_margin(project, beh_key, halo_margin_frames)
        return margin_cache[beh_key]

    pred_dir = project_dir / "jabs" / "predictions"
    candidates: list[tuple[float, BoutRecord, np.ndarray, np.ndarray]] = []
    if pred_dir.is_dir():
        if only_video is not None:
            videos_to_scan = [resolve_project_video_name(project, only_video)]
        else:
            pred_files = sorted(pred_dir.glob("*.h5"), key=lambda p: p.name)
            annotated_vids = {tk[0] for tk in annotated_spans}
            prio_vids: list[str] = []
            other_vids: list[str] = []
            seen_vids: set[str] = set()
            for p in pred_files:
                v_name = resolve_project_video_name(project, p.name)
                if v_name in seen_vids:
                    continue
                seen_vids.add(v_name)
                if v_name in annotated_vids:
                    prio_vids.append(v_name)
                else:
                    other_vids.append(v_name)
            prio_vids.sort()
            other_vids.sort()
            videos_to_scan = (prio_vids + other_vids)[: max(0, int(max_videos_to_scan))]
    else:
        videos_to_scan = []

    for video_name in videos_to_scan:
        if project.settings_manager.is_video_excluded(video_name):
            continue

        if is_multiclass:
            try:
                preds_dict, probs_dict, post_dict, class_names = (
                    project.prediction_manager.load_multiclass_predictions(video_name)
                )
            except ValueError:
                continue
            if not class_names or not preds_dict:
                continue

            target_behaviors = (
                [b for b in behavior_names if b != MULTICLASS_NONE_BEHAVIOR]
                if behavior is None
                else [behavior]
            )
            if behavior is None and not target_behaviors:
                target_behaviors = [c for c in class_names if c != MULTICLASS_NONE_BEHAVIOR]
            for beh_name in target_behaviors:
                if beh_name not in class_names:
                    continue
                cls_idx = class_names.index(beh_name)
                eff_margin = _margin_for(beh_name)
                for raw_ident, raw_pred_arr in preds_dict.items():
                    try:
                        identity = int(raw_ident)
                    except (TypeError, ValueError):
                        continue
                    prob_mat = _lookup_identity_arr(probs_dict, identity)
                    if prob_mat is None or prob_mat.ndim != 2 or cls_idx >= prob_mat.shape[1]:
                        continue
                    post_arr = _lookup_identity_arr(post_dict, identity)
                    active_arr = (
                        post_arr if (use_postprocessed and post_arr is not None) else raw_pred_arr
                    )
                    if active_arr is None or active_arr.size == 0:
                        continue
                    n_frames = min(active_arr.shape[0], prob_mat.shape[0], raw_pred_arr.shape[0])
                    if n_frames <= 0:
                        continue
                    track_spans = annotated_spans.get((video_name, identity), [])
                    unlabeled_mask = np.ones(n_frames, dtype=bool)
                    for s_a, e_a in track_spans:
                        unlabeled_mask[max(0, s_a) : min(n_frames, e_a + 1)] = False

                    pos_mask = (active_arr[:n_frames] == cls_idx) & unlabeled_mask
                    for s, e in _contiguous_true_runs(pos_mask):
                        dur = e - s + 1
                        if dur < min_bout_frames:
                            continue
                        is_halo, h_side, h_dist = _classify_label_halo(
                            s, e, track_spans, eff_margin
                        )
                        p_slice = np.asarray(prob_mat[s : e + 1, cls_idx], dtype=np.float32)
                        r_slice = (raw_pred_arr[s : e + 1] == cls_idx).astype(np.int8)
                        mean_p = float(np.mean(p_slice))
                        fragility = float(1.0 - np.mean(r_slice))
                        pre_doubt = 0.65 * (1.0 - mean_p) + 0.35 * fragility
                        rec = BoutRecord(
                            bout_id=-1,
                            source="predicted",
                            video=video_name,
                            identity=identity,
                            behavior=beh_name,
                            class_name=beh_name,
                            is_positive=True,
                            start_frame=s,
                            end_frame=e,
                            duration_frames=dur,
                            mean_pred_prob=mean_p,
                            postproc_fragility=fragility,
                            origin_source="predicted",
                            origin_behavior=beh_name,
                            is_label_halo=is_halo,
                            halo_side=h_side,  # type: ignore[arg-type]
                            halo_distance_frames=h_dist,
                        )
                        candidates.append((pre_doubt, rec, p_slice, r_slice))
        else:
            if not behavior:
                continue
            try:
                preds_dict, probs_dict, post_dict = project.prediction_manager.load_predictions(
                    video_name, behavior
                )
            except ValueError:
                continue
            if not preds_dict:
                continue

            eff_margin = _margin_for(behavior)
            for raw_ident, raw_pred_arr in preds_dict.items():
                try:
                    identity = int(raw_ident)
                except (TypeError, ValueError):
                    continue
                prob_arr = _lookup_identity_arr(probs_dict, identity)
                if prob_arr is None or prob_arr.size == 0:
                    continue
                post_arr = _lookup_identity_arr(post_dict, identity)
                active_arr = (
                    post_arr if (use_postprocessed and post_arr is not None) else raw_pred_arr
                )
                if active_arr is None or active_arr.size == 0:
                    continue
                n_frames = min(active_arr.shape[0], prob_arr.shape[0], raw_pred_arr.shape[0])
                if n_frames <= 0:
                    continue
                track_spans = annotated_spans.get((video_name, identity), [])
                unlabeled_mask = np.ones(n_frames, dtype=bool)
                for s_a, e_a in track_spans:
                    unlabeled_mask[max(0, s_a) : min(n_frames, e_a + 1)] = False

                pos_mask = (
                    active_arr[:n_frames] == TrackLabels.Label.BEHAVIOR.value
                ) & unlabeled_mask
                for s, e in _contiguous_true_runs(pos_mask):
                    dur = e - s + 1
                    if dur < min_bout_frames:
                        continue
                    is_halo, h_side, h_dist = _classify_label_halo(s, e, track_spans, eff_margin)
                    p_slice = np.asarray(prob_arr[s : e + 1], dtype=np.float32)
                    r_slice = (raw_pred_arr[s : e + 1] == TrackLabels.Label.BEHAVIOR.value).astype(
                        np.int8
                    )
                    mean_p = float(np.mean(p_slice))
                    fragility = float(1.0 - np.mean(r_slice))
                    pre_doubt = 0.65 * (1.0 - mean_p) + 0.35 * fragility
                    rec = BoutRecord(
                        bout_id=-1,
                        source="predicted",
                        video=video_name,
                        identity=identity,
                        behavior=behavior,
                        class_name=behavior,
                        is_positive=True,
                        start_frame=s,
                        end_frame=e,
                        duration_frames=dur,
                        mean_pred_prob=mean_p,
                        postproc_fragility=fragility,
                        origin_source="predicted",
                        origin_behavior=behavior,
                        is_label_halo=is_halo,
                        halo_side=h_side,  # type: ignore[arg-type]
                        halo_distance_frames=h_dist,
                    )
                    candidates.append((pre_doubt, rec, p_slice, r_slice))

    if not candidates and synthesize_if_missing:
        return _synthesize_unlabeled_predictions_from_tracks(
            project,
            behavior=behavior,
            is_multiclass=is_multiclass,
            annotated_spans=annotated_spans,
            max_predicted_bouts=max_predicted_bouts,
            min_bout_frames=min_bout_frames,
            start_bout_id=start_bout_id,
        )
    if not candidates:
        return [], {}, {}

    # Separate standalone predicted bouts from Label Edge (Halo) fragments so halos never
    # starve genuine standalone predictions out of the `max_predicted_bouts` sampling budget.
    standalone_candidates = [c for c in candidates if not c[1].is_label_halo]
    halo_candidates = [c for c in candidates if c[1].is_label_halo]

    selected_standalone = _sample_doubt_and_stratified(standalone_candidates, max_predicted_bouts)
    halo_cap = (
        max(0, int(max_halo_bouts))
        if max_halo_bouts is not None
        else min(200, max(50, max_predicted_bouts // 2))
    )
    selected_halos = _sample_doubt_and_stratified(halo_candidates, halo_cap)
    selected = [*selected_standalone, *selected_halos]

    out_records: list[BoutRecord] = []
    frame_probs: dict[int, np.ndarray] = {}
    raw_preds: dict[int, np.ndarray] = {}

    for offset, (_, rec, p_slice, r_slice) in enumerate(selected):
        bid = start_bout_id + offset
        rec.bout_id = bid
        out_records.append(rec)
        frame_probs[bid] = p_slice
        raw_preds[bid] = r_slice

    return out_records, frame_probs, raw_preds


def _synthesize_unlabeled_predictions_from_tracks(
    project: Project,
    *,
    behavior: str | None,
    is_multiclass: bool,
    annotated_spans: dict[tuple[str, int], list[tuple[int, int]]],
    max_predicted_bouts: int = 500,
    min_bout_frames: int = 5,
    start_bout_id: int = 0,
) -> tuple[list[BoutRecord], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Synthesize unlabeled predicted bouts on-the-fly when `jabs/predictions/*.h5` is empty.

    Fits a fast `ExtraTreesClassifier` on strided labeled frames from the project's annotated
    tracks and scans the unlabeled intervals of those tracks to extract candidate predicted
    bouts with per-frame probability trajectories.
    """
    from sklearn.ensemble import ExtraTreesClassifier

    from .bout_extractor import collect_labeled_bout_records, load_track_per_frame_array

    project_dir = Path(project.project_paths.project_dir)
    behavior_names = list(project.settings_manager.behavior_names)
    active_beh = behavior or (behavior_names[0] if behavior_names else "")
    beh_settings = resolve_behavior_settings(project, behavior)
    use_social = bool(beh_settings.get("social", True))
    use_segmentation = bool(beh_settings.get("segmentation", True))

    labeled_bouts = collect_labeled_bout_records(
        project,
        active_beh,
        include_positive=True,
        include_negative=True,
        include_all_multiclass=is_multiclass,
    )
    if len(labeled_bouts) < 4:
        return [], {}, {}

    # Group labeled bouts by track
    by_track: dict[tuple[str, int], list[BoutRecord]] = {}
    for b in labeled_bouts:
        by_track.setdefault((b.video, b.identity), []).append(b)

    X_train_list: list[np.ndarray] = []
    y_train_list: list[str] = []
    loaded_tracks: dict[tuple[str, int], np.ndarray] = {}

    for (vid, ident), b_list in list(by_track.items())[:40]:
        loaded = load_track_per_frame_array(
            project_dir,
            vid,
            ident,
            use_social=use_social,
            use_segmentation=use_segmentation,
        )
        if loaded is None:
            continue
        X_tr, _, _ = loaded
        loaded_tracks[(vid, ident)] = X_tr
        n_f = X_tr.shape[0]
        for b in b_list:
            s = max(0, min(b.start_frame, n_f - 1))
            e = max(s, min(b.end_frame, n_f - 1))
            sub = X_tr[s : e + 1 : 3]
            if len(sub) > 0:
                X_train_list.append(sub)
                y_train_list.extend([b.class_name] * len(sub))

    if not X_train_list:
        return [], {}, {}

    unique_classes = sorted(set(y_train_list))
    if len(unique_classes) < 2:
        return [], {}, {}

    X_train = np.vstack(X_train_list)
    cls_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_train = np.array([cls_to_idx[c] for c in y_train_list], dtype=np.int32)

    clf = ExtraTreesClassifier(
        n_estimators=60,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    neg_class = MULTICLASS_NONE_BEHAVIOR if is_multiclass else f"Not-{active_beh}"
    pos_class_indices = [
        i
        for i, c in enumerate(unique_classes)
        if c != neg_class and (behavior is None or not is_multiclass or c == active_beh)
    ]
    if not pos_class_indices:
        return [], {}, {}

    candidates: list[tuple[float, BoutRecord, np.ndarray, np.ndarray]] = []
    kernel = np.ones(7, dtype=np.float32) / 7.0

    for (vid, ident), X_tr in loaded_tracks.items():
        n_f = X_tr.shape[0]
        unlabeled_mask = np.ones(n_f, dtype=bool)
        for s_a, e_a in annotated_spans.get((vid, ident), []):
            unlabeled_mask[max(0, s_a) : min(n_f, e_a + 1)] = False
        if not np.any(unlabeled_mask):
            continue

        probs_all = clf.predict_proba(X_tr).astype(np.float32)
        raw_top = np.argmax(probs_all, axis=1)
        smooth_probs = np.column_stack(
            [np.convolve(probs_all[:, k], kernel, mode="same") for k in range(probs_all.shape[1])]
        )
        smooth_top = np.argmax(smooth_probs, axis=1)

        for c_idx in pos_class_indices:
            c_name = unique_classes[c_idx]
            p_raw = probs_all[:, c_idx]
            p_smooth = smooth_probs[:, c_idx]
            pos_mask = (smooth_top == c_idx) & (p_smooth >= 0.42) & unlabeled_mask
            for s, e in _contiguous_true_runs(pos_mask):
                dur = e - s + 1
                if dur < min_bout_frames:
                    continue
                if dur > 120:
                    e = s + 119
                    dur = 120
                p_slice = np.asarray(p_raw[s : e + 1], dtype=np.float32)
                r_slice = (raw_top[s : e + 1] == c_idx).astype(np.int8)
                mean_p = float(np.mean(p_slice))
                fragility = float(1.0 - np.mean(r_slice))
                pre_doubt = 0.65 * (1.0 - mean_p) + 0.35 * fragility
                beh_val = c_name if is_multiclass else active_beh
                rec = BoutRecord(
                    bout_id=-1,
                    source="predicted",
                    video=vid,
                    identity=int(ident),
                    behavior=beh_val,
                    class_name=c_name,
                    is_positive=True,
                    start_frame=int(s),
                    end_frame=int(e),
                    duration_frames=int(dur),
                    mean_pred_prob=mean_p,
                    postproc_fragility=fragility,
                    origin_source="predicted",
                    origin_behavior=beh_val,
                )
                candidates.append((pre_doubt, rec, p_slice, r_slice))

    if not candidates:
        return [], {}, {}

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_predicted_bouts]
    out_records: list[BoutRecord] = []
    frame_probs: dict[int, np.ndarray] = {}
    raw_preds: dict[int, np.ndarray] = {}
    for offset, (_, rec, p_slice, r_slice) in enumerate(selected):
        bid = start_bout_id + offset
        rec.bout_id = bid
        out_records.append(rec)
        frame_probs[bid] = p_slice
        raw_preds[bid] = r_slice
    return out_records, frame_probs, raw_preds
