"""Fast track-grouped bout feature extractor for `jabs.bale`.

Reads `per_frame.parquet` (or legacy HDF5 features) once per `(video, identity)` track,
filters base features according to the active behavior's feature settings (`social`,
`segmentation`, `static_objects`), and computes 7 duration-invariant bout statistics
(`median`, `iqr`, `q10`, `q90`, `mean`, `mad`, `half_delta`) plus `log2_duration`
and `valid_pose_fraction` using vectorized C-level `np.sort` order statistics.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pyarrow.parquet as pq

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import CacheFormat, ClassifierMode
from jabs.core.utils.utilities import pose_file_stem
from jabs.feature_extraction.landmark_features.landmark_group import LandmarkFeatureGroup
from jabs.feature_extraction.segmentation_features.segment_group import SegmentationFeatureGroup
from jabs.feature_extraction.social_features.huddle_features import HuddleFeatures
from jabs.feature_extraction.social_features.social_group import SocialFeatureGroup
from jabs.io.annotations import read_document
from jabs.io.feature_cache import detect_cache_format, inspect_identity_cache
from jabs.io.feature_cache.hdf5 import HDF5FeatureCacheReader

from .models import BoutFeatureMatrix, BoutRecord

if TYPE_CHECKING:
    from jabs.project import Project

logger = logging.getLogger(__name__)

STAT_SUFFIXES: tuple[str, ...] = (
    "median",
    "iqr",
    "q10",
    "q90",
    "mean",
    "mad",
    "half_delta",
)

_INTERNAL_META_PREFIXES: tuple[str, ...] = (
    "_jabs_",
    "point_mask",
)

_SOCIAL_PREFIXES: tuple[str, ...] = tuple(SocialFeatureGroup._features.keys())

_SEGMENTATION_PREFIXES: tuple[str, ...] = tuple(
    dict.fromkeys([*SegmentationFeatureGroup._features.keys(), HuddleFeatures.name()])
)

# In-memory LRU cache of loaded per-track feature matrices with strict byte budget:
# key -> (cache_file_path, identity, mtime_ns, use_social, use_segmentation, disabled_landmarks)
_TRACK_ARRAY_CACHE: OrderedDict[
    tuple[str, int, int, bool, bool, tuple[str, ...]],
    tuple[np.ndarray, np.ndarray, list[str]],
] = OrderedDict()
_TRACK_CACHE_LOCK = threading.Lock()
_MAX_CACHED_TRACKS = 512
_MAX_TRACK_CACHE_BYTES = 256 * 1024 * 1024  # 256 MB hard budget
_TRACK_CACHE_BYTES = 0

# O(1) per-Project video-name resolver cache (eliminates O(V^2) resolution scans)
_VIDEO_NAME_INDEX_CACHE: dict[int, tuple[tuple[int, str, str], set[str], dict[str, str]]] = {}


_KNOWN_VIDEO_EXTS: set[str] = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".wmv",
    ".mpg",
    ".mpeg",
}
_KNOWN_SIDE_EXTS: set[str] = _KNOWN_VIDEO_EXTS | {
    ".json",
    ".h5",
    ".hdf5",
    ".parquet",
}


def strip_known_video_suffix(name: str) -> str:
    """Strip at most one trailing video/sidecar extension from `name`.

    Preserves multi-dot video stems such as `...21.19.top` or `...21.08` so
    `Path(...).stem` never truncates `.top` or timestamp suffixes.
    """
    base = Path(name).name
    suf = Path(base).suffix
    if suf.lower() in _KNOWN_SIDE_EXTS:
        return base[: -len(suf)]
    return base


def _entry_nbytes(entry: tuple[np.ndarray, np.ndarray, list[str]]) -> int:
    return int(entry[0].nbytes + entry[1].nbytes)


def _resolve_cache_stem(video_name: str) -> str:
    """Resolve the feature cache subdirectory stem using JABS's `pose_file_stem` convention."""
    base = Path(video_name).name
    if Path(base).suffix.lower() in _KNOWN_SIDE_EXTS:
        return pose_file_stem(base)
    return strip_known_video_suffix(base)


def _disabled_landmark_prefixes(static_objects: dict[str, Any] | None) -> tuple[str, ...]:
    """Return landmark feature prefixes explicitly disabled in behavior `static_objects` config."""
    if not isinstance(static_objects, dict) or not static_objects:
        return ()
    disabled: list[str] = []
    for obj_name, enabled in static_objects.items():
        if not bool(enabled):
            disabled.extend(LandmarkFeatureGroup.static_object_features(str(obj_name)))
    return tuple(dict.fromkeys(sorted(disabled)))


def clear_bout_extractor_cache() -> None:
    """Clear the in-memory track feature array cache and video-name index."""
    global _TRACK_CACHE_BYTES
    with _TRACK_CACHE_LOCK:
        _TRACK_ARRAY_CACHE.clear()
        _TRACK_CACHE_BYTES = 0
        _VIDEO_NAME_INDEX_CACHE.clear()


def invalidate_track_cache(
    project: Project | None = None,
    video: str | None = None,
    identity: int | None = None,
) -> None:
    """Invalidate cached track arrays for `(video, identity)` (or all if omitted)."""
    global _TRACK_CACHE_BYTES
    with _TRACK_CACHE_LOCK:
        if video is None and identity is None:
            _TRACK_ARRAY_CACHE.clear()
            _TRACK_CACHE_BYTES = 0
            return
        stem = _resolve_cache_stem(video) if video else None
        to_remove = [
            k
            for k in list(_TRACK_ARRAY_CACHE.keys())
            if (
                stem is None
                or Path(k[0]).parent.parent.name == stem
                or f"/{stem}/" in k[0].replace("\\", "/")
            )
            and (identity is None or k[1] == int(identity))
        ]
        for k in to_remove:
            removed = _TRACK_ARRAY_CACHE.pop(k, None)
            if removed is not None:
                _TRACK_CACHE_BYTES = max(0, _TRACK_CACHE_BYTES - _entry_nbytes(removed))


def _get_video_name_lookup(project: Project) -> tuple[set[str], dict[str, str]]:
    """Build or reuse an O(1) `(exact_names, stem_to_name)` index for `project.video_manager`."""
    try:
        vm = project.video_manager
        raw_videos = getattr(vm, "videos", []) or []
        n_vids = len(raw_videos)
        first_v = str(raw_videos[0]) if n_vids > 0 else ""
        last_v = str(raw_videos[-1]) if n_vids > 0 else ""
    except Exception:
        return set(), {}

    sig = (n_vids, first_v, last_v)
    vm_key = id(vm)
    cached = _VIDEO_NAME_INDEX_CACHE.get(vm_key)
    if cached is not None and cached[0] == sig:
        return cached[1], cached[2]

    exact_set: set[str] = set()
    stem_map: dict[str, str] = {}
    for v in raw_videos:
        v_name = Path(str(v)).name
        exact_set.add(v_name)
        v_stem = strip_known_video_suffix(v_name)
        stem_map.setdefault(v_stem, v_name)

    if len(_VIDEO_NAME_INDEX_CACHE) >= 32:
        _VIDEO_NAME_INDEX_CACHE.clear()
    _VIDEO_NAME_INDEX_CACHE[vm_key] = (sig, exact_set, stem_map)
    return exact_set, stem_map


def resolve_project_video_name(
    project: Project,
    stem_or_name: str,
    doc: dict[str, object] | None = None,
    *,
    anno_doc: dict[str, object] | None = None,
) -> str:
    """Resolve the exact video filename (e.g. `.mp4`, `.avi`, `.mov`) for `stem_or_name`."""
    eff_doc = doc if doc is not None else anno_doc
    exact_names, stem_map = _get_video_name_lookup(project)

    if eff_doc is not None:
        doc_file = eff_doc.get("file")
        if isinstance(doc_file, str) and doc_file.strip():
            doc_clean = doc_file.strip()
            doc_base = Path(doc_clean).name
            if doc_clean in exact_names:
                return doc_clean
            if doc_base in exact_names:
                return doc_base
            if Path(doc_base).suffix.lower() in _KNOWN_VIDEO_EXTS:
                return doc_base
            doc_stem = strip_known_video_suffix(doc_base)
            matched = stem_map.get(doc_stem)
            if matched is not None:
                return matched

    name = Path(stem_or_name).name
    target_stem = strip_known_video_suffix(name)

    if stem_or_name in exact_names:
        return stem_or_name
    if name in exact_names:
        return name
    matched = stem_map.get(target_stem)
    if matched is not None:
        return matched

    if Path(name).suffix.lower() in _KNOWN_VIDEO_EXTS:
        return name
    return f"{target_stem}.mp4"


def resolve_behavior_settings(project: Project, behavior: str | None) -> dict[str, Any]:
    """Resolve behavior feature/window settings with Multi-Class and `defaults` fallbacks."""
    sm = project.settings_manager
    behavior_names = list(getattr(sm, "behavior_names", []) or [])
    raw_proj_settings = getattr(sm, "project_settings", None)
    defaults: dict[str, Any] = {}
    if isinstance(raw_proj_settings, dict):
        raw_defaults = raw_proj_settings.get("defaults", {})
        if isinstance(raw_defaults, dict):
            defaults = dict(raw_defaults)

    candidates: list[str] = []
    if behavior and (
        behavior in behavior_names or behavior not in (MULTICLASS_NONE_BEHAVIOR, "__ALL_MC__")
    ):
        candidates.append(behavior)
    for b_name in behavior_names:
        if b_name not in candidates:
            candidates.append(b_name)

    for cand in candidates:
        try:
            cfg = sm.get_behavior(cand)
        except Exception:
            cfg = None
        if isinstance(cfg, dict) and cfg:
            return {**defaults, **cfg}
    return defaults


def _select_active_feature_columns(
    all_columns: list[str],
    *,
    use_social: bool = True,
    use_segmentation: bool = True,
    disabled_landmark_prefixes: tuple[str, ...] = (),
) -> list[str]:
    """Filter per-frame feature columns according to behavior feature & static-object flags."""
    selected: list[str] = []
    for col in all_columns:
        if any(col.startswith(p) for p in _INTERNAL_META_PREFIXES):
            continue
        if not use_social and any(col.startswith(p) for p in _SOCIAL_PREFIXES):
            continue
        if not use_segmentation and any(col.startswith(p) for p in _SEGMENTATION_PREFIXES):
            continue
        if disabled_landmark_prefixes and any(
            col.startswith(p) for p in disabled_landmark_prefixes
        ):
            continue
        selected.append(col)
    return selected


def _cache_get_track_array(
    cache_key: tuple[str, int, int, bool, bool, tuple[str, ...]],
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    with _TRACK_CACHE_LOCK:
        cached = _TRACK_ARRAY_CACHE.get(cache_key)
        if cached is not None:
            _TRACK_ARRAY_CACHE.move_to_end(cache_key)
        return cached


def _cache_put_track_array(
    cache_key: tuple[str, int, int, bool, bool, tuple[str, ...]],
    result: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    global _TRACK_CACHE_BYTES
    entry_bytes = _entry_nbytes(result)
    with _TRACK_CACHE_LOCK:
        old_existing = _TRACK_ARRAY_CACHE.pop(cache_key, None)
        if old_existing is not None:
            _TRACK_CACHE_BYTES = max(0, _TRACK_CACHE_BYTES - _entry_nbytes(old_existing))
        while _TRACK_ARRAY_CACHE and (
            len(_TRACK_ARRAY_CACHE) >= _MAX_CACHED_TRACKS
            or (_TRACK_CACHE_BYTES + entry_bytes) > _MAX_TRACK_CACHE_BYTES
        ):
            _, evicted = _TRACK_ARRAY_CACHE.popitem(last=False)
            _TRACK_CACHE_BYTES = max(0, _TRACK_CACHE_BYTES - _entry_nbytes(evicted))
        if entry_bytes <= _MAX_TRACK_CACHE_BYTES:
            _TRACK_ARRAY_CACHE[cache_key] = result
            _TRACK_CACHE_BYTES += entry_bytes


def load_track_per_frame_array(
    project_dir: Path,
    video_name: str,
    identity: int,
    *,
    use_social: bool = True,
    use_segmentation: bool = True,
    disabled_landmark_prefixes: tuple[str, ...] = (),
    feature_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Load `(X_track, valid_mask, feature_names)` for one `(video, identity)` track.

    Uses `detect_cache_format` + Parquet / `HDF5FeatureCacheReader` under
    `<feature_dir>/<stem>/<identity>/`, backed by a byte-budgeted LRU cache keyed by
    `(cache_path, identity, mtime_ns, use_social, use_segmentation, disabled_landmark_prefixes)`.
    """
    eff_feature_dir = (
        Path(feature_dir) if feature_dir is not None else (project_dir / "jabs" / "features")
    )
    stem = _resolve_cache_stem(video_name)
    identity_dir = eff_feature_dir / stem / str(identity)
    if not identity_dir.is_dir():
        return None

    fmt = detect_cache_format(identity_dir)
    parquet_path = identity_dir / "per_frame.parquet"
    h5_path = identity_dir / "features.h5"

    if fmt == CacheFormat.PARQUET or (fmt is None and parquet_path.is_file()):
        if not parquet_path.is_file():
            return None
        mtime_ns = parquet_path.stat().st_mtime_ns
        cache_key = (
            str(parquet_path),
            int(identity),
            mtime_ns,
            use_social,
            use_segmentation,
            disabled_landmark_prefixes,
        )
        cached = _cache_get_track_array(cache_key)
        if cached is not None:
            return cached

        schema = pq.read_schema(parquet_path)
        all_cols = schema.names
        feat_cols = _select_active_feature_columns(
            all_cols,
            use_social=use_social,
            use_segmentation=use_segmentation,
            disabled_landmark_prefixes=disabled_landmark_prefixes,
        )
        read_cols = list(feat_cols)
        has_valid_col = "_jabs_frame_valid" in all_cols
        if has_valid_col:
            read_cols = ["_jabs_frame_valid", *feat_cols]

        table = pq.read_table(parquet_path, columns=read_cols, use_threads=True)
        n_rows = table.num_rows

        if has_valid_col:
            valid_arr = table.column("_jabs_frame_valid").to_numpy(zero_copy_only=False)
            valid_mask = np.asarray(valid_arr, dtype=np.uint8) > 0
        else:
            valid_mask = np.ones(n_rows, dtype=bool)

        if feat_cols and n_rows > 0:
            X_track = np.empty((n_rows, len(feat_cols)), dtype=np.float32)
            for j, c_name in enumerate(feat_cols):
                col_np = table.column(c_name).to_numpy(zero_copy_only=False)
                X_track[:, j] = np.asarray(col_np, dtype=np.float32)
            np.nan_to_num(X_track, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X_track = np.zeros((n_rows, len(feat_cols)), dtype=np.float32)

        result = (X_track, valid_mask, feat_cols)
        _cache_put_track_array(cache_key, result)
        return result

    if fmt == CacheFormat.HDF5 or h5_path.is_file():
        if not h5_path.is_file():
            return None
        mtime_ns = h5_path.stat().st_mtime_ns
        cache_key = (
            str(h5_path),
            int(identity),
            mtime_ns,
            use_social,
            use_segmentation,
            disabled_landmark_prefixes,
        )
        cached = _cache_get_track_array(cache_key)
        if cached is not None:
            return cached

        info = inspect_identity_cache(identity_dir)
        raw_features: dict[str, np.ndarray] | None = None
        valid_mask: np.ndarray | None = None
        if info is not None and info.cache_format == CacheFormat.HDF5:
            try:
                reader = HDF5FeatureCacheReader(
                    expected_feature_version=info.feature_version,
                    expected_pose_hash=info.pose_hash,
                    expected_distance_scale_factor=info.distance_scale_factor,
                )
                pf_data = reader.read_per_frame(identity_dir)
                raw_features = pf_data.features
                valid_mask = np.asarray(pf_data.frame_valid, dtype=np.uint8) > 0
            except Exception:
                raw_features = None
                valid_mask = None

        if raw_features is None:
            with h5py.File(h5_path, "r") as h5f:
                if "features" not in h5f or "per_frame" not in h5f["features"]:
                    return None
                pf_grp = h5f["features"]["per_frame"]
                raw_features = {k: pf_grp[k][:] for k in pf_grp}
                if "frame_valid" in h5f:
                    valid_mask = np.asarray(h5f["frame_valid"][:], dtype=np.uint8) > 0

        all_cols = sorted(raw_features.keys())
        feat_cols = _select_active_feature_columns(
            all_cols,
            use_social=use_social,
            use_segmentation=use_segmentation,
            disabled_landmark_prefixes=disabled_landmark_prefixes,
        )
        if not feat_cols:
            return None
        first_arr = np.asarray(raw_features[feat_cols[0]])
        n_rows = int(first_arr.shape[0])
        if valid_mask is None:
            valid_mask = np.ones(n_rows, dtype=bool)
        X_track = np.empty((n_rows, len(feat_cols)), dtype=np.float32)
        for j, col_name in enumerate(feat_cols):
            arr = np.asarray(raw_features[col_name], dtype=np.float32).reshape(n_rows, -1)[:, 0]
            X_track[:, j] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        result = (X_track, valid_mask, feat_cols)
        _cache_put_track_array(cache_key, result)
        return result

    return None


def compute_fast_bout_quantiles(
    X_slice: np.ndarray,
    valid_slice: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute the 7 duration-invariant bout statistics + `log2_duration` + `valid_pose_fraction`.

    Uses a single C-level `np.sort(..., axis=0)` call on the valid frames of the bout slice
    for linear-interpolated order statistics (`q10, q25, q50, q75, q90`), achieving a >20x
    speedup over `np.nanpercentile`.

    Returns:
        `(bout_vec, valid_pose_fraction)` where `bout_vec` has length `7 * F + 2`.
    """
    n_total, n_feats = X_slice.shape
    if n_total == 0:
        return np.zeros(7 * n_feats + 2, dtype=np.float32), 0.0

    n_valid = int(np.count_nonzero(valid_slice))
    valid_fraction = float(n_valid) / float(n_total)

    sub = X_slice[valid_slice] if n_valid >= 2 else X_slice
    m = sub.shape[0]

    if m == 1:
        row = sub[0]
        zeros = np.zeros_like(row)
        stats_7 = np.concatenate([row, zeros, row, row, row, zeros, zeros])
    else:
        s_sorted = np.sort(sub, axis=0)

        def _interp_q(p: float) -> np.ndarray:
            pos = p * (m - 1)
            lo = int(np.floor(pos))
            hi = min(lo + 1, m - 1)
            frac = np.float32(pos - lo)
            return s_sorted[lo] * (np.float32(1.0) - frac) + s_sorted[hi] * frac

        q10 = _interp_q(0.10)
        q25 = _interp_q(0.25)
        q50 = _interp_q(0.50)
        q75 = _interp_q(0.75)
        q90 = _interp_q(0.90)
        iqr = q75 - q25
        mean = np.mean(sub, axis=0, dtype=np.float32)

        abs_dev_sorted = np.sort(np.abs(sub - q50), axis=0)
        pos50 = 0.50 * (m - 1)
        lo50 = int(np.floor(pos50))
        hi50 = min(lo50 + 1, m - 1)
        frac50 = np.float32(pos50 - lo50)
        mad = abs_dev_sorted[lo50] * (np.float32(1.0) - frac50) + abs_dev_sorted[hi50] * frac50

        mid = max(1, m // 2)
        half_delta = np.mean(sub[mid:], axis=0, dtype=np.float32) - np.mean(
            sub[:mid], axis=0, dtype=np.float32
        )

        # Layout: [median(F), iqr(F), q10(F), q90(F), mean(F), mad(F), half_delta(F)]
        stats_7 = np.concatenate([q50, iqr, q10, q90, mean, mad, half_delta])

    meta = np.array(
        [np.log2(float(n_total) + 1.0), valid_fraction],
        dtype=np.float32,
    )
    return np.concatenate([stats_7.astype(np.float32, copy=False), meta]), valid_fraction


def collect_labeled_bout_records(
    project: Project,
    behavior: str | None = None,
    *,
    include_positive: bool = True,
    include_negative: bool = True,
    include_all_multiclass: bool = False,
    min_bout_frames: int = 2,
) -> list[BoutRecord]:
    """Scan `jabs/annotations/*.json` and return `BoutRecord` items for human-labeled bouts."""
    project_dir = Path(project.project_paths.project_dir)
    anno_dir = project_dir / "jabs" / "annotations"
    is_multiclass = project.settings_manager.classifier_mode == ClassifierMode.MULTICLASS
    behavior_names = list(project.settings_manager.behavior_names)

    if behavior is None and behavior_names:
        behavior = behavior_names[0]

    records: list[BoutRecord] = []
    bout_id = 0

    for anno_path in sorted(anno_dir.glob("*.json")):
        doc = read_document(anno_path)
        if not doc:
            continue
        video_name = resolve_project_video_name(project, anno_path.name, doc)
        if project.settings_manager.is_video_excluded(video_name):
            continue

        unfrag = doc.get("unfragmented_labels") or doc.get("labels") or {}
        for id_str, beh_map in sorted(
            unfrag.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0
        ):
            try:
                identity = int(id_str)
            except ValueError:
                continue

            if is_multiclass:
                target_behaviors = (
                    [*behavior_names, MULTICLASS_NONE_BEHAVIOR]
                    if include_all_multiclass or behavior is None
                    else [behavior, MULTICLASS_NONE_BEHAVIOR]
                )
                for beh_key in target_behaviors:
                    for blk in beh_map.get(beh_key, []):
                        if not blk.get("present", False):
                            continue
                        s, e = int(blk["start"]), int(blk["end"])
                        dur = e - s + 1
                        if dur < min_bout_frames:
                            continue
                        is_pos = beh_key != MULTICLASS_NONE_BEHAVIOR
                        if is_pos and not include_positive:
                            continue
                        if not is_pos and not include_negative:
                            continue
                        records.append(
                            BoutRecord(
                                bout_id=bout_id,
                                source="labeled",
                                video=video_name,
                                identity=identity,
                                behavior=beh_key
                                if is_pos
                                else (behavior or MULTICLASS_NONE_BEHAVIOR),
                                class_name=beh_key,
                                is_positive=is_pos,
                                start_frame=s,
                                end_frame=e,
                                duration_frames=dur,
                            )
                        )
                        bout_id += 1
            else:
                if not behavior or behavior not in beh_map:
                    continue
                for blk in beh_map.get(behavior, []):
                    is_pos = bool(blk.get("present", False))
                    if is_pos and not include_positive:
                        continue
                    if not is_pos and not include_negative:
                        continue
                    s, e = int(blk["start"]), int(blk["end"])
                    dur = e - s + 1
                    if dur < min_bout_frames:
                        continue
                    records.append(
                        BoutRecord(
                            bout_id=bout_id,
                            source="labeled",
                            video=video_name,
                            identity=identity,
                            behavior=behavior,
                            class_name=behavior if is_pos else f"Not-{behavior}",
                            is_positive=is_pos,
                            start_frame=s,
                            end_frame=e,
                            duration_frames=dur,
                        )
                    )
                    bout_id += 1

    return records


def extract_bout_feature_matrix(
    project: Project,
    records: list[BoutRecord],
    behavior: str | None = None,
    *,
    retain_frame_features: bool = True,
) -> BoutFeatureMatrix:
    """Extract the bout-level summary feature matrix `(N_bouts, D_stats)` for `records`.

    Groups `records` by `(video, identity)` so each `per_frame.parquet` is opened at most once.
    """
    pp = getattr(project, "project_paths", None)
    project_dir = Path(pp.project_dir) if pp is not None else Path(project.dir)
    raw_feat_dir = getattr(pp, "feature_dir", None) if pp is not None else None
    feature_dir = (
        Path(raw_feat_dir)
        if isinstance(raw_feat_dir, str | Path)
        else (project_dir / "jabs" / "features")
    )
    beh_settings = resolve_behavior_settings(project, behavior)
    use_social = bool(beh_settings.get("social", True))
    use_segmentation = bool(beh_settings.get("segmentation", True))
    disabled_landmarks = _disabled_landmark_prefixes(beh_settings.get("static_objects"))

    # Group bout indices by (video, identity) for single-open I/O
    by_track: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        by_track[(rec.video, rec.identity)].append(idx)

    base_feature_names: list[str] | None = None
    valid_records: list[BoutRecord] = []
    bout_vectors: list[np.ndarray] = []
    frame_features_by_bout: dict[int, np.ndarray] = {}

    # Temporary map from original record index to extracted data so we preserve record order
    extracted_by_idx: dict[int, tuple[np.ndarray, float, np.ndarray | None]] = {}

    # PyArrow's `pq.read_table(..., use_threads=True)` already parallelizes column decoding in C++
    # without spawning Python threads that collide with PySide6 Shiboken's lazy import hook.

    for (video_name, identity), rec_indices in by_track.items():
        loaded = load_track_per_frame_array(
            project_dir,
            video_name,
            identity,
            use_social=use_social,
            use_segmentation=use_segmentation,
            disabled_landmark_prefixes=disabled_landmarks,
            feature_dir=feature_dir,
        )
        if loaded is None:
            continue
        X_track, valid_mask, feat_cols = loaded
        if base_feature_names is None:
            base_feature_names = feat_cols
        elif feat_cols != base_feature_names:
            # Align columns if a video has a slightly different column subset
            col_to_idx = {c: i for i, c in enumerate(feat_cols)}
            aligned = np.zeros((X_track.shape[0], len(base_feature_names)), dtype=np.float32)
            for j, c_name in enumerate(base_feature_names):
                if c_name in col_to_idx:
                    aligned[:, j] = X_track[:, col_to_idx[c_name]]
            X_track = aligned

        n_frames_track = X_track.shape[0]
        for r_idx in rec_indices:
            rec = records[r_idx]
            s = max(0, min(rec.start_frame, n_frames_track - 1))
            e = max(s, min(rec.end_frame, n_frames_track - 1))
            X_slice = X_track[s : e + 1]
            v_slice = valid_mask[s : e + 1]
            vec, valid_frac = compute_fast_bout_quantiles(X_slice, v_slice)
            extracted_by_idx[r_idx] = (
                vec,
                valid_frac,
                X_slice if retain_frame_features else None,
            )

    if base_feature_names is None:
        base_feature_names = []

    stat_names: list[str] = []
    stat_to_base_idx_list: list[int] = []
    for suffix in STAT_SUFFIXES:
        for b_idx, b_name in enumerate(base_feature_names):
            stat_names.append(f"{b_name}_{suffix}")
            stat_to_base_idx_list.append(b_idx)
    stat_names.extend(["bout_log2_duration", "bout_valid_pose_fraction"])
    stat_to_base_idx_list.extend([-1, -1])

    new_bout_id = 0
    for r_idx, rec in enumerate(records):
        if r_idx not in extracted_by_idx:
            continue
        vec, valid_frac, f_slice = extracted_by_idx[r_idx]
        rec.bout_id = new_bout_id
        rec.valid_pose_fraction = valid_frac
        valid_records.append(rec)
        bout_vectors.append(vec)
        if f_slice is not None:
            frame_features_by_bout[new_bout_id] = f_slice
        new_bout_id += 1

    if bout_vectors:
        X_bout = np.vstack(bout_vectors).astype(np.float32, copy=False)
    else:
        X_bout = np.zeros((0, len(stat_names)), dtype=np.float32)

    return BoutFeatureMatrix(
        records=valid_records,
        X_bout=X_bout,
        stat_names=stat_names,
        base_feature_names=base_feature_names,
        stat_to_base_idx=np.asarray(stat_to_base_idx_list, dtype=np.int32),
        frame_features_by_bout=frame_features_by_bout,
    )
