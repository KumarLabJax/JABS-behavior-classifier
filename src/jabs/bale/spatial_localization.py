"""Fast cached 2D arena spatial localization extractor for `jabs.bale`.

Computes per-bout median body centroid and nose coordinates (`(x, y)` in pixels,
or normalized `[0, 1]` when mixing video resolutions) alongside optional arena
reference frame imagery and static landmarks (`corners`, `food_hopper`, `lixit`).

Design prioritizes sub-second latency on large (100-465 video) projects:
1. Bounded LRU caches keyed by `(resolved_path, mtime_ns, size)` of the exact
   backing file loaded avoid stale entries across retracked pose files.
2. Fast ISO-BMFF container header parsing reads MP4/MOV `(width, height)` in
   `< 0.3 ms` per video (falling back to `cv2.VideoCapture` only when needed).
3. Sliced three-tier pose reader:
   - Tier 1 (Fastest) slices `[s_min:e_max+1, :, :10, :]` from uncompressed
     raw `v4-v8` HDF5 files (`points`, `confidence`, `id_mask`, `instance_embed_id`),
     completely skipping heavy `seg_data` masks.
   - Tier 2 falls back to JABS `*_cache.h5` (`cache_file_version in (4, 5)`).
   - Tier 3 falls back to `load_pose_cached(project, video)` for `v2`/`v3` or
     custom/mocked `Project` implementations.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import h5py
import numpy as np

from .bout_extractor import strip_known_video_suffix
from .models import BoutRecord, BoutSpatialResult
from .trajectory_and_export import load_pose_cached

if TYPE_CHECKING:
    from jabs.project import Project

logger = logging.getLogger(__name__)

_MINIMUM_CONFIDENCE = 0.3
_MIN_VALID_KEYPOINTS = 3
_MAX_SPATIAL_VIDEO_CACHE = 256
_MAX_REF_FRAME_CACHE = 8
_MAX_VIDEO_DIM_CACHE = 512
_ACCEPTED_CACHE_VERSIONS = (4, 5)
_OBJECTS_STORED_YX = ("lixit", "food_hopper")
_MP4_CONTAINER_SUFFIXES = {".mp4", ".mov", ".m4v"}


@dataclass
class _CachedVideoSpatial:
    """Lightweight per-video `(N_id, T_slice, 2)` spatial coordinate cache."""

    s_offset: int
    n_frames_total: int
    body_xy: np.ndarray  # shape: (N_id, T_slice, 2), dtype=float32
    nose_xy: np.ndarray  # shape: (N_id, T_slice, 2), dtype=float32
    cm_per_pixel: float | None = None
    static_objects: dict[str, np.ndarray] = field(default_factory=dict)


_SPATIAL_LOCK = threading.Lock()
_SPATIAL_VIDEO_CACHE: OrderedDict[tuple[str, int, int], _CachedVideoSpatial] = OrderedDict()
_REF_FRAME_CACHE: OrderedDict[tuple[str, int, int, int], tuple[np.ndarray, int, int]] = (
    OrderedDict()
)
_VIDEO_DIM_CACHE: OrderedDict[tuple[str, int, int], tuple[float, float]] = OrderedDict()


def clear_spatial_cache() -> None:
    """Clear all in-memory spatial pose, dimension, and reference-frame caches."""
    with _SPATIAL_LOCK:
        _SPATIAL_VIDEO_CACHE.clear()
        _REF_FRAME_CACHE.clear()
        _VIDEO_DIM_CACHE.clear()


def _lru_get(cache: OrderedDict[Any, Any], key: Any) -> Any:
    with _SPATIAL_LOCK:
        val = cache.get(key)
        if val is not None:
            cache.move_to_end(key)
        return val


def _lru_put(cache: OrderedDict[Any, Any], key: Any, val: Any, max_size: int) -> None:
    with _SPATIAL_LOCK:
        cache[key] = val
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)


def _file_stat_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size))


def _resolve_cache_dir(project: Any) -> Path | None:
    for attr in ("_paths", "project_paths"):
        holder = getattr(project, attr, None)
        if holder is not None:
            cdir = getattr(holder, "cache_dir", None)
            if isinstance(cdir, Path):
                return cdir
    proj_dir = getattr(project, "dir", None)
    if isinstance(proj_dir, Path):
        cand = proj_dir / "jabs" / "cache"
        if cand.is_dir():
            return cand
    return None


def _resolve_pose_path(project: Any, video_name: str) -> Path | None:
    vm = getattr(project, "video_manager", None)
    if vm is not None and hasattr(vm, "get_cached_pose_path"):
        try:
            p = Path(vm.get_cached_pose_path(video_name))
            if p.is_file():
                return p
        except Exception:
            pass
    proj_dir = getattr(project, "dir", None)
    if isinstance(proj_dir, Path):
        stem = strip_known_video_suffix(video_name)
        for match in sorted(proj_dir.glob(f"{stem}_pose_est_v*.h5")):
            if match.is_file():
                return match
    return None


def _resolve_video_path(project: Any, video_name: str) -> Path | None:
    vm = getattr(project, "video_manager", None)
    if vm is not None and hasattr(vm, "video_path"):
        try:
            p = Path(vm.video_path(video_name))
            if p.is_file():
                return p
        except Exception:
            pass
    proj_dir = getattr(project, "dir", None)
    if isinstance(proj_dir, Path):
        p = proj_dir / Path(video_name).name
        if p.is_file():
            return p
    return None


def _sort_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    """Sort 2D polygon vertices monotonically around their centroid."""
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(arr) <= 2:
        return arr
    cx, cy = np.mean(arr, axis=0)
    angles = np.arctan2(arr[:, 1] - cy, arr[:, 0] - cx)
    return arr[np.argsort(angles)]


def _normalize_static_objects_dict(
    raw: dict[str, Any], *, raw_hdf5_yx: bool
) -> dict[str, np.ndarray]:
    """Convert raw static objects dictionary into clean `(M, 2)` float32 `(x, y)` arrays."""
    out: dict[str, np.ndarray] = {}
    for name, val in raw.items():
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.float32)
        if arr.size < 2:
            continue
        if raw_hdf5_yx and name in _OBJECTS_STORED_YX:
            arr = np.flip(arr, axis=-1)
        arr_2d = np.ascontiguousarray(arr.reshape(-1, 2), dtype=np.float32)
        valid = np.isfinite(arr_2d[:, 0]) & np.isfinite(arr_2d[:, 1])
        if not np.any(valid):
            continue
        arr_2d = arr_2d[valid]
        if name == "corners" and len(arr_2d) >= 3:
            arr_2d = _sort_corners_clockwise(arr_2d)
        out[str(name)] = arr_2d
    return out


def _read_static_objects_from_h5(hf: h5py.File) -> dict[str, np.ndarray]:
    if "static_objects" not in hf:
        return {}
    raw: dict[str, Any] = {}
    grp = hf["static_objects"]
    for key in grp:
        try:
            raw[str(key)] = grp[key][:]
        except Exception:
            continue
    return _normalize_static_objects_dict(raw, raw_hdf5_yx=True)


def _compute_body_and_nose_from_identity_arrays(
    pts_xy: np.ndarray,
    kp_valid: np.ndarray,
    ident_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute vectorized `(N_id, T, 2)` body centroid and nose `(x, y)` arrays cleanly."""
    n_id, n_frames, n_kpts, _ = pts_xy.shape
    if n_id == 0 or n_frames == 0 or n_kpts == 0:
        empty = np.full((n_id, n_frames, 2), np.nan, dtype=np.float32)
        return empty, empty.copy()

    mask = np.asarray(kp_valid, dtype=bool)
    if ident_mask is not None:
        mask = mask & np.asarray(ident_mask, dtype=bool)[:, :, None]
    finite_xy = np.isfinite(pts_xy[:, :, :, 0]) & np.isfinite(pts_xy[:, :, :, 1])
    mask = mask & finite_xy

    k_body = min(10, n_kpts)
    body_mask = mask[:, :, :k_body]
    counts = body_mask.sum(axis=2)  # (N_id, T)
    safe_pts = np.where(body_mask[:, :, :, None], pts_xy[:, :, :k_body, :], 0.0)
    sums = safe_pts.sum(axis=2, dtype=np.float32)  # (N_id, T, 2)

    min_req = min(_MIN_VALID_KEYPOINTS, k_body)
    valid_frames = counts >= min_req
    body_xy = np.full((n_id, n_frames, 2), np.nan, dtype=np.float32)
    if np.any(valid_frames):
        body_xy[valid_frames] = sums[valid_frames] / counts[valid_frames, None].astype(np.float32)

    nose_valid = mask[:, :, 0:1] & valid_frames[:, :, None]
    nose_xy = np.where(
        nose_valid,
        pts_xy[:, :, 0, :].astype(np.float32, copy=False),
        np.nan,
    ).astype(np.float32, copy=False)
    return body_xy, nose_xy


def _try_load_from_raw_v4_h5(
    pose_path: Path,
    s_req: int,
    e_req: int,
) -> _CachedVideoSpatial | None:
    """Slice only `[s:e+1, :, :10, :]` from raw `v4-v8` H5, skipping `seg_data`."""
    try:
        with h5py.File(pose_path, "r") as hf:
            if "poseest" not in hf:
                return None
            grp = hf["poseest"]
            required = ("points", "confidence", "id_mask", "instance_embed_id")
            if not all(k in grp for k in required):
                return None
            pts_ds = grp["points"]
            if pts_ds.ndim != 4 or pts_ds.shape[-1] != 2:
                return None
            n_frames_total = int(pts_ds.shape[0])
            n_slots = int(pts_ds.shape[1])
            n_kpts = int(pts_ds.shape[2])
            cm_raw = grp.attrs.get("cm_per_pixel", None)
            cm_val = float(cm_raw) if cm_raw is not None else None
            static_objs = _read_static_objects_from_h5(hf)

            if n_frames_total <= 0 or n_kpts <= 0 or s_req >= n_frames_total or e_req < 0:
                empty = np.full((0, 0, 2), np.nan, dtype=np.float32)
                return _CachedVideoSpatial(
                    s_offset=0,
                    n_frames_total=n_frames_total,
                    body_xy=empty,
                    nose_xy=empty.copy(),
                    cm_per_pixel=cm_val,
                    static_objects=static_objs,
                )

            s0 = max(0, int(s_req))
            e0 = min(n_frames_total - 1, int(e_req))
            k_body = min(10, n_kpts)
            pts_raw = np.asarray(pts_ds[s0 : e0 + 1, :, :k_body, :], dtype=np.float32)
            conf = np.asarray(grp["confidence"][s0 : e0 + 1, :, :k_body], dtype=np.float32)
            id_mask = np.asarray(grp["id_mask"][s0 : e0 + 1, :])
            embed_id = np.asarray(grp["instance_embed_id"][s0 : e0 + 1, :])

            valid_slot = (id_mask == 0) & (embed_id >= 1)
            max_id = int(embed_id[valid_slot].max()) if np.any(valid_slot) else 0
            if "instance_id_center" in grp:
                n_id = max(max_id, int(grp["instance_id_center"].shape[0]))
            else:
                n_id = max_id
    except Exception:
        return None

    n_slice = pts_raw.shape[0]
    if n_id == 0 or n_slice == 0:
        empty = np.full((n_id, n_slice, 2), np.nan, dtype=np.float32)
        return _CachedVideoSpatial(
            s_offset=s0,
            n_frames_total=n_frames_total,
            body_xy=empty,
            nose_xy=empty.copy(),
            cm_per_pixel=cm_val,
            static_objects=static_objs,
        )

    # Flip (y, x) -> (x, y) over required body/nose keypoints
    pts_xy = np.flip(pts_raw, axis=-1)
    finite_pts = np.isfinite(pts_xy[:, :, :, 0]) & np.isfinite(pts_xy[:, :, :, 1])
    kp_valid = (conf > _MINIMUM_CONFIDENCE) & finite_pts & valid_slot[:, :, None]

    counts = kp_valid.sum(axis=2)  # (T_slice, M)
    safe_pts = np.where(kp_valid[:, :, :, None], pts_xy, 0.0)
    slot_sums = safe_pts.sum(axis=2, dtype=np.float32)  # (T_slice, M, 2)
    slot_body = np.full((n_slice, n_slots, 2), np.nan, dtype=np.float32)
    min_req = min(_MIN_VALID_KEYPOINTS, k_body)
    good_slot = valid_slot & (counts >= min_req)
    if np.any(good_slot):
        slot_body[good_slot] = slot_sums[good_slot] / counts[good_slot, None].astype(np.float32)

    slot_nose = np.where(
        kp_valid[:, :, 0:1] & (counts[:, :, None] >= min_req),
        pts_xy[:, :, 0, :],
        np.nan,
    ).astype(np.float32, copy=False)

    body_xy = np.full((n_id, n_slice, 2), np.nan, dtype=np.float32)
    nose_xy = np.full((n_id, n_slice, 2), np.nan, dtype=np.float32)
    f_idx, m_idx = np.nonzero(valid_slot)
    if f_idx.size > 0:
        id_idx = embed_id[f_idx, m_idx].astype(np.int64) - 1
        body_xy[id_idx, f_idx] = slot_body[f_idx, m_idx]
        nose_xy[id_idx, f_idx] = slot_nose[f_idx, m_idx]

    return _CachedVideoSpatial(
        s_offset=s0,
        n_frames_total=n_frames_total,
        body_xy=body_xy,
        nose_xy=nose_xy,
        cm_per_pixel=cm_val,
        static_objects=static_objs,
    )


def _try_load_from_pose_cache_h5(
    cache_path: Path,
    raw_pose_path: Path | None,
    s_req: int,
    e_req: int,
) -> _CachedVideoSpatial | None:
    if not cache_path.is_file():
        return None
    try:
        with h5py.File(cache_path, "r") as ch5:
            ver = ch5.attrs.get("cache_file_version")
            if ver not in _ACCEPTED_CACHE_VERSIONS:
                return None
            if "poseest" not in ch5:
                return None
            grp = ch5["poseest"]
            cm_raw = grp.attrs.get("cm_per_pixel", None)
            cm_val = float(cm_raw) if cm_raw is not None else None
            if "points" not in grp or "point_mask" not in grp:
                return None
            pts_ds = grp["points"]
            if pts_ds.ndim != 4 or pts_ds.shape[-1] != 2:
                return None
            n_frames_total = int(pts_ds.shape[1])
            n_kpts = int(pts_ds.shape[2])
            static_objs = _read_static_objects_from_h5(ch5)

            if n_frames_total <= 0 or n_kpts <= 0 or s_req >= n_frames_total or e_req < 0:
                empty = np.full((int(pts_ds.shape[0]), 0, 2), np.nan, dtype=np.float32)
                return _CachedVideoSpatial(
                    s_offset=0,
                    n_frames_total=n_frames_total,
                    body_xy=empty,
                    nose_xy=empty.copy(),
                    cm_per_pixel=cm_val,
                    static_objects=static_objs,
                )

            s0 = max(0, int(s_req))
            e0 = min(n_frames_total - 1, int(e_req))
            k_body = min(10, n_kpts)
            pts_xy = np.asarray(pts_ds[:, s0 : e0 + 1, :k_body, :], dtype=np.float32)
            pt_mask = np.asarray(grp["point_mask"][:, s0 : e0 + 1, :k_body]) > 0
            ident_mask = (
                np.asarray(grp["identity_mask"][:, s0 : e0 + 1]) > 0
                if "identity_mask" in grp
                else None
            )
    except Exception:
        return None

    if not static_objs and raw_pose_path is not None and raw_pose_path.is_file():
        try:
            with h5py.File(raw_pose_path, "r") as rhf:
                static_objs = _read_static_objects_from_h5(rhf)
        except Exception:
            pass

    body_xy, nose_xy = _compute_body_and_nose_from_identity_arrays(pts_xy, pt_mask, ident_mask)
    return _CachedVideoSpatial(
        s_offset=s0,
        n_frames_total=n_frames_total,
        body_xy=body_xy,
        nose_xy=nose_xy,
        cm_per_pixel=cm_val,
        static_objects=static_objs,
    )


def _load_video_spatial(
    project: Project,
    video_name: str,
    s_req: int,
    e_req: int,
) -> _CachedVideoSpatial | None:
    pose_path = _resolve_pose_path(project, video_name)
    cache_dir = _resolve_cache_dir(project)
    cache_h5_path: Path | None = None
    if cache_dir is not None:
        if pose_path is not None:
            cand = cache_dir / pose_path.name.replace(".h5", "_cache.h5")
            if cand.is_file():
                cache_h5_path = cand
        if cache_h5_path is None:
            stem = strip_known_video_suffix(video_name)
            matches = sorted(cache_dir.glob(f"{stem}_pose_est_*_cache.h5"))
            if matches and matches[0].is_file():
                cache_h5_path = matches[0]

    def _cache_hit_covers(entry: _CachedVideoSpatial | None) -> bool:
        if entry is None:
            return False
        if entry.n_frames_total <= 0 or s_req >= entry.n_frames_total or e_req < 0:
            return True
        s_need = max(0, s_req)
        e_need = min(entry.n_frames_total - 1, e_req)
        return entry.s_offset <= s_need and e_need < (entry.s_offset + entry.body_xy.shape[1])

    # Check Tier 1 (raw uncompressed v4-v8 H5) key first, then Tier 2 (*_cache.h5)
    raw_key = (
        _file_stat_key(pose_path) if (pose_path is not None and pose_path.is_file()) else None
    )
    if raw_key is not None:
        hit = _lru_get(_SPATIAL_VIDEO_CACHE, raw_key)
        if _cache_hit_covers(hit):
            return hit

    cache_key = (
        _file_stat_key(cache_h5_path)
        if (cache_h5_path is not None and cache_h5_path.is_file())
        else None
    )
    if cache_key is not None:
        hit = _lru_get(_SPATIAL_VIDEO_CACHE, cache_key)
        if _cache_hit_covers(hit):
            return hit

    # Tier 1: Direct sliced uncompressed v4-v8 H5 read (2.5x-7.5x faster than gzipped *_cache.h5)
    if pose_path is not None and pose_path.is_file():
        loaded = _try_load_from_raw_v4_h5(pose_path, s_req, e_req)
        if loaded is not None:
            if raw_key is not None:
                _lru_put(_SPATIAL_VIDEO_CACHE, raw_key, loaded, _MAX_SPATIAL_VIDEO_CACHE)
            return loaded

    # Tier 2: JABS *_cache.h5 fallback (v4 or v5)
    if cache_h5_path is not None and cache_h5_path.is_file():
        loaded = _try_load_from_pose_cache_h5(cache_h5_path, pose_path, s_req, e_req)
        if loaded is not None:
            if cache_key is not None:
                _lru_put(_SPATIAL_VIDEO_CACHE, cache_key, loaded, _MAX_SPATIAL_VIDEO_CACHE)
            return loaded

    # Tier 3: Universal fallback (v2/v3 HDF5, future pose formats, or mock projects)
    try:
        pose = load_pose_cached(project, video_name)
    except Exception:
        return None
    identities = sorted(int(i) for i in getattr(pose, "identities", []) or [])
    n_frames = int(getattr(pose, "num_frames", 0) or 0)
    n_id = (max(identities) + 1) if identities else 0
    body_xy = np.full((n_id, n_frames, 2), np.nan, dtype=np.float32)
    nose_xy = np.full((n_id, n_frames, 2), np.nan, dtype=np.float32)
    for ident in identities:
        try:
            pts_i, mask_i = pose.get_identity_poses(ident)
            pts_arr = np.asarray(pts_i, dtype=np.float32)
            msk_arr = np.asarray(mask_i) > 0
            if pts_arr.ndim == 3 and pts_arr.shape[0] > 0:
                b_arr, n_arr = _compute_body_and_nose_from_identity_arrays(
                    pts_arr[None, ...], msk_arr[None, ...]
                )
                t_lim = min(n_frames, b_arr.shape[1]) if n_frames > 0 else b_arr.shape[1]
                if n_frames == 0:
                    body_xy = np.full((n_id, t_lim, 2), np.nan, dtype=np.float32)
                    nose_xy = np.full((n_id, t_lim, 2), np.nan, dtype=np.float32)
                    n_frames = t_lim
                body_xy[ident, :t_lim] = b_arr[0, :t_lim]
                nose_xy[ident, :t_lim] = n_arr[0, :t_lim]
        except Exception:
            continue
    raw_so = getattr(pose, "static_objects", {}) or {}
    cm_raw = getattr(pose, "cm_per_pixel", None)
    loaded = _CachedVideoSpatial(
        s_offset=0,
        n_frames_total=n_frames,
        body_xy=body_xy,
        nose_xy=nose_xy,
        cm_per_pixel=float(cm_raw) if cm_raw is not None else None,
        static_objects=_normalize_static_objects_dict(raw_so, raw_hdf5_yx=False),
    )
    t3_key = raw_key
    if t3_key is None:
        pose_attr = getattr(pose, "pose_file", None)
        if isinstance(pose_attr, Path) and pose_attr.is_file():
            t3_key = _file_stat_key(pose_attr)
    if t3_key is not None:
        _lru_put(_SPATIAL_VIDEO_CACHE, t3_key, loaded, _MAX_SPATIAL_VIDEO_CACHE)
    return loaded


def _iter_iso_boxes(fh: Any, start: int, end: int) -> Sequence[tuple[bytes, int, int]]:
    """Yield `(box_type, payload_start, box_end)` tuples inside `[start, end)`."""
    out: list[tuple[bytes, int, int]] = []
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        hdr = fh.read(8)
        if len(hdr) < 8:
            break
        size, btype = struct.unpack(">I4s", hdr)
        hdr_len = 8
        if size == 1:
            ext = fh.read(8)
            if len(ext) < 8:
                break
            (size,) = struct.unpack(">Q", ext)
            hdr_len = 16
        elif size == 0:
            size = end - pos
        if size < hdr_len:
            break
        box_end = min(end, pos + size)
        out.append((btype, pos + hdr_len, box_end))
        pos = box_end
    return out


def _read_mp4_dimensions(video_path: Path) -> tuple[float, float] | None:
    """Fast ISO-BMFF `moov -> trak -> tkhd` width/height parser (`< 0.3 ms` per MP4)."""
    if video_path.suffix.lower() not in _MP4_CONTAINER_SUFFIXES:
        return None
    try:
        file_size = video_path.stat().st_size
        with video_path.open("rb") as fh:
            for btype, m_start, m_end in _iter_iso_boxes(fh, 0, file_size):
                if btype != b"moov":
                    continue
                for ttype, t_start, t_end in _iter_iso_boxes(fh, m_start, m_end):
                    if ttype != b"trak":
                        continue
                    for ktype, k_start, k_end in _iter_iso_boxes(fh, t_start, t_end):
                        if ktype != b"tkhd":
                            continue
                        k_len = k_end - k_start
                        if k_len < 84:
                            continue
                        fh.seek(k_start)
                        payload = fh.read(min(96, k_len))
                        version = payload[0]
                        off = 76 if version == 0 else 88
                        if len(payload) < off + 8:
                            continue
                        w_fixed, h_fixed = struct.unpack(">II", payload[off : off + 8])
                        w = float(w_fixed) / 65536.0
                        h = float(h_fixed) / 65536.0
                        if w >= 16.0 and h >= 16.0:
                            return (w, h)
                return None
    except Exception:
        return None
    return None


def _get_video_dimensions(video_path: Path) -> tuple[float, float] | None:
    if not video_path.is_file():
        return None
    key = _file_stat_key(video_path)
    hit = _lru_get(_VIDEO_DIM_CACHE, key)
    if hit is not None:
        return hit

    dims = _read_mp4_dimensions(video_path)
    if dims is None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                return None
            w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0.0 and h > 0.0:
                dims = (w, h)
        finally:
            cap.release()

    if dims is not None:
        _lru_put(_VIDEO_DIM_CACHE, key, dims, _MAX_VIDEO_DIM_CACHE)
    return dims


def _load_reference_frame_gray(
    video_path: Path,
    *,
    max_dim: int = 640,
) -> tuple[np.ndarray, float, float] | None:
    if not video_path.is_file():
        return None
    st = video_path.stat()
    key = (str(video_path.resolve()), int(st.st_mtime_ns), int(st.st_size), int(max_dim))
    hit = _lru_get(_REF_FRAME_CACHE, key)
    if hit is not None:
        gray, w_f, h_f = hit
        return gray, float(w_f), float(h_f)

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        w_orig = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.astype(np.uint8, copy=False)
        if w_orig <= 0.0 or h_orig <= 0.0:
            h_orig, w_orig = float(gray.shape[0]), float(gray.shape[1])
        gh, gw = gray.shape[:2]
        scale = min(1.0, float(max_dim) / float(max(gh, gw)))
        if scale < 1.0:
            new_w = max(1, round(gw * scale))
            new_h = max(1, round(gh * scale))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray_u8 = np.ascontiguousarray(gray, dtype=np.uint8)
    finally:
        cap.release()

    _lru_put(_REF_FRAME_CACHE, key, (gray_u8, int(w_orig), int(h_orig)), _MAX_REF_FRAME_CACHE)
    return gray_u8, w_orig, h_orig


def extract_bout_spatial_locations(
    project: Project,
    records: Sequence[BoutRecord],
    *,
    include_reference_frame: bool = True,
    max_ref_dim: int = 640,
    cancel_event: threading.Event | None = None,
) -> BoutSpatialResult:
    """Compute bout median 2D arena `(x, y)` coordinates and arena context for `records`."""
    t0 = time.perf_counter()
    n_bouts = len(records)
    coords_xy = np.full((n_bouts, 2), np.nan, dtype=np.float32)
    nose_xy = np.full((n_bouts, 2), np.nan, dtype=np.float32)
    if n_bouts == 0:
        return BoutSpatialResult(
            coords_xy=coords_xy,
            nose_xy=nose_xy,
            valid_mask=np.zeros(0, dtype=bool),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    bouts_by_video: dict[str, list[tuple[int, BoutRecord]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        bouts_by_video[rec.video].append((idx, rec))

    # Prefetch unique video path resolutions & MP4 container dimensions concurrently (pure I/O)
    vid_paths_by_name: dict[str, Path] = {}
    uncached_dim_paths: list[Path] = []
    seen_uncached_resolved: set[str] = set()
    for video_name in bouts_by_video:
        vp = _resolve_video_path(project, video_name)
        if vp is not None:
            vid_paths_by_name[video_name] = vp
            try:
                vkey = _file_stat_key(vp)
            except OSError:
                continue
            if _lru_get(_VIDEO_DIM_CACHE, vkey) is None and vkey[0] not in seen_uncached_resolved:
                seen_uncached_resolved.add(vkey[0])
                uncached_dim_paths.append(vp)

    if len(uncached_dim_paths) >= 4:
        max_workers = min(8, len(uncached_dim_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for _ in pool.map(_get_video_dimensions, uncached_dim_paths):
                if cancel_event is not None and cancel_event.is_set():
                    break

    ref_static_objects: dict[str, np.ndarray] = {}
    cm_per_pixel: float | None = None
    video_dims: dict[str, tuple[float, float]] = {}
    first_video_candidate: tuple[str, Path] | None = None
    aligned_ref_candidate: tuple[str, Path] | None = None

    for video_name, idx_recs in bouts_by_video.items():
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Spatial localization cancelled")
        s_min = min(int(r.start_frame) for _, r in idx_recs)
        e_max = max(int(r.end_frame) for _, r in idx_recs)
        vid_spatial = _load_video_spatial(project, video_name, s_min, e_max)
        if vid_spatial is None:
            continue

        if cm_per_pixel is None and vid_spatial.cm_per_pixel is not None:
            cm_per_pixel = vid_spatial.cm_per_pixel

        vid_path = vid_paths_by_name.get(video_name)
        if vid_path is not None:
            if first_video_candidate is None:
                first_video_candidate = (video_name, vid_path)
            dims = _get_video_dimensions(vid_path)
            if dims is not None:
                video_dims[video_name] = dims

        if not ref_static_objects and vid_spatial.static_objects:
            ref_static_objects = {k: v.copy() for k, v in vid_spatial.static_objects.items()}
            if vid_path is not None:
                aligned_ref_candidate = (video_name, vid_path)
        elif aligned_ref_candidate is None and vid_path is not None and vid_spatial.static_objects:
            # Prefer a video that has both `.mp4` and `static_objects` so backdrop matches landmarks
            ref_static_objects = {k: v.copy() for k, v in vid_spatial.static_objects.items()}
            aligned_ref_candidate = (video_name, vid_path)

        n_id, n_slice, _ = vid_spatial.body_xy.shape
        n_total = vid_spatial.n_frames_total
        if n_id <= 0 or n_slice <= 0 or n_total <= 0:
            continue
        s_off = vid_spatial.s_offset
        body_arr = vid_spatial.body_xy
        nose_arr = vid_spatial.nose_xy

        for b_idx, rec in idx_recs:
            ident = int(rec.identity)
            if ident < 0 or ident >= n_id:
                continue
            if int(rec.start_frame) >= n_total or int(rec.end_frame) < 0:
                continue
            s = max(0, int(rec.start_frame)) - s_off
            e = min(n_total - 1, int(rec.end_frame)) - s_off
            s = max(0, min(s, n_slice - 1))
            e = max(s, min(e, n_slice - 1))
            seg_b = body_arr[ident, s : e + 1]
            valid_b = np.isfinite(seg_b[:, 0]) & np.isfinite(seg_b[:, 1])
            if np.any(valid_b):
                coords_xy[b_idx] = np.median(seg_b[valid_b], axis=0)
            seg_n = nose_arr[ident, s : e + 1]
            valid_n = np.isfinite(seg_n[:, 0]) & np.isfinite(seg_n[:, 1])
            if np.any(valid_n):
                nose_xy[b_idx] = np.median(seg_n[valid_n], axis=0)

    # Check resolution homogeneity across all videos that reported dimensions
    unique_dims = {
        (round(w, 1), round(h, 1)) for w, h in video_dims.values() if w > 0.0 and h > 0.0
    }
    normalized = len(unique_dims) > 1
    frame_width: float | None = None
    frame_height: float | None = None
    ref_gray: np.ndarray | None = None
    ref_video: str | None = None
    ref_dims: tuple[float, float] | None = None

    chosen_ref = aligned_ref_candidate or first_video_candidate
    if chosen_ref is not None:
        ref_video, chosen_path = chosen_ref
        ref_dims = video_dims.get(ref_video)
        if include_reference_frame and not normalized:
            loaded_ref = _load_reference_frame_gray(chosen_path, max_dim=max_ref_dim)
            if loaded_ref is not None:
                ref_gray, rw, rh = loaded_ref
                ref_dims = (rw, rh)
                video_dims[ref_video] = ref_dims
    elif video_dims:
        ref_video = next(iter(video_dims))
        ref_dims = video_dims[ref_video]

    if normalized:
        for video_name, idx_recs in bouts_by_video.items():
            dims = video_dims.get(video_name) or ref_dims
            if dims is None:
                continue
            w_v, h_v = dims
            if w_v <= 0.0 or h_v <= 0.0:
                continue
            b_idxs = np.asarray([b_i for b_i, _ in idx_recs], dtype=np.int32)
            coords_xy[b_idxs, 0] /= float(w_v)
            coords_xy[b_idxs, 1] /= float(h_v)
            nose_xy[b_idxs, 0] /= float(w_v)
            nose_xy[b_idxs, 1] /= float(h_v)
        # Drop single-video backdrop & pixel static objects when coordinates mix resolutions
        ref_gray = None
        ref_static_objects = {}
        frame_width, frame_height = 1.0, 1.0
    elif ref_dims is not None:
        frame_width, frame_height = float(ref_dims[0]), float(ref_dims[1])
    else:
        # Pose-only fallback: derive bounds cleanly from valid points & static landmarks
        pts_list: list[np.ndarray] = []
        valid_c = np.isfinite(coords_xy[:, 0]) & np.isfinite(coords_xy[:, 1])
        if np.any(valid_c):
            pts_list.append(coords_xy[valid_c])
        for arr in ref_static_objects.values():
            if len(arr) > 0:
                pts_list.append(arr)
        if pts_list:
            all_p = np.vstack(pts_list)
            frame_width = max(100.0, float(np.max(all_p[:, 0])) * 1.05)
            frame_height = max(100.0, float(np.max(all_p[:, 1])) * 1.05)
        else:
            frame_width, frame_height = 800.0, 800.0

    valid_mask = np.isfinite(coords_xy[:, 0]) & np.isfinite(coords_xy[:, 1])
    return BoutSpatialResult(
        coords_xy=coords_xy,
        nose_xy=nose_xy,
        valid_mask=valid_mask,
        frame_width=frame_width,
        frame_height=frame_height,
        normalized=normalized,
        reference_frame_gray=ref_gray,
        reference_video=ref_video,
        static_objects=ref_static_objects,
        cm_per_pixel=cm_per_pixel,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
