"""Fast cached 2D arena spatial localization extractor for `jabs.bale`.

Computes per-bout median body centroid and nose coordinates (`(x, y)` in pixels,
or normalized `[0, 1]` when mixing video resolutions) alongside optional arena
reference frame imagery and static landmarks (`corners`, `food_hopper`, `lixit`).

Design prioritizes sub-second latency and bounded memory on large (100-465 video) projects:
1. Bounded byte-budgeted LRU caches keyed by `(resolved_path, mtime_ns, size)` of the
   exact backing pose file loaded avoid stale entries and cap RAM at 128 MB.
2. Fast ISO-BMFF container header parsing reads MP4/MOV `(width, height)` in
   `< 0.3 ms` per video (falling back to `cv2.VideoCapture` only when needed).
3. Unified pose loading via `load_pose_cached(project, video)` (`Project.load_pose_est`
   / `PoseLike`), avoiding duplicate format-specific parsers or legacy cache globs.
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
import numpy as np
import numpy.typing as npt

from jabs.pose_estimation import find_pose_file

from .models import BoutRecord, BoutSpatialResult
from .trajectory_and_export import load_pose_cached

if TYPE_CHECKING:
    from jabs.project import Project

logger = logging.getLogger(__name__)

_MINIMUM_CONFIDENCE = 0.3
_MIN_VALID_KEYPOINTS = 3
_MAX_SPATIAL_VIDEO_CACHE = 256
_MAX_SPATIAL_CACHE_BYTES = 128 * 1024 * 1024  # 128 MB hard budget
_MAX_REF_FRAME_CACHE = 8
_MAX_VIDEO_DIM_CACHE = 512
_MP4_CONTAINER_SUFFIXES = {".mp4", ".mov", ".m4v"}


@dataclass
class _CachedVideoSpatial:
    """Lightweight per-video `(N_id, T_slice, 2)` spatial coordinate cache."""

    s_offset: int
    n_frames_total: int
    body_xy: npt.NDArray[np.float32]  # shape: (N_id, T_slice, 2)
    nose_xy: npt.NDArray[np.float32]  # shape: (N_id, T_slice, 2)
    cm_per_pixel: float | None = None
    static_objects: dict[str, npt.NDArray[np.float32]] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        """Total heap footprint in bytes of cached arrays for LRU budget accounting."""
        so_bytes = sum(int(arr.nbytes) for arr in self.static_objects.values())
        return int(self.body_xy.nbytes + self.nose_xy.nbytes + so_bytes)


_SPATIAL_LOCK = threading.Lock()
_SPATIAL_VIDEO_CACHE: OrderedDict[tuple[str, int, int], _CachedVideoSpatial] = OrderedDict()
_SPATIAL_CACHE_BYTES = 0
_REF_FRAME_CACHE: OrderedDict[
    tuple[str, int, int, int], tuple[npt.NDArray[np.uint8], int, int]
] = OrderedDict()
_VIDEO_DIM_CACHE: OrderedDict[tuple[str, int, int], tuple[float, float]] = OrderedDict()


def clear_spatial_cache() -> None:
    """Clear all in-memory spatial pose, dimension, and reference-frame caches."""
    global _SPATIAL_CACHE_BYTES
    with _SPATIAL_LOCK:
        _SPATIAL_VIDEO_CACHE.clear()
        _SPATIAL_CACHE_BYTES = 0
        _REF_FRAME_CACHE.clear()
        _VIDEO_DIM_CACHE.clear()


def _lru_get(cache: OrderedDict[Any, Any], key: Any) -> Any:
    """Fetch `key` from `cache` under `_SPATIAL_LOCK` and mark most-recently used."""
    with _SPATIAL_LOCK:
        val = cache.get(key)
        if val is not None:
            cache.move_to_end(key)
        return val


def _lru_put(cache: OrderedDict[Any, Any], key: Any, val: Any, max_size: int) -> None:
    """Insert `(key, val)` into `cache` under `_SPATIAL_LOCK`, evicting past `max_size`."""
    with _SPATIAL_LOCK:
        cache[key] = val
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)


def _spatial_cache_get(key: tuple[str, int, int]) -> _CachedVideoSpatial | None:
    """Fetch cached spatial coordinates for `key` under `_SPATIAL_LOCK` and mark MRU."""
    with _SPATIAL_LOCK:
        val = _SPATIAL_VIDEO_CACHE.get(key)
        if val is not None:
            _SPATIAL_VIDEO_CACHE.move_to_end(key)
        return val


def _spatial_cache_put(key: tuple[str, int, int], val: _CachedVideoSpatial) -> None:
    """Store `val` in `_SPATIAL_VIDEO_CACHE`, enforcing both byte and entry-count budgets."""
    global _SPATIAL_CACHE_BYTES
    entry_bytes = val.nbytes
    if entry_bytes > _MAX_SPATIAL_CACHE_BYTES:
        return
    with _SPATIAL_LOCK:
        old = _SPATIAL_VIDEO_CACHE.pop(key, None)
        if old is not None:
            _SPATIAL_CACHE_BYTES = max(0, _SPATIAL_CACHE_BYTES - old.nbytes)
        _SPATIAL_VIDEO_CACHE[key] = val
        _SPATIAL_CACHE_BYTES += entry_bytes
        _SPATIAL_VIDEO_CACHE.move_to_end(key)
        while (
            len(_SPATIAL_VIDEO_CACHE) > _MAX_SPATIAL_VIDEO_CACHE
            or _SPATIAL_CACHE_BYTES > _MAX_SPATIAL_CACHE_BYTES
        ):
            _, evicted = _SPATIAL_VIDEO_CACHE.popitem(last=False)
            _SPATIAL_CACHE_BYTES = max(0, _SPATIAL_CACHE_BYTES - evicted.nbytes)


def _file_stat_key(path: Path) -> tuple[str, int, int]:
    """Build a `(resolved_path, mtime_ns, size_bytes)` stat key for `path`."""
    st = path.stat()
    return (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size))


def _resolve_pose_path(project: Any, video_name: str) -> Path | None:
    """Resolve the active pose file path (unified `*_pose.h5` or legacy) for `video_name`."""
    vm = getattr(project, "video_manager", None)
    if vm is not None and hasattr(vm, "get_cached_pose_path"):
        try:
            p = Path(vm.get_cached_pose_path(video_name))
            if p.is_file():
                return p
        except Exception:
            pass
    vid_path = _resolve_video_path(project, video_name)
    if vid_path is not None:
        return find_pose_file(vid_path)
    proj_dir = getattr(project, "dir", None)
    if isinstance(proj_dir, Path):
        return find_pose_file(proj_dir / Path(video_name).name)
    return None


def _resolve_video_path(project: Any, video_name: str) -> Path | None:
    """Resolve the video file path for `video_name` inside `project`."""
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


def _sort_corners_clockwise(pts: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Sort 2D polygon vertices monotonically around their centroid."""
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(arr) <= 2:
        return arr
    cx, cy = np.mean(arr, axis=0)
    angles = np.arctan2(arr[:, 1] - cy, arr[:, 0] - cx)
    return arr[np.argsort(angles)]


def _normalize_static_objects_dict(raw: dict[str, Any]) -> dict[str, npt.NDArray[np.float32]]:
    """Convert `(x, y)` `PoseLike.static_objects` dictionary into clean `(M, 2)` float32 arrays."""
    out: dict[str, npt.NDArray[np.float32]] = {}
    for name, val in raw.items():
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.float32)
        if arr.size < 2:
            continue
        arr_2d = np.ascontiguousarray(arr.reshape(-1, 2), dtype=np.float32)
        valid = np.isfinite(arr_2d[:, 0]) & np.isfinite(arr_2d[:, 1])
        if not np.any(valid):
            continue
        arr_2d = arr_2d[valid]
        if name == "corners" and len(arr_2d) >= 3:
            arr_2d = _sort_corners_clockwise(arr_2d)
        out[str(name)] = arr_2d
    return out


def _compute_body_and_nose_from_identity_arrays(
    pts_xy: npt.NDArray[np.float32],
    kp_valid: npt.NDArray[np.bool_],
    ident_mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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


def _cache_hit_covers(entry: _CachedVideoSpatial | None, s_req: int, e_req: int) -> bool:
    """Return True if cached slice `entry` covers requested frame span `[s_req, e_req]`."""
    if entry is None:
        return False
    if entry.n_frames_total <= 0 or s_req >= entry.n_frames_total or e_req < 0:
        return True
    s_need = max(0, s_req)
    e_need = min(entry.n_frames_total - 1, e_req)
    return entry.s_offset <= s_need and e_need < (entry.s_offset + entry.body_xy.shape[1])


def _load_video_spatial(
    project: Project,
    video_name: str,
    s_req: int,
    e_req: int,
) -> _CachedVideoSpatial | None:
    """Load `(body_xy, nose_xy)` over `[s_req, e_req]` via unified `PoseLike` API + byte-capped LRU."""
    pose_path = _resolve_pose_path(project, video_name)
    pose_key = (
        _file_stat_key(pose_path) if (pose_path is not None and pose_path.is_file()) else None
    )
    prev_hit: _CachedVideoSpatial | None = None
    if pose_key is not None:
        prev_hit = _spatial_cache_get(pose_key)
        if _cache_hit_covers(prev_hit, s_req, e_req):
            return prev_hit

    try:
        pose = load_pose_cached(project, video_name)
    except Exception:
        return None

    identities = sorted(int(i) for i in getattr(pose, "identities", []) or [])
    n_frames = int(getattr(pose, "num_frames", 0) or 0)
    raw_so = getattr(pose, "static_objects", {}) or {}
    cm_raw = getattr(pose, "cm_per_pixel", None)
    cm_val = float(cm_raw) if cm_raw is not None else None
    static_objs = _normalize_static_objects_dict(raw_so)

    if n_frames <= 0 or not identities:
        empty = np.full((0, 0, 2), np.nan, dtype=np.float32)
        return _CachedVideoSpatial(
            s_offset=0,
            n_frames_total=max(0, n_frames),
            body_xy=empty,
            nose_xy=empty.copy(),
            cm_per_pixel=cm_val,
            static_objects=static_objs,
        )

    if s_req >= n_frames or e_req < 0:
        n_id = max(identities) + 1
        empty = np.full((n_id, 0, 2), np.nan, dtype=np.float32)
        return _CachedVideoSpatial(
            s_offset=0,
            n_frames_total=n_frames,
            body_xy=empty,
            nose_xy=empty.copy(),
            cm_per_pixel=cm_val,
            static_objects=static_objs,
        )

    s0 = max(0, min(int(s_req), n_frames - 1))
    e0 = max(s0, min(int(e_req), n_frames - 1))
    if (
        prev_hit is not None
        and prev_hit.n_frames_total == n_frames
        and prev_hit.body_xy.shape[1] > 0
    ):
        s0 = min(s0, prev_hit.s_offset)
        e0 = max(e0, prev_hit.s_offset + prev_hit.body_xy.shape[1] - 1)
    n_slice = e0 - s0 + 1
    n_id = max(identities) + 1
    body_xy = np.full((n_id, n_slice, 2), np.nan, dtype=np.float32)
    nose_xy = np.full((n_id, n_slice, 2), np.nan, dtype=np.float32)
    id_mask_fn = getattr(pose, "identity_mask", None)

    for ident in identities:
        try:
            pts_i, mask_i = pose.get_identity_poses(ident)
            pts_slice = np.asarray(pts_i[s0 : e0 + 1], dtype=np.float32)
            msk_slice = np.asarray(mask_i[s0 : e0 + 1], dtype=bool)
            if pts_slice.ndim != 3 or pts_slice.shape[0] <= 0:
                continue
            im_slice: npt.NDArray[np.bool_] | None = None
            if callable(id_mask_fn):
                raw_im = id_mask_fn(ident)
                if isinstance(raw_im, np.ndarray) and raw_im.ndim >= 1:
                    im_slice = np.asarray(raw_im[s0 : e0 + 1], dtype=bool)
            b_arr, n_arr = _compute_body_and_nose_from_identity_arrays(
                pts_slice[None, ...],
                msk_slice[None, ...],
                ident_mask=im_slice[None, ...] if im_slice is not None else None,
            )
            t_copy = min(n_slice, b_arr.shape[1])
            body_xy[ident, :t_copy] = b_arr[0, :t_copy]
            nose_xy[ident, :t_copy] = n_arr[0, :t_copy]
        except Exception:
            continue

    loaded = _CachedVideoSpatial(
        s_offset=s0,
        n_frames_total=n_frames,
        body_xy=body_xy,
        nose_xy=nose_xy,
        cm_per_pixel=cm_val,
        static_objects=static_objs,
    )
    if pose_key is None:
        pose_attr = getattr(pose, "pose_file", None)
        if isinstance(pose_attr, Path) and pose_attr.is_file():
            pose_key = _file_stat_key(pose_attr)
    if pose_key is not None:
        _spatial_cache_put(pose_key, loaded)
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
    """Return `(width, height)` in pixels for `video_path` via fast header scan or OpenCV."""
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
) -> tuple[npt.NDArray[np.uint8], float, float] | None:
    """Read and downscale the first frame of `video_path` in grayscale for the Arena backdrop."""
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

    ref_static_objects: dict[str, npt.NDArray[np.float32]] = {}
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
    ref_gray: npt.NDArray[np.uint8] | None = None
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
            b_idxs = np.asarray([b_i for b_i, _ in idx_recs], dtype=np.int32)
            dims = video_dims.get(video_name)
            if dims is None or dims[0] <= 0.0 or dims[1] <= 0.0:
                coords_xy[b_idxs] = np.nan
                nose_xy[b_idxs] = np.nan
                logger.warning(
                    "Skipping %d bout(s) from %s in normalized arena view: unreadable video dimensions",
                    len(b_idxs),
                    video_name,
                )
                continue
            w_v, h_v = dims
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
        pts_list: list[npt.NDArray[np.float32]] = []
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
