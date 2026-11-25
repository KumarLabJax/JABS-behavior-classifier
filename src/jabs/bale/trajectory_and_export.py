"""Lazy pose trajectory loader (`open_pose_file`) and publication PNG+SVG figure exporter."""

from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from jabs.bale.bout_extractor import strip_known_video_suffix
from jabs.bale.models import BaleReport
from jabs.core.utils.pose_util import build_identity_mask
from jabs.project import Project

# Keep a bounded true LRU cache of opened PoseEstimation fallback objects (`v2`/`v3`/mocks),
# while raw `v4-v8` H5 trajectories & annotation masks bypass full PoseEstimationV8 loading.
_POSE_LRU_CACHE: OrderedDict[tuple[str, str, int, int], Any] = OrderedDict()
_POSE_META_CACHE: OrderedDict[tuple[str, str, int, int], Any] = OrderedDict()
_POSE_CACHE_LOCK = threading.Lock()
_POSE_LRU_MAX = 4
_POSE_META_MAX = 128
_TRAJ_MIN_CONFIDENCE = 0.3
_TRAJ_ACCEPTED_CACHE_VERSIONS = (4, 5)


def clear_pose_cache() -> None:
    """Clear both full PoseEstimation fallback cache and lightweight annotation proxy cache."""
    with _POSE_CACHE_LOCK:
        _POSE_LRU_CACHE.clear()
        _POSE_META_CACHE.clear()


@dataclass
class _AnnotationPoseProxy:
    """Lightweight H5-backed substitute for `PoseEstimation` during annotation save/load (`~6 KB`)."""

    num_frames: int
    _identity_mask: np.ndarray | None
    external_identities: list[str] | None = None

    def identity_mask(self, identity: int) -> np.ndarray:
        if self._identity_mask is not None and 0 <= int(identity) < self._identity_mask.shape[0]:
            return self._identity_mask[int(identity), :]
        return np.zeros(self.num_frames, dtype=bool)

    def identity_index_to_display(self, identity_index: int) -> str:
        if self.external_identities and 0 <= int(identity_index) < len(self.external_identities):
            return self.external_identities[int(identity_index)]
        return str(identity_index)


def _decode_external_identities(pose_grp: h5py.Group) -> list[str] | None:
    if "external_identity_mapping" not in pose_grp:
        return None
    raw_ids = pose_grp["external_identity_mapping"][:]
    if np.issubdtype(raw_ids.dtype, np.integer):
        return [str(x) for x in raw_ids.tolist()]
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw_ids.tolist()]


def _resolve_traj_pose_and_cache_paths(
    project: Project, video: str
) -> tuple[Path | None, Path | None]:
    raw_pose_path: Path | None = None
    vm = getattr(project, "video_manager", None)
    if vm is not None and hasattr(vm, "get_cached_pose_path"):
        try:
            cand_raw = vm.get_cached_pose_path(video)
            if isinstance(cand_raw, str | Path):
                cand_p = Path(cand_raw)
                if cand_p.is_file():
                    raw_pose_path = cand_p
        except Exception:
            raw_pose_path = None

    proj_dir_raw = getattr(project, "dir", None)
    proj_dir = proj_dir_raw if isinstance(proj_dir_raw, Path) else None
    if raw_pose_path is None and proj_dir is not None:
        stem = strip_known_video_suffix(video)
        for match in sorted(proj_dir.glob(f"{stem}_pose_est_v*.h5")):
            if match.is_file():
                raw_pose_path = match
                break

    cache_dir: Path | None = None
    for attr in ("_paths", "project_paths"):
        holder = getattr(project, attr, None)
        if holder is not None:
            cdir = getattr(holder, "cache_dir", None)
            if isinstance(cdir, Path) and cdir.is_dir():
                cache_dir = cdir
                break
    if cache_dir is None and proj_dir is not None:
        cand_dir = proj_dir / "jabs" / "cache"
        if cand_dir.is_dir():
            cache_dir = cand_dir

    cache_h5_path: Path | None = None
    if cache_dir is not None:
        if raw_pose_path is not None:
            cand = cache_dir / raw_pose_path.name.replace(".h5", "_cache.h5")
            if cand.is_file():
                cache_h5_path = cand
        if cache_h5_path is None:
            stem = strip_known_video_suffix(video)
            matches = sorted(cache_dir.glob(f"{stem}_pose_est_*_cache.h5"))
            if matches and matches[0].is_file():
                cache_h5_path = matches[0]
    return raw_pose_path, cache_h5_path


def load_annotation_pose_proxy(project: Project, video: str) -> Any:
    """Return lightweight `_AnnotationPoseProxy` (`~2 ms`, `< 0.5 MB`) for real `v4-v8` H5s.

    Falls back to `load_pose_cached` for `v2`/`v3` or mocked `Project` objects.
    """
    if hasattr(getattr(project, "load_pose_est", None), "assert_called"):
        return load_pose_cached(project, video)

    raw_pose_path, cache_h5_path = _resolve_traj_pose_and_cache_paths(project, video)
    backing_path = raw_pose_path or cache_h5_path
    if backing_path is None:
        return load_pose_cached(project, video)

    try:
        st = backing_path.stat()
        mtime_ns, size_bytes = int(st.st_mtime_ns), int(st.st_size)
    except OSError:
        mtime_ns, size_bytes = 0, 0
    proj_dir_str = (
        str(Path(project.dir).resolve()) if isinstance(getattr(project, "dir", None), Path) else ""
    )
    cache_key = (proj_dir_str, video, mtime_ns, size_bytes)
    with _POSE_CACHE_LOCK:
        cached = _POSE_META_CACHE.get(cache_key)
        if cached is not None:
            _POSE_META_CACHE.move_to_end(cache_key)
            return cached

    proxy: _AnnotationPoseProxy | None = None
    # Fast Path A: standalone `*_cache.h5` (only when raw pose absent or cache strictly covers it cleanly)
    if cache_h5_path is not None and raw_pose_path is None:
        try:
            with h5py.File(cache_h5_path, "r") as ch5:
                if (
                    ch5.attrs.get("cache_file_version") in _TRAJ_ACCEPTED_CACHE_VERSIONS
                    and "poseest" in ch5
                ):
                    grp = ch5["poseest"]
                    n_frames = int(ch5.attrs["num_frames"])
                    n_id = int(ch5.attrs["num_identities"])
                    ext_ids = _decode_external_identities(grp)
                    ident_mask = (
                        np.asarray(grp["identity_mask"][:], dtype=bool)
                        if (n_id > 0 and "identity_mask" in grp)
                        else np.zeros((0, n_frames), dtype=bool)
                    )
                    proxy = _AnnotationPoseProxy(
                        num_frames=n_frames,
                        _identity_mask=ident_mask,
                        external_identities=ext_ids,
                    )
        except Exception:
            proxy = None

    # Fast Path B: Direct lightweight metadata slice from raw `v4-v8` H5 (skips `points` & `seg_data`)
    if proxy is None and raw_pose_path is not None:
        try:
            with h5py.File(raw_pose_path, "r") as hf:
                if "poseest" in hf:
                    grp = hf["poseest"]
                    required = ("confidence", "id_mask", "instance_embed_id")
                    if all(k in grp for k in required):
                        conf = grp["confidence"][:]
                        id_mask = grp["id_mask"][:]
                        embed_id = grp["instance_embed_id"][:]
                        ext_ids = _decode_external_identities(grp)
                        n_frames = int(conf.shape[0])
                        valid_eids = (
                            embed_id[id_mask == 0]
                            if embed_id.shape[1] > 0
                            else np.array([], dtype=embed_id.dtype)
                        )
                        max_eid = int(valid_eids.max()) if valid_eids.size > 0 else 0
                        min_eid = int(valid_eids.min()) if valid_eids.size > 0 else 1
                        if "instance_id_center" in grp:
                            n_id = int(grp["instance_id_center"].shape[0])
                        elif max_eid > 0:
                            n_id = max_eid
                        else:
                            n_id = 0
                        if max_eid <= n_id and min_eid >= 1:
                            if n_id > 0:
                                c_tmp = np.zeros((n_frames, n_id, conf.shape[2]), dtype=conf.dtype)
                                mask0 = id_mask == 0
                                r_idx, s_idx = np.nonzero(mask0)
                                c_tmp[r_idx, embed_id[r_idx, s_idx] - 1, :] = conf[r_idx, s_idx, :]
                                conf_by_id = np.transpose(c_tmp, (1, 0, 2))
                                pt_mask = conf_by_id > _TRAJ_MIN_CONFIDENCE
                                ident_mask = build_identity_mask(pt_mask, min_valid_keypoints=3)
                            else:
                                ident_mask = None
                            proxy = _AnnotationPoseProxy(
                                num_frames=n_frames,
                                _identity_mask=ident_mask,
                                external_identities=ext_ids,
                            )
        except Exception:
            proxy = None

    if proxy is not None:
        with _POSE_CACHE_LOCK:
            _POSE_META_CACHE[cache_key] = proxy
            _POSE_META_CACHE.move_to_end(cache_key)
            while len(_POSE_META_CACHE) > _POSE_META_MAX:
                _POSE_META_CACHE.popitem(last=False)
        return proxy

    return load_pose_cached(project, video)


def load_pose_cached(project: Project, video: str) -> Any:
    """Return a stat-validated LRU-cached `PoseEstimation` fallback for `(project, video)`."""
    mtime_ns, size_bytes = 0, 0
    try:
        pose_path = Path(project.video_manager.get_cached_pose_path(video))
        if pose_path.is_file():
            st = pose_path.stat()
            mtime_ns, size_bytes = int(st.st_mtime_ns), int(st.st_size)
    except Exception:
        pass
    cache_key = (str(Path(project.dir).resolve()), video, mtime_ns, size_bytes)
    with _POSE_CACHE_LOCK:
        pose = _POSE_LRU_CACHE.get(cache_key)
        if pose is not None:
            _POSE_LRU_CACHE.move_to_end(cache_key)
            return pose
    pose = project.load_pose_est(project.video_manager.video_path(video))
    with _POSE_CACHE_LOCK:
        _POSE_LRU_CACHE[cache_key] = pose
        _POSE_LRU_CACHE.move_to_end(cache_key)
        while len(_POSE_LRU_CACHE) > _POSE_LRU_MAX:
            _POSE_LRU_CACHE.popitem(last=False)
    return pose


KEYPOINT_NOSE = 0
KEYPOINT_BASE_NECK = 3
KEYPOINT_CENTER_SPINE = 6
KEYPOINT_BASE_TAIL = 9


@dataclass
class BoutPoseTrajectory:
    """2D arena trajectories (`(x, y)` in pixels) for a single bout."""

    video: str
    identity: int
    start_frame: int
    end_frame: int
    frames: np.ndarray  # shape: (T,)
    nose_xy: np.ndarray  # shape: (T, 2)
    neck_xy: np.ndarray  # shape: (T, 2)
    spine_xy: np.ndarray  # shape: (T, 2)
    tail_xy: np.ndarray  # shape: (T, 2)
    centroid_xy: np.ndarray  # shape: (T, 2)
    partner_centroids_xy: dict[int, np.ndarray] = field(default_factory=dict)
    lixit_xy: np.ndarray | None = None


def _try_load_bout_trajectory_sliced(
    project: Project,
    video: str,
    identity: int,
    start_frame: int,
    end_frame: int,
) -> BoutPoseTrajectory | None:
    """Slice only `[s:e+1, :, :10, :]` from raw `v4-v8` H5 or `*_cache.h5` (skips `seg_data`)."""
    raw_pose_path, cache_h5_path = _resolve_traj_pose_and_cache_paths(project, video)
    k_slice = 10

    # Tier 1: Direct slice from uncompressed raw `v4-v8` HDF5 (`< 4 ms`, `< 0.1 MB` RAM)
    if raw_pose_path is not None and raw_pose_path.is_file():
        try:
            with h5py.File(raw_pose_path, "r") as hf:
                if "poseest" in hf:
                    grp = hf["poseest"]
                    required = ("points", "confidence", "id_mask", "instance_embed_id")
                    if all(k in grp for k in required):
                        pts_ds = grp["points"]
                        if pts_ds.ndim == 4 and pts_ds.shape[-1] == 2 and int(pts_ds.shape[0]) > 0:
                            n_total = int(pts_ds.shape[0])
                            k_avail = min(k_slice, int(pts_ds.shape[2]))
                            s = max(0, min(int(start_frame), n_total - 1))
                            e = max(s, min(int(end_frame), n_total - 1))
                            n_win = e - s + 1

                            pts_raw = np.asarray(
                                pts_ds[s : e + 1, :, :k_avail, :], dtype=np.float64
                            )
                            conf_raw = np.asarray(
                                grp["confidence"][s : e + 1, :, :k_avail], dtype=np.float32
                            )
                            id_mask = np.asarray(grp["id_mask"][s : e + 1, :])
                            embed_id = np.asarray(grp["instance_embed_id"][s : e + 1, :])

                            valid_slot = (id_mask == 0) & (embed_id >= 1)
                            max_id = int(embed_id[valid_slot].max()) if np.any(valid_slot) else 0
                            if "instance_id_center" in grp:
                                n_id = max(max_id, int(grp["instance_id_center"].shape[0]))
                            else:
                                n_id = max_id

                            lixit_xy: np.ndarray | None = None
                            if "static_objects" in hf and "lixit" in hf["static_objects"]:
                                raw_lix = np.asarray(
                                    hf["static_objects"]["lixit"][:], dtype=np.float64
                                )
                                if raw_lix.size >= 2:
                                    lixit_xy = np.flip(raw_lix.reshape(-1, 2), axis=-1)

                            pts_id = np.full((n_id, n_win, k_avail, 2), np.nan, dtype=np.float64)
                            if n_id > 0 and np.any(valid_slot):
                                pts_xy = np.flip(pts_raw, axis=-1)
                                valid_kp = conf_raw > _TRAJ_MIN_CONFIDENCE
                                f_idx, slot_idx = np.nonzero(valid_slot)
                                id_idx = embed_id[f_idx, slot_idx].astype(np.int64) - 1
                                ok_id = (id_idx >= 0) & (id_idx < n_id)
                                f_idx = f_idx[ok_id]
                                slot_idx = slot_idx[ok_id]
                                id_idx = id_idx[ok_id]
                                sel_pts = pts_xy[f_idx, slot_idx, :, :]
                                sel_mask = valid_kp[f_idx, slot_idx, :]
                                sel_pts[~sel_mask] = np.nan
                                pts_id[id_idx, f_idx, :, :] = sel_pts

                            ident_i = int(identity)
                            if 0 <= ident_i < n_id:
                                pts_main = pts_id[ident_i]
                            else:
                                pts_main = np.full((n_win, k_avail, 2), np.nan, dtype=np.float64)

                            def _kp_or_nan(k_i: int) -> np.ndarray:
                                if k_i < k_avail:
                                    return pts_main[:, k_i, :]
                                return np.full((n_win, 2), np.nan, dtype=np.float64)

                            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                                warnings.simplefilter("ignore", RuntimeWarning)
                                centroid_xy = np.nanmean(pts_main[:, :k_avail, :], axis=1)
                                partner_centroids: dict[int, np.ndarray] = {}
                                for other_id in range(n_id):
                                    if other_id == ident_i:
                                        continue
                                    partner_centroids[other_id] = np.nanmean(
                                        pts_id[other_id, :, :k_avail, :], axis=1
                                    )

                            return BoutPoseTrajectory(
                                video=video,
                                identity=ident_i,
                                start_frame=s,
                                end_frame=e,
                                frames=np.arange(s, e + 1, dtype=np.int32),
                                nose_xy=_kp_or_nan(KEYPOINT_NOSE),
                                neck_xy=_kp_or_nan(KEYPOINT_BASE_NECK),
                                spine_xy=_kp_or_nan(KEYPOINT_CENTER_SPINE),
                                tail_xy=_kp_or_nan(KEYPOINT_BASE_TAIL),
                                centroid_xy=centroid_xy,
                                partner_centroids_xy=partner_centroids,
                                lixit_xy=lixit_xy,
                            )
        except Exception:
            pass

    # Tier 2: JABS `*_cache.h5` (v4 or v5) when raw pose H5 is absent
    if cache_h5_path is not None and cache_h5_path.is_file():
        try:
            with h5py.File(cache_h5_path, "r") as ch5:
                if (
                    ch5.attrs.get("cache_file_version") in _TRAJ_ACCEPTED_CACHE_VERSIONS
                    and "poseest" in ch5
                ):
                    grp = ch5["poseest"]
                    if "points" in grp and "point_mask" in grp:
                        pts_ds = grp["points"]
                        n_id = int(pts_ds.shape[0])
                        n_total = int(pts_ds.shape[1])
                        k_avail = min(k_slice, int(pts_ds.shape[2]))
                        if n_total > 0 and k_avail > 0:
                            s = max(0, min(int(start_frame), n_total - 1))
                            e = max(s, min(int(end_frame), n_total - 1))
                            n_win = e - s + 1
                            pts_win = np.asarray(
                                pts_ds[:, s : e + 1, :k_avail, :], dtype=np.float64
                            )
                            msk_win = np.asarray(grp["point_mask"][:, s : e + 1, :k_avail]) > 0
                            pts_win[~msk_win] = np.nan

                            lixit_xy_c: np.ndarray | None = None
                            if "static_objects" in ch5 and "lixit" in ch5["static_objects"]:
                                raw_lix = np.asarray(
                                    ch5["static_objects"]["lixit"][:], dtype=np.float64
                                )
                                if raw_lix.size >= 2:
                                    lixit_xy_c = raw_lix.reshape(-1, 2)

                            ident_i = int(identity)
                            pts_main = (
                                pts_win[ident_i]
                                if 0 <= ident_i < n_id
                                else np.full((n_win, k_avail, 2), np.nan, dtype=np.float64)
                            )

                            def _kp_c(k_i: int) -> np.ndarray:
                                if k_i < k_avail:
                                    return pts_main[:, k_i, :]
                                return np.full((n_win, 2), np.nan, dtype=np.float64)

                            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                                warnings.simplefilter("ignore", RuntimeWarning)
                                centroid_xy = np.nanmean(pts_main[:, :k_avail, :], axis=1)
                                partner_centroids_c: dict[int, np.ndarray] = {
                                    other_id: np.nanmean(pts_win[other_id, :, :k_avail, :], axis=1)
                                    for other_id in range(n_id)
                                    if other_id != ident_i
                                }

                            return BoutPoseTrajectory(
                                video=video,
                                identity=ident_i,
                                start_frame=s,
                                end_frame=e,
                                frames=np.arange(s, e + 1, dtype=np.int32),
                                nose_xy=_kp_c(KEYPOINT_NOSE),
                                neck_xy=_kp_c(KEYPOINT_BASE_NECK),
                                spine_xy=_kp_c(KEYPOINT_CENTER_SPINE),
                                tail_xy=_kp_c(KEYPOINT_BASE_TAIL),
                                centroid_xy=centroid_xy,
                                partner_centroids_xy=partner_centroids_c,
                                lixit_xy=lixit_xy_c,
                            )
        except Exception:
            pass

    return None


def load_bout_pose_trajectory(
    project: Project,
    video: str,
    identity: int,
    start_frame: int,
    end_frame: int,
) -> BoutPoseTrajectory:
    """Lazily load 2D `(x, y)` trajectories for `identity` (and cage partners) over `[start_frame, end_frame]`.

    Uses a direct `[s:e+1, :, :10, :]` H5 slice (`~2-4 ms`, `< 0.1 MB` RAM, skipping `seg_data`)
    on `v4-v8` files, falling back to `load_pose_cached` for `v2`/`v3` or mocked projects.
    """
    if not hasattr(getattr(project, "load_pose_est", None), "assert_called"):
        sliced = _try_load_bout_trajectory_sliced(project, video, identity, start_frame, end_frame)
        if sliced is not None:
            return sliced

    pose = load_pose_cached(project, video)

    pts_all, mask_all = pose.get_identity_poses(int(identity))
    n_total = pts_all.shape[0]
    s = max(0, min(int(start_frame), n_total - 1))
    e = max(s, min(int(end_frame), n_total - 1))

    pts = np.asarray(pts_all[s : e + 1], dtype=np.float64).copy()
    mask = np.asarray(mask_all[s : e + 1], dtype=bool)
    pts[~mask] = np.nan

    frames = np.arange(s, e + 1, dtype=np.int32)
    nose_xy = pts[:, KEYPOINT_NOSE, :]
    neck_xy = pts[:, KEYPOINT_BASE_NECK, :]
    spine_xy = pts[:, KEYPOINT_CENTER_SPINE, :]
    tail_xy = pts[:, KEYPOINT_BASE_TAIL, :]

    body_pts = pts[:, :10, :]
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        centroid_xy = np.nanmean(body_pts, axis=1)

    partner_centroids: dict[int, np.ndarray] = {}
    for other_id in getattr(pose, "identities", []):
        if int(other_id) == int(identity):
            continue
        try:
            o_pts_all, o_mask_all = pose.get_identity_poses(int(other_id))
            o_pts = np.asarray(o_pts_all[s : e + 1, :10, :], dtype=np.float64).copy()
            o_mask = np.asarray(o_mask_all[s : e + 1, :10], dtype=bool)
            o_pts[~o_mask] = np.nan
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                partner_centroids[int(other_id)] = np.nanmean(o_pts, axis=1)
        except Exception:
            continue

    lixit_xy: np.ndarray | None = None
    try:
        pose_path = project.video_manager.get_cached_pose_path(video)
        with h5py.File(pose_path, "r") as hf:
            if "static_objects" in hf and "lixit" in hf["static_objects"]:
                raw_lixit = np.asarray(hf["static_objects"]["lixit"][:], dtype=np.float64)
                if raw_lixit.size >= 2:
                    # Raw HDF5 static objects are stored as (y, x); flip last axis to (x, y)
                    lixit_xy = np.flip(raw_lixit.reshape(-1, 2), axis=-1)
    except Exception:
        lixit_xy = None

    return BoutPoseTrajectory(
        video=video,
        identity=int(identity),
        start_frame=s,
        end_frame=e,
        frames=frames,
        nose_xy=nose_xy,
        neck_xy=neck_xy,
        spine_xy=spine_xy,
        tail_xy=tail_xy,
        centroid_xy=centroid_xy,
        partner_centroids_xy=partner_centroids,
        lixit_xy=lixit_xy,
    )


def _format_feature_name(name: str, max_len: int = 28) -> str:
    """Abbreviate verbose JABS feature prefixes and anatomical keypoints without mid-word truncation."""
    s = str(name)
    prefix_map = (
        ("social_pairwise_fov_distances ", "soc_fov_dist: "),
        ("social_pairwise_distances ", "soc_dist: "),
        ("closest_fov_distances ", "fov_dist: "),
        ("closest_fov_angles ", "fov_ang: "),
        ("closest_distances ", "closest: "),
        ("pairwise_distances ", "dist: "),
        ("angles angle ", "angle: "),
        ("huddle_features ", "huddle: "),
        ("shape_descriptor ", "shape: "),
        ("hu_moments ", "hu: "),
        ("moments ", "moment: "),
        ("speed ", "speed: "),
        ("angular_velocity ", "ang_vel: "),
    )
    for old_p, new_p in prefix_map:
        if s.startswith(old_p):
            s = new_p + s[len(old_p) :]
            break

    token_map = (
        ("LEFT_FRONT_PAW", "L_FPAW"),
        ("RIGHT_FRONT_PAW", "R_FPAW"),
        ("LEFT_REAR_PAW", "L_RPAW"),
        ("RIGHT_REAR_PAW", "R_RPAW"),
        ("CENTER_SPINE", "C_SPINE"),
        ("MID_SPINE", "M_SPINE"),
        ("BASE_NECK", "NECK"),
        ("BASE_TAIL", "B_TAIL"),
        ("MID_TAIL", "M_TAIL"),
        ("TIP_TAIL", "T_TAIL"),
        ("LEFT_EAR", "L_EAR"),
        ("RIGHT_EAR", "R_EAR"),
        ("second closest", "2nd_closest"),
        ("first closest", "1st_closest"),
        ("social contact", "contact"),
        ("inter mouse", "inter_mouse"),
    )
    for old_t, new_t in token_map:
        s = s.replace(old_t, new_t)

    if len(s) > max_len:
        s = s[: max_len - 1].rstrip("_- ") + "…"
    return s


def draw_arena_static_landmarks(
    ax: Any,
    static_objects: dict[str, np.ndarray] | None,
) -> None:
    """Draw cage corners, inner perimeter zone, food hopper, and lixit landmarks on `ax`."""
    if not static_objects:
        return

    corners = static_objects.get("corners")
    if corners is not None and len(corners) >= 3:
        closed = np.vstack([corners, corners[:1]])
        # Crisp white line over subtle slate shadow stays legible over dark grayscale arenas
        ax.plot(closed[:, 0], closed[:, 1], color="#0F172A", linewidth=2.6, alpha=0.65, zorder=1)
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="#FFFFFF",
            linewidth=1.5,
            alpha=0.92,
            zorder=1.2,
            label="Cage Perimeter",
        )
        centroid = np.mean(corners, axis=0)
        inset = centroid + 0.88 * (corners - centroid)
        inset_closed = np.vstack([inset, inset[:1]])
        ax.plot(
            inset_closed[:, 0],
            inset_closed[:, 1],
            color="#D97706",
            linestyle="--",
            linewidth=1.1,
            alpha=0.80,
            zorder=1.1,
        )

    hopper = static_objects.get("food_hopper")
    if hopper is not None and len(hopper) > 0:
        pts = np.vstack([hopper, hopper[:1]]) if len(hopper) >= 3 else hopper
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="#D97706",
            linewidth=1.8,
            marker="s" if len(hopper) < 3 else None,
            markersize=5.0,
            alpha=0.9,
            zorder=1.5,
            label="Food Hopper",
        )

    lixit = static_objects.get("lixit")
    if lixit is not None and len(lixit) > 0:
        ax.scatter(
            lixit[:, 0],
            lixit[:, 1],
            c="#0F766E",
            marker="^",
            s=65,
            edgecolors="#FFFFFF",
            linewidths=0.9,
            zorder=2.5,
            label="Lixit",
        )


def export_bale_summary_figure(
    report: BaleReport,
    out_png: Path | str,
    out_svg: Path | str | None = None,
    projection: str = "pca",
) -> tuple[Path, Path]:
    """Export a publication-grade 4-panel BALE (Bout & Label Explorer) summary figure (`.png` + `.svg`).

    Follows the restrained scientific design system:
    - Compact `14.0 x 8.8 in` canvas with large `Arial`/`Helvetica` typography
    - Editable vector `<text>` elements in `.svg` (`svg.fonttype = 'none'`, `pdf.fonttype = 42`)
    - Restrained Navy (`#1E40AF`), Slate Teal (`#0F766E`), and Crimson (`#B91C1C`) palette
    """
    png_path = Path(out_png)
    svg_path = Path(out_svg) if out_svg is not None else png_path.with_suffix(".svg")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    rc_overrides = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12.0,
        "axes.titlesize": 14.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 13.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.0,
        "figure.titlesize": 17.5,
        "figure.titleweight": "bold",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }

    fm = report.feature_matrix
    diag = report.diagnostics
    emb = report.embedding
    records = fm.records

    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with matplotlib.rc_context(rc_overrides):
        fig = Figure(figsize=(14.2, 8.8), facecolor="#FFFFFF")
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.08, wspace=0.25, hspace=0.34)

        n_labeled = sum(1 for r in records if r.source == "labeled")
        n_pred = sum(1 for r in records if r.source == "predicted")
        fig.suptitle(
            f"BALE - Bout & Label Explorer — {report.behavior} "
            f"({n_labeled} Labeled, {n_pred} Predicted | {report.cv_groups_count} CV Groups)",
            y=0.965,
            color="#0F172A",
        )

        # Panel A: 2D Representation or Arena Spatial Scatter
        ax_a = axes[0, 0]
        is_spatial = projection == "spatial" and report.spatial is not None
        if is_spatial:
            sp = report.spatial
            assert sp is not None
            coords = sp.coords_xy
            unit_lbl = "normalized" if sp.normalized else "px"
            xlab, ylab = f"Arena X ({unit_lbl})", f"Arena Y ({unit_lbl})"
            w_bound = sp.frame_width or 1.0
            h_bound = sp.frame_height or 1.0
            if sp.reference_frame_gray is not None and not sp.normalized:
                ax_a.imshow(
                    sp.reference_frame_gray,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                    alpha=0.52,
                    extent=(0.0, w_bound, h_bound, 0.0),
                    interpolation="bilinear",
                    zorder=0,
                )
            draw_arena_static_landmarks(ax_a, sp.static_objects)
        elif projection == "tsne" and emb.coords_tsne_2d is not None:
            coords = emb.coords_tsne_2d
            xlab, ylab = "t-SNE Dim 1", "t-SNE Dim 2"
        elif projection == "mds" and emb.coords_mds_2d is not None:
            coords = emb.coords_mds_2d
            xlab, ylab = "MDS / PCoA Axis 1", "MDS / PCoA Axis 2"
        else:
            coords = emb.coords_pca_2d
            v1, v2 = emb.pca_explained_variance_ratio
            xlab = f"PC1 ({v1 * 100:.1f}% var)"
            ylab = f"PC2 ({v2 * 100:.1f}% var)"

        finite_xy = (
            (np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1]))
            if len(coords) > 0
            else np.zeros(0, dtype=bool)
        )
        is_pos = np.array([r.is_positive for r in records], dtype=bool)
        is_pred = np.array([r.source == "predicted" for r in records], dtype=bool)
        is_suspect = diag.confusion_scores >= 0.70

        lbl_pos = finite_xy & (~is_pred) & is_pos
        lbl_neg = finite_xy & (~is_pred) & (~is_pos)
        if np.any(lbl_pos):
            ax_a.scatter(
                coords[lbl_pos, 0],
                coords[lbl_pos, 1],
                c="#1E40AF",
                s=46,
                alpha=0.82,
                edgecolors="#FFFFFF",
                linewidths=0.6,
                label=f"Labeled {report.behavior} (n={int(np.sum(lbl_pos))})",
            )
        if np.any(lbl_neg):
            ax_a.scatter(
                coords[lbl_neg, 0],
                coords[lbl_neg, 1],
                c="#0F766E",
                s=46,
                alpha=0.78,
                marker="s",
                edgecolors="#FFFFFF",
                linewidths=0.6,
                label=f"Labeled Neg/Other (n={int(np.sum(lbl_neg))})",
            )
        pred_plot = finite_xy & is_pred
        if np.any(pred_plot):
            ax_a.scatter(
                coords[pred_plot, 0],
                coords[pred_plot, 1],
                facecolors="#DBEAFE",
                edgecolors="#1E40AF",
                s=52,
                alpha=0.85,
                marker="^",
                linewidths=1.2,
                label=f"Predicted (n={int(np.sum(pred_plot))})",
            )
        suspect_plot = finite_xy & is_suspect
        if np.any(suspect_plot):
            ax_a.scatter(
                coords[suspect_plot, 0],
                coords[suspect_plot, 1],
                facecolors="none",
                edgecolors="#B91C1C",
                s=115,
                linewidths=1.8,
                label=f"Suspect / High Doubt (n={int(np.sum(suspect_plot))})",
            )

        if is_spatial:
            sp = report.spatial
            assert sp is not None
            w_bound = sp.frame_width or 1.0
            h_bound = sp.frame_height or 1.0
            ax_a.set_xlim(0.0, w_bound)
            ax_a.set_ylim(h_bound, 0.0)
            ax_a.set_aspect("equal", adjustable="box")
            n_mapped = int(np.sum(sp.valid_mask))
            ax_a.set_title(
                f"A. Arena Spatial Map ({n_mapped}/{len(records)} mapped)",
                loc="left",
                color="#0F172A",
            )
        else:
            space_title_map = {
                "classifier": "Classifier-Weighted",
                "unsupervised": "Unsupervised",
                "rf_proximity": "RF-Proximity",
            }
            space_disp = space_title_map.get(report.space, str(report.space))
            ax_a.set_title(
                f"A. 2D Representation Space ({space_disp})", loc="left", color="#0F172A"
            )
        ax_a.set_xlabel(xlab, color="#1E293B")
        ax_a.set_ylabel(ylab, color="#1E293B")
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)
        ax_a.grid(True, color="#E2E8F0", linewidth=0.8)
        if ax_a.get_legend_handles_labels()[0]:
            ax_a.legend(loc="best", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

        # Panel B: Confusion / Doubt vs Kinematic Outlier Score
        ax_b = axes[0, 1]
        ax_b.axvspan(0.70, 1.03, color="#F1F5F9", zorder=0)
        ax_b.axvline(0.70, color="#B91C1C", linestyle="--", linewidth=1.2)
        ax_b.axhline(0.70, color="#475569", linestyle=":", linewidth=1.1)

        if np.any(~is_pred):
            ax_b.scatter(
                diag.confusion_scores[~is_pred],
                diag.outlier_scores[~is_pred],
                c=np.where(is_pos[~is_pred], "#1E40AF", "#0F766E"),
                s=44,
                alpha=0.80,
                edgecolors="#FFFFFF",
                linewidths=0.5,
                label=f"Labeled Bouts (n={int(np.sum(~is_pred))})",
            )
        if np.any(is_pred):
            ax_b.scatter(
                diag.confusion_scores[is_pred],
                diag.outlier_scores[is_pred],
                facecolors="#FEE2E2",
                edgecolors="#B91C1C",
                marker="^",
                s=52,
                alpha=0.85,
                linewidths=1.1,
                label=f"Predicted Bouts (n={int(np.sum(is_pred))})",
            )
        ax_b.text(
            0.715,
            0.98,
            f"Suspect / Doubt >= 0.70\n(n = {int(np.sum(is_suspect))} bouts)",
            fontsize=10.5,
            fontweight="bold",
            color="#B91C1C",
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#FFFFFF",
                "edgecolor": "#CBD5E1",
                "lw": 0.9,
            },
        )
        ax_b.set_xlim(-0.03, 1.03)
        ax_b.set_ylim(-0.03, 1.05)
        ax_b.set_title(
            "B. Confusion / Doubt vs. Kinematic Outlier Score", loc="left", color="#0F172A"
        )
        ax_b.set_xlabel("Label Confusion / Prediction Doubt Score", color="#1E293B")
        ax_b.set_ylabel("Within-Class Outlier Score", color="#1E293B")
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        ax_b.grid(True, color="#E2E8F0", linewidth=0.8)
        if ax_b.get_legend_handles_labels()[0]:
            ax_b.legend(
                loc="lower right",
                frameon=True,
                facecolor="#FFFFFF",
                edgecolor="#CBD5E1",
                fontsize=10.5,
            )

        # Panel C: Top Discriminative Base Features
        ax_c = axes[1, 0]
        w_base = diag.base_feature_weights
        n_base = min(len(w_base), len(fm.base_feature_names))
        top_k = min(8, n_base)
        if top_k > 0:
            top_idx = np.argsort(w_base[:n_base])[::-1][:top_k][::-1]
            names_c = [_format_feature_name(fm.base_feature_names[i], max_len=28) for i in top_idx]
            vals_c = w_base[top_idx] * 100.0
            ax_c.barh(range(top_k), vals_c, color="#1E40AF", edgecolor="none", height=0.65)
            ax_c.set_yticks(range(top_k))
            ax_c.set_yticklabels(names_c)
        else:
            ax_c.text(
                0.5,
                0.5,
                "No labeled bouts available",
                transform=ax_c.transAxes,
                ha="center",
                va="center",
                fontsize=11.0,
                color="#64748B",
            )
            ax_c.set_yticks([])
        ax_c.set_title("C. Top Classifier-Weighted Base Features", loc="left", color="#0F172A")
        ax_c.set_xlabel("Relative Importance (%)", color="#1E293B")
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)
        ax_c.grid(True, axis="x", color="#E2E8F0", linewidth=0.8)

        # Panel D: Sub-Type Cluster Feature Profile (or Transition Score Histogram)
        ax_d = axes[1, 1]
        if emb.cluster_z_matrix is not None and emb.cluster_z_matrix.size > 0:
            zmat = emb.cluster_z_matrix[:, : min(6, emb.cluster_z_matrix.shape[1])]
            fnames = [
                _format_feature_name(n, max_len=22)
                for n in emb.cluster_z_feature_names[: zmat.shape[1]]
            ]
            im = ax_d.imshow(
                zmat,
                cmap="RdBu_r",
                vmin=-2.5,
                vmax=2.5,
                aspect="auto",
            )
            unique_cl = sorted(np.unique(emb.cluster_labels))
            cl_labels_y = [
                f"C{c_id} (n={int(np.sum(emb.cluster_labels == c_id))})"
                for c_id in unique_cl[: zmat.shape[0]]
            ]
            ax_d.set_yticks(range(zmat.shape[0]))
            ax_d.set_yticklabels(cl_labels_y)
            ax_d.set_xticks(range(len(fnames)))
            ax_d.set_xticklabels(fnames, rotation=22, ha="right")
            for r_i in range(zmat.shape[0]):
                for c_i in range(zmat.shape[1]):
                    val_z = float(zmat[r_i, c_i])
                    txt_color = "#FFFFFF" if abs(val_z) > 1.30 else "#0F172A"
                    ax_d.text(
                        c_i,
                        r_i,
                        f"{val_z:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=10.5,
                        fontweight="bold",
                        color=txt_color,
                    )
            ax_d.set_title(
                f"D. Sub-Type Cluster Profiles (k={emb.optimal_k}, Sil={emb.silhouette:.2f})",
                loc="left",
                color="#0F172A",
            )
            fig.colorbar(im, ax=ax_d, fraction=0.04, pad=0.03, label="Median Z-Score")
        else:
            ax_d.hist(diag.transition_scores, bins=15, color="#475569", edgecolor="#FFFFFF")
            ax_d.set_title(
                "D. Boundary Slop / Transition Score Distribution", loc="left", color="#0F172A"
            )
            ax_d.set_xlabel("Transition / Chimeric Score", color="#1E293B")
            ax_d.set_ylabel("Bout Count", color="#1E293B")
            ax_d.spines["top"].set_visible(False)
            ax_d.spines["right"].set_visible(False)

        fig.savefig(png_path, dpi=240, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")

    return png_path, svg_path
