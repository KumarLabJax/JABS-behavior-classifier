"""Cached pose trajectory loader (via `Project.load_pose_est`) and publication PNG+SVG exporter."""

from __future__ import annotations

import contextlib
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from jabs.bale.models import BaleReport
from jabs.project import Project

# Keep bounded LRU caches for unified `PoseLike` objects and lightweight annotation proxies.
_POSE_LRU_CACHE: OrderedDict[tuple[str, str, int, int], Any] = OrderedDict()
_POSE_META_CACHE: OrderedDict[tuple[str, str, int, int], Any] = OrderedDict()
_POSE_CACHE_LOCK = threading.Lock()
_POSE_LRU_MAX = 4
_POSE_META_MAX = 128


def clear_pose_cache() -> None:
    """Clear both unified pose LRU cache and lightweight annotation proxy cache."""
    with _POSE_CACHE_LOCK:
        _POSE_LRU_CACHE.clear()
        _POSE_META_CACHE.clear()


@dataclass
class _AnnotationPoseProxy:
    """Lightweight cached substitute for `PoseLike` during annotation save/load."""

    num_frames: int
    _identity_mask: npt.NDArray[np.bool_] | None
    external_identities: list[str] | None = None

    def identity_mask(self, identity: int) -> npt.NDArray[np.bool_]:
        """Return boolean frame validity mask for `identity`."""
        if self._identity_mask is not None and 0 <= int(identity) < self._identity_mask.shape[0]:
            return self._identity_mask[int(identity), :]
        return np.zeros(self.num_frames, dtype=bool)

    def identity_index_to_display(self, identity_index: int) -> str:
        """Map numeric identity index to its external display name when available."""
        if self.external_identities and 0 <= int(identity_index) < len(self.external_identities):
            return self.external_identities[int(identity_index)]
        return str(identity_index)


def _pose_stat_cache_key(project: Project, video: str) -> tuple[str, str, int, int]:
    """Build a file-stat cache key `(resolved_project_dir, video, mtime_ns, size_bytes)`."""
    mtime_ns, size_bytes = 0, 0
    try:
        pose_path = Path(project.video_manager.get_cached_pose_path(video))
        if pose_path.is_file():
            st = pose_path.stat()
            mtime_ns, size_bytes = int(st.st_mtime_ns), int(st.st_size)
    except Exception:
        pass
    proj_dir_raw = getattr(project, "dir", None)
    proj_dir_str = str(Path(proj_dir_raw).resolve()) if isinstance(proj_dir_raw, Path) else ""
    return (proj_dir_str, video, mtime_ns, size_bytes)


def load_pose_cached(project: Project, video: str) -> Any:
    """Return a stat-validated LRU-cached `PoseLike` object for `(project, video)`."""
    cache_key = _pose_stat_cache_key(project, video)
    with _POSE_CACHE_LOCK:
        pose = _POSE_LRU_CACHE.get(cache_key)
        if pose is not None:
            _POSE_LRU_CACHE.move_to_end(cache_key)
            return pose
    pose = project.load_pose_est(project.video_manager.video_path(video))
    close_fn = getattr(pose, "close", None)
    if callable(close_fn):
        with contextlib.suppress(Exception):
            close_fn()
    with _POSE_CACHE_LOCK:
        _POSE_LRU_CACHE[cache_key] = pose
        _POSE_LRU_CACHE.move_to_end(cache_key)
        while len(_POSE_LRU_CACHE) > _POSE_LRU_MAX:
            _POSE_LRU_CACHE.popitem(last=False)
    return pose


def load_annotation_pose_proxy(project: Project, video: str) -> Any:
    """Return lightweight cached `_AnnotationPoseProxy` backed by the project pose API."""
    cache_key = _pose_stat_cache_key(project, video)
    with _POSE_CACHE_LOCK:
        cached = _POSE_META_CACHE.get(cache_key)
        if cached is not None:
            _POSE_META_CACHE.move_to_end(cache_key)
            return cached

    pose = load_pose_cached(project, video)
    n_frames = int(getattr(pose, "num_frames", 0) or 0)
    raw_ext = getattr(pose, "external_identities", None)
    ext_ids = [str(x) for x in raw_ext] if raw_ext is not None else None

    ident_mask: npt.NDArray[np.bool_] | None = None
    id_mask_fn = getattr(pose, "identity_mask", None)
    if callable(id_mask_fn) and n_frames > 0:
        identities = sorted(int(i) for i in (getattr(pose, "identities", None) or []))
        if identities:
            n_id = max(identities) + 1
            ident_mask = np.zeros((n_id, n_frames), dtype=bool)
            for ident in identities:
                arr = np.asarray(id_mask_fn(ident), dtype=bool)
                n_copy = min(n_frames, int(arr.shape[0]))
                if n_copy > 0:
                    ident_mask[ident, :n_copy] = arr[:n_copy]

    proxy = _AnnotationPoseProxy(
        num_frames=n_frames,
        _identity_mask=ident_mask,
        external_identities=ext_ids,
    )
    with _POSE_CACHE_LOCK:
        _POSE_META_CACHE[cache_key] = proxy
        _POSE_META_CACHE.move_to_end(cache_key)
        while len(_POSE_META_CACHE) > _POSE_META_MAX:
            _POSE_META_CACHE.popitem(last=False)
    return proxy


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
    frames: npt.NDArray[np.int32]  # shape: (T,)
    nose_xy: npt.NDArray[np.float64]  # shape: (T, 2)
    neck_xy: npt.NDArray[np.float64]  # shape: (T, 2)
    spine_xy: npt.NDArray[np.float64]  # shape: (T, 2)
    tail_xy: npt.NDArray[np.float64]  # shape: (T, 2)
    centroid_xy: npt.NDArray[np.float64]  # shape: (T, 2)
    partner_centroids_xy: dict[int, npt.NDArray[np.float64]] = field(default_factory=dict)
    lixit_xy: npt.NDArray[np.float64] | None = None


def load_bout_pose_trajectory(
    project: Project,
    video: str,
    identity: int,
    start_frame: int,
    end_frame: int,
) -> BoutPoseTrajectory:
    """Lazily load 2D `(x, y)` trajectories for `identity` (and cage partners) over `[start_frame, end_frame]`."""
    pose = load_pose_cached(project, video)

    pts_all, mask_all = pose.get_identity_poses(int(identity))
    n_total = int(pts_all.shape[0])
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

    partner_centroids: dict[int, npt.NDArray[np.float64]] = {}
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

    lixit_xy: npt.NDArray[np.float64] | None = None
    raw_so = getattr(pose, "static_objects", None) or {}
    if "lixit" in raw_so and raw_so["lixit"] is not None:
        raw_lixit = np.asarray(raw_so["lixit"], dtype=np.float64)
        if raw_lixit.size >= 2:
            lixit_xy = raw_lixit.reshape(-1, 2)

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
    static_objects: dict[str, npt.NDArray[np.float32]] | None,
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
