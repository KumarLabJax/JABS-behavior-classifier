"""Lazy pose trajectory loader (`open_pose_file`) and publication PNG+SVG figure exporter."""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from jabs.bale.models import BaleReport
from jabs.project import Project

# Keep a thread-safe LRU cache of opened PoseEstimation objects so clicking or
# editing multiple bouts in the same video is instantaneous (< 1 ms).
_POSE_LRU_CACHE: dict[tuple[str, str], Any] = {}
_POSE_CACHE_LOCK = threading.Lock()
_POSE_LRU_MAX = 32


def load_pose_cached(project: Project, video: str) -> Any:
    """Return a cached `PoseEstimation` instance for `(project, video)` (`< 1 ms` warm)."""
    cache_key = (str(Path(project.dir).resolve()), video)
    with _POSE_CACHE_LOCK:
        pose = _POSE_LRU_CACHE.get(cache_key)
    if pose is not None:
        return pose
    pose = project.load_pose_est(project.video_manager.video_path(video))
    with _POSE_CACHE_LOCK:
        if len(_POSE_LRU_CACHE) >= _POSE_LRU_MAX:
            oldest_key = next(iter(_POSE_LRU_CACHE))
            _POSE_LRU_CACHE.pop(oldest_key, None)
        _POSE_LRU_CACHE[cache_key] = pose
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


def load_bout_pose_trajectory(
    project: Project,
    video: str,
    identity: int,
    start_frame: int,
    end_frame: int,
) -> BoutPoseTrajectory:
    """Lazily load 2D `(x, y)` trajectories for `identity` (and cage partners) over `[start_frame, end_frame]`.

    Note: `PoseEstimation.get_identity_poses` in JABS (`PoseEstimationV2` / `V3+`)
    already flips the raw HDF5 `[y, x]` coordinates to `[x, y]`.
    """
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

        # Panel A: 2D Representation Scatter
        ax_a = axes[0, 0]
        if projection == "tsne" and emb.coords_tsne_2d is not None:
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

        is_pos = np.array([r.is_positive for r in records], dtype=bool)
        is_pred = np.array([r.source == "predicted" for r in records], dtype=bool)
        is_suspect = diag.confusion_scores >= 0.70

        lbl_pos = (~is_pred) & is_pos
        lbl_neg = (~is_pred) & (~is_pos)
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
        if np.any(is_pred):
            ax_a.scatter(
                coords[is_pred, 0],
                coords[is_pred, 1],
                facecolors="#DBEAFE",
                edgecolors="#1E40AF",
                s=52,
                alpha=0.85,
                marker="^",
                linewidths=1.2,
                label=f"Predicted (n={int(np.sum(is_pred))})",
            )
        if np.any(is_suspect):
            ax_a.scatter(
                coords[is_suspect, 0],
                coords[is_suspect, 1],
                facecolors="none",
                edgecolors="#B91C1C",
                s=115,
                linewidths=1.8,
                label=f"Suspect / High Doubt (n={int(np.sum(is_suspect))})",
            )

        space_title_map = {
            "classifier": "Classifier-Weighted",
            "unsupervised": "Unsupervised",
            "rf_proximity": "RF-Proximity",
        }
        space_disp = space_title_map.get(report.space, str(report.space))
        ax_a.set_title(f"A. 2D Representation Space ({space_disp})", loc="left", color="#0F172A")
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
