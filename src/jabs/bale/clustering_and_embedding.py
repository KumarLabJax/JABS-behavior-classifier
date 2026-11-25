"""PCA 2D, t-SNE, RF-Proximity MDS, and Ward.D2 sub-type clustering for `jabs.bale`.

Supports ethological feature lenses, inlier-seeded Ward.D2 clustering, 1-way ANOVA
eta-squared feature ranking, and cage-skew batch auditing.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, ward
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from .models import (
    BoutFeatureMatrix,
    BoutRecord,
    ClusterAndEmbeddingResult,
    ConfusionAndOutlierResult,
    FeatureLens,
    RepresentationSpace,
    SubtypeScope,
)

logger = logging.getLogger(__name__)

_CAGE_RE = re.compile(r"(cage[_-]?\d+)", re.IGNORECASE)
_META_VALIDITY_TOKENS = (
    "_jabs_frame_valid",
    "point_mask",
    "_jabs_closest_identities",
    "_jabs_closest_fov_identities",
    "closest_identities",
    "closest_fov_identities",
)
_SOCIAL_PREFIXES = (
    "closest_",
    "huddle_",
    "social_",
    "_jabs_closest_",
)
_ARENA_DIR_TOKENS = (
    "centroid_velocity_dir",
    "point_velocity_dirs",
    "velocity direction",
    "velocity_dir",
)
_LOCOMOTION_TOKENS = (
    "centroid_velocity_mag",
    "point_speeds",
    "angular_velocity",
    "angles",
    "speed",
    "turn",
    "angular",
)
_POSTURE_TOKENS = (
    "shape_descriptor",
    "hu_moments",
    "moments",
    "pairwise_distances",
    "angles",
    "posture",
    "elongat",
    "aspect",
    "area",
    "hull",
)


def _extract_cage_id(video_name: str) -> str:
    """Extract normalized cage/cohort identifier from a video name or path."""
    m = _CAGE_RE.search(video_name)
    if m:
        return m.group(1).lower().replace("-", "_")
    stem = Path(video_name).stem
    parts = [p for p in stem.split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return stem or "unknown"


def select_feature_lens_mask(
    stat_names: Sequence[str],
    lens: FeatureLens,
) -> np.ndarray:
    """Return boolean mask over bout statistic columns for an ethological `FeatureLens`."""
    n_cols = len(stat_names)
    if n_cols == 0 or lens in ("all", "classifier_weighted"):
        return np.ones(n_cols, dtype=bool)

    mask = np.zeros(n_cols, dtype=bool)
    for i, raw_name in enumerate(stat_names):
        name = raw_name.lower()
        is_meta_or_id = any(tok in name for tok in _META_VALIDITY_TOKENS)
        if is_meta_or_id:
            continue
        is_social = any(name.startswith(pref) or pref in name for pref in _SOCIAL_PREFIXES)
        if lens == "social":
            if is_social:
                mask[i] = True
            continue

        if is_social:
            continue
        is_arena_dir = any(tok in name for tok in _ARENA_DIR_TOKENS)
        if is_arena_dir:
            continue
        is_distal_tail = "tip_tail" in name

        if lens == "intrinsic":
            if not is_distal_tail:
                mask[i] = True
        elif lens == "locomotion":
            if any(tok in name for tok in _LOCOMOTION_TOKENS):
                mask[i] = True
        elif (
            lens == "posture"
            and not is_distal_tail
            and any(tok in name for tok in _POSTURE_TOKENS)
        ):
            mask[i] = True

    # Safety fallback for minimal synthetic test fixtures that lack standard JABS feature names
    if not np.any(mask):
        return np.ones(n_cols, dtype=bool)
    return mask


def _resolve_fit_indices(
    records: Sequence[BoutRecord],
    subtype_scope: SubtypeScope,
    target_behavior: str | None = None,
) -> np.ndarray:
    """Resolve cohort indices used to fit scaling, PCA, MDS, and Ward.D2 sub-clusters."""
    n_bouts = len(records)
    if subtype_scope == "pos_only":
        pos_idx = [
            i
            for i, r in enumerate(records)
            if r.source == "labeled"
            and r.is_positive
            and (
                target_behavior is None
                or r.class_name == target_behavior
                or r.behavior == target_behavior
            )
        ]
        if len(pos_idx) >= 4:
            return np.asarray(pos_idx, dtype=np.int64)
        all_pos_idx = [i for i, r in enumerate(records) if r.source == "labeled" and r.is_positive]
        if len(all_pos_idx) >= 4:
            return np.asarray(all_pos_idx, dtype=np.int64)
        labeled_idx = [
            i
            for i, r in enumerate(records)
            if r.source == "labeled"
            and (
                target_behavior is None
                or r.class_name in (target_behavior, "None")
                or r.behavior == target_behavior
            )
        ]
        if len(labeled_idx) >= 4:
            return np.asarray(labeled_idx, dtype=np.int64)
        all_labeled_idx = [i for i, r in enumerate(records) if r.source == "labeled"]
        if len(all_labeled_idx) >= 4:
            return np.asarray(all_labeled_idx, dtype=np.int64)
    elif target_behavior is not None:
        scoped_idx = [
            i
            for i, r in enumerate(records)
            if r.source == "labeled"
            and (r.class_name in (target_behavior, "None") or r.behavior == target_behavior)
        ]
        if len(scoped_idx) >= 4:
            return np.asarray(scoped_idx, dtype=np.int64)
    return np.arange(n_bouts, dtype=np.int64)


def _resolve_seed_indices(
    fit_idx: np.ndarray,
    outlier_votes: np.ndarray,
) -> np.ndarray:
    """Select consensus-inlier indices within `fit_idx` to seed Ward linkage without outliers."""
    n_fit = len(fit_idx)
    if n_fit < 8:
        return fit_idx
    min_keep = max(6, int(0.55 * n_fit))
    votes_fit = outlier_votes[fit_idx]
    inliers_strict = fit_idx[votes_fit < 2]
    if len(inliers_strict) >= min_keep:
        return inliers_strict
    inliers_relaxed = fit_idx[votes_fit < 3]
    if len(inliers_relaxed) >= min_keep:
        return inliers_relaxed
    return fit_idx


def _compute_mds_coords(
    dist_rf_all: np.ndarray,
    fit_idx: np.ndarray,
    n_bouts: int,
) -> np.ndarray | None:
    """Compute classical PCoA on `fit_idx` with Gower Nyström projection for remaining bouts."""
    if len(fit_idx) < 4:
        return None
    d2_fit = dist_rf_all[np.ix_(fit_idx, fit_idx)] ** 2
    row_mean_fit = np.mean(d2_fit, axis=1, keepdims=True)
    col_mean_fit = np.mean(d2_fit, axis=0, keepdims=True)
    total_mean_fit = np.mean(d2_fit)
    b_fit = -0.5 * (d2_fit - row_mean_fit - col_mean_fit + total_mean_fit)
    eigvals, eigvecs = np.linalg.eigh(b_fit)
    top2_idx = np.argsort(eigvals)[::-1][:2]
    pos_eigs = np.maximum(eigvals[top2_idx], 1e-8)
    top2_vecs = eigvecs[:, top2_idx]
    coords_fit = (top2_vecs * np.sqrt(pos_eigs)[None, :]).astype(np.float32)

    if len(fit_idx) == n_bouts:
        return coords_fit

    d2_all_fit = dist_rf_all[:, fit_idx] ** 2
    row_mean_all = np.mean(d2_all_fit, axis=1, keepdims=True)
    b_all = -0.5 * (d2_all_fit - row_mean_all - col_mean_fit + total_mean_fit)
    coords_all = (b_all @ (top2_vecs / np.sqrt(pos_eigs)[None, :])).astype(np.float32)
    coords_all[fit_idx] = coords_fit
    return coords_all


def _select_ward_clusters_euclidean(
    X_pca: np.ndarray,
    seed_idx: np.ndarray,
    k_range: tuple[int, int],
    forced_k: int | None,
) -> tuple[np.ndarray, int, float]:
    """Run single-tree Ward.D2 on `X_pca[seed_idx]` and assign all bouts to nearest centroid."""
    n_bouts = X_pca.shape[0]
    n_seed = len(seed_idx)
    if n_seed < 4:
        return np.ones(n_bouts, dtype=np.int32), 1, 0.0

    X_seed = X_pca[seed_idx]
    Z_tree = ward(pdist(X_seed, metric="euclidean"))
    k_min = max(2, k_range[0])
    k_max = min(k_range[1], n_seed - 1)

    sil_sample = min(600, n_seed) if n_seed > 600 else None
    if forced_k is not None and n_seed >= 3:
        best_k = min(max(2, int(forced_k)), n_seed - 1)
        seed_labels = fcluster(Z_tree, t=best_k, criterion="maxclust").astype(np.int32)
        best_sil = (
            float(silhouette_score(X_seed, seed_labels, sample_size=sil_sample, random_state=42))
            if len(np.unique(seed_labels)) >= 2
            else 0.0
        )
    elif k_max >= k_min:
        ks = list(range(k_min, k_max + 1))
        sils: list[float] = []
        wcss_list: list[float] = []
        labels_by_k: list[np.ndarray] = []

        for k_cand in ks:
            lbls = fcluster(Z_tree, t=k_cand, criterion="maxclust").astype(np.int32)
            labels_by_k.append(lbls)
            if len(np.unique(lbls)) >= 2:
                sils.append(
                    float(silhouette_score(X_seed, lbls, sample_size=sil_sample, random_state=42))
                )
            else:
                sils.append(-1.0)
            wcss = 0.0
            for c_id in np.unique(lbls):
                c_pts = X_seed[lbls == c_id]
                if c_pts.shape[0] > 1:
                    wcss += float(np.sum((c_pts - np.mean(c_pts, axis=0, keepdims=True)) ** 2))
            wcss_list.append(wcss)

        sil_arr = np.asarray(sils, dtype=np.float32)
        sil_norm = (sil_arr - np.min(sil_arr)) / max(1e-6, float(np.ptp(sil_arr)))

        if len(ks) >= 3:
            x_norm = np.linspace(0.0, 1.0, len(ks), dtype=np.float32)
            w_arr = np.asarray(wcss_list, dtype=np.float32)
            y_norm = (w_arr - np.min(w_arr)) / max(1e-6, float(np.ptp(w_arr)))
            line_y = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
            elbow_raw = np.maximum(0.0, line_y - y_norm)
            elbow_norm = elbow_raw / max(1e-6, float(np.max(elbow_raw)))
        else:
            elbow_norm = np.zeros_like(sil_norm)

        best_idx = int(np.argmax(0.65 * sil_norm + 0.35 * elbow_norm))
        best_k = ks[best_idx]
        seed_labels = labels_by_k[best_idx]
        best_sil = sils[best_idx]
    else:
        return np.ones(n_bouts, dtype=np.int32), 1, 0.0

    # Project all bouts (including consensus outliers / non-fit bouts) onto nearest seed centroid
    all_labels = np.empty(n_bouts, dtype=np.int32)
    all_labels[seed_idx] = seed_labels
    if n_seed < n_bouts:
        raw_cids = sorted(np.unique(seed_labels))
        centroids = np.vstack(
            [np.mean(X_seed[seed_labels == cid], axis=0) for cid in raw_cids]
        ).astype(np.float32)
        non_seed_mask = np.ones(n_bouts, dtype=bool)
        non_seed_mask[seed_idx] = False
        X_rest = X_pca[non_seed_mask]
        d2_rest = np.sum((X_rest[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        all_labels[non_seed_mask] = np.asarray(raw_cids, dtype=np.int32)[
            np.argmin(d2_rest, axis=1)
        ]

    return all_labels, best_k, best_sil


def _select_ward_clusters_rf_proximity(
    dist_rf_all: np.ndarray,
    seed_idx: np.ndarray,
    k_range: tuple[int, int],
    forced_k: int | None,
) -> tuple[np.ndarray, int, float]:
    """Run Ward.D2 directly on `sqrt(1 - rf_proximity)` and assign bouts via exact centroid D2."""
    n_bouts = dist_rf_all.shape[0]
    n_seed = len(seed_idx)
    if n_seed < 4:
        return np.ones(n_bouts, dtype=np.int32), 1, 0.0

    D_seed = dist_rf_all[np.ix_(seed_idx, seed_idx)].astype(np.float64, copy=True)
    D_seed = 0.5 * (D_seed + D_seed.T)
    np.fill_diagonal(D_seed, 0.0)
    Z_tree = ward(squareform(D_seed, checks=False))
    D2_seed = D_seed**2

    k_min = max(2, k_range[0])
    k_max = min(k_range[1], n_seed - 1)

    if forced_k is not None and n_seed >= 3:
        best_k = min(max(2, int(forced_k)), n_seed - 1)
        seed_labels = fcluster(Z_tree, t=best_k, criterion="maxclust").astype(np.int32)
        best_sil = (
            float(silhouette_score(D_seed, seed_labels, metric="precomputed"))
            if len(np.unique(seed_labels)) >= 2
            else 0.0
        )
    elif k_max >= k_min:
        ks = list(range(k_min, k_max + 1))
        sils: list[float] = []
        wcss_list: list[float] = []
        labels_by_k: list[np.ndarray] = []

        for k_cand in ks:
            lbls = fcluster(Z_tree, t=k_cand, criterion="maxclust").astype(np.int32)
            labels_by_k.append(lbls)
            if len(np.unique(lbls)) >= 2:
                sils.append(float(silhouette_score(D_seed, lbls, metric="precomputed")))
            else:
                sils.append(-1.0)
            wcss = 0.0
            for c_id in np.unique(lbls):
                m_c = lbls == c_id
                n_c = int(np.sum(m_c))
                if n_c > 1:
                    wcss += 0.5 * float(np.sum(D2_seed[np.ix_(m_c, m_c)])) / float(n_c)
            wcss_list.append(wcss)

        sil_arr = np.asarray(sils, dtype=np.float32)
        sil_norm = (sil_arr - np.min(sil_arr)) / max(1e-6, float(np.ptp(sil_arr)))

        if len(ks) >= 3:
            x_norm = np.linspace(0.0, 1.0, len(ks), dtype=np.float32)
            w_arr = np.asarray(wcss_list, dtype=np.float32)
            y_norm = (w_arr - np.min(w_arr)) / max(1e-6, float(np.ptp(w_arr)))
            line_y = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
            elbow_raw = np.maximum(0.0, line_y - y_norm)
            elbow_norm = elbow_raw / max(1e-6, float(np.max(elbow_raw)))
        else:
            elbow_norm = np.zeros_like(sil_norm)

        best_idx = int(np.argmax(0.65 * sil_norm + 0.35 * elbow_norm))
        best_k = ks[best_idx]
        seed_labels = labels_by_k[best_idx]
        best_sil = sils[best_idx]
    else:
        return np.ones(n_bouts, dtype=np.int32), 1, 0.0

    all_labels = np.empty(n_bouts, dtype=np.int32)
    all_labels[seed_idx] = seed_labels
    if n_seed < n_bouts:
        D2_all = dist_rf_all**2
        non_seed_mask = np.ones(n_bouts, dtype=bool)
        non_seed_mask[seed_idx] = False
        raw_cids = sorted(np.unique(seed_labels))
        d2_to_c = np.zeros((int(np.sum(non_seed_mask)), len(raw_cids)), dtype=np.float64)
        for c_pos, cid in enumerate(raw_cids):
            members = seed_idx[seed_labels == cid]
            within_half = 0.5 * float(np.mean(D2_all[np.ix_(members, members)]))
            d2_to_c[:, c_pos] = np.mean(D2_all[non_seed_mask][:, members], axis=1) - within_half
        all_labels[non_seed_mask] = np.asarray(raw_cids, dtype=np.int32)[
            np.argmin(d2_to_c, axis=1)
        ]

    return all_labels, best_k, best_sil


def _compute_cage_skew(
    records: Sequence[BoutRecord],
    fit_idx: np.ndarray,
    labels: np.ndarray,
    best_k: int,
) -> tuple[float | None, float | None, dict[int, tuple[str, float]]]:
    """Audit whether Ward sub-clusters primarily reflect animal/cage recording batch."""
    fit_cages = [_extract_cage_id(records[int(i)].video) for i in fit_idx]
    fit_labels = labels[fit_idx]

    dominant_by_cluster: dict[int, tuple[str, float]] = {}
    for c_id in range(1, best_k + 1):
        c_cages = [
            cage for lbl, cage in zip(fit_labels, fit_cages, strict=False) if int(lbl) == c_id
        ]
        if c_cages:
            top_cage, top_cnt = Counter(c_cages).most_common(1)[0]
            dominant_by_cluster[c_id] = (top_cage, round(float(top_cnt / len(c_cages)), 4))

    unique_cages = sorted(set(fit_cages))
    if best_k < 2 or len(unique_cages) < 2 or len(fit_idx) < 4:
        return None, None, dominant_by_cluster

    cage_to_col = {cage: col for col, cage in enumerate(unique_cages)}
    obs = np.zeros((best_k, len(unique_cages)), dtype=np.int64)
    for lbl, cage in zip(fit_labels, fit_cages, strict=False):
        c_row = int(lbl) - 1
        if 0 <= c_row < best_k:
            obs[c_row, cage_to_col[cage]] += 1

    obs = obs[np.sum(obs, axis=1) > 0][:, np.sum(obs, axis=0) > 0]
    min_dim = min(obs.shape) - 1
    n_total = int(np.sum(obs))
    if min_dim < 1 or n_total < 4:
        return None, None, dominant_by_cluster

    chi2_val, p_val, _, _ = chi2_contingency(obs, correction=False)
    cramers_v = float(np.sqrt(max(0.0, float(chi2_val)) / float(n_total * min_dim)))
    return float(p_val), min(1.0, cramers_v), dominant_by_cluster


def compute_clustering_and_embedding(
    feature_matrix: BoutFeatureMatrix,
    diagnostics: ConfusionAndOutlierResult,
    *,
    space: RepresentationSpace = "classifier",
    subtype_scope: SubtypeScope = "all_labeled",
    feature_lens: FeatureLens = "all",
    target_behavior: str | None = None,
    compute_tsne: bool = True,
    compute_mds: bool = False,
    k_range: tuple[int, int] | None = None,
    forced_k: int | None = None,
) -> ClusterAndEmbeddingResult:
    """Compute 2D projections (PCA, t-SNE, MDS) and single-tree Ward.D2 sub-clusters.

    Supports cohort-restricted scaling/PCA/clustering (`subtype_scope="pos_only"`),
    ethological `feature_lens` selection, inlier-seeded Ward.D2 clustering, direct
    `rf_proximity` Ward linkage on `sqrt(1 - proximity)`, unbiased vectorized 1-way
    ANOVA eta-squared feature ranking, and chi-squared/Cramér's V cage-skew audits.
    """
    records = feature_matrix.records
    X_bout = feature_matrix.X_bout
    n_bouts, n_stats = X_bout.shape

    eff_k_range = (
        k_range if k_range is not None else ((2, 6) if subtype_scope == "pos_only" else (2, 7))
    )

    if n_bouts == 0:
        return ClusterAndEmbeddingResult(
            coords_pca_2d=np.zeros((0, 2), dtype=np.float32),
            pca_explained_variance_ratio=(0.0, 0.0),
            coords_tsne_2d=None,
            coords_mds_2d=None,
            cluster_labels=np.zeros(0, dtype=np.int32),
            optimal_k=1,
            silhouette=0.0,
            cluster_top_features={},
            top_z_diffs_per_bout=[],
            cluster_feature_eta2={},
            cluster_counts={},
            cage_skew_p=None,
            cage_skew_cramers_v=None,
            dominant_cage_by_cluster={},
            subtype_scope=subtype_scope,
            feature_lens=feature_lens,
            forced_k=forced_k,
            target_behavior=target_behavior,
        )

    X_clean = np.nan_to_num(X_bout, nan=0.0, posinf=0.0, neginf=0.0)
    fit_idx = _resolve_fit_indices(records, subtype_scope, target_behavior=target_behavior)
    lens_mask = select_feature_lens_mask(feature_matrix.stat_names, feature_lens)

    std_mask = lens_mask & (np.std(X_clean[fit_idx], axis=0) > 1e-8)
    if not np.any(std_mask):
        std_mask = lens_mask & (np.std(X_clean, axis=0) > 1e-8)
    if not np.any(std_mask):
        std_mask = lens_mask
    if not np.any(std_mask):
        std_mask = np.ones(n_stats, dtype=bool)

    active_stat_indices = np.flatnonzero(std_mask)
    X_sub_clean = X_clean[:, std_mask]
    X_fit_clean = X_sub_clean[fit_idx]
    n_fit = len(fit_idx)

    s_sorted = np.sort(X_fit_clean, axis=0)
    med = (
        s_sorted[n_fit // 2]
        if (n_fit % 2 == 1)
        else 0.5 * (s_sorted[n_fit // 2 - 1] + s_sorted[n_fit // 2])
    )
    q25 = s_sorted[int(0.25 * max(0, n_fit - 1))]
    q75 = s_sorted[int(0.75 * max(0, n_fit - 1))]
    std_col = np.std(X_fit_clean, axis=0)
    # Robust SD scaling (`IQR / 1.349` or `0.5 * std`) + tight ±3.5 SD Winsorization
    scale = np.maximum((q75 - q25) / 1.3489795, 0.5 * std_col)
    robust_sd = np.where(scale > 1e-5, scale, 1.0).astype(np.float32)
    X_scaled = np.nan_to_num(
        (X_sub_clean - med[None, :]) / robust_sd[None, :],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    X_scaled = np.clip(X_scaled, -3.5, 3.5).astype(np.float32)

    use_rf_weights = (
        space in ("classifier", "rf_proximity") or feature_lens == "classifier_weighted"
    )
    if use_rf_weights:
        w_active = diagnostics.stat_feature_weights[std_mask]
        X_repr = X_scaled * np.sqrt(np.maximum(w_active, 1e-6))[None, :]
    else:
        X_repr = X_scaled

    # 1. PCA 2D and 95%-variance PCA subspace fitted on `fit_idx`, projected to all bouts
    max_comp = min(n_fit, X_repr.shape[1], 50)
    if max_comp >= 2:
        pca = PCA(n_components=max_comp, random_state=42)
        pca.fit(X_repr[fit_idx])
        X_pca_all = pca.transform(X_repr).astype(np.float32)
        coords_pca_2d = X_pca_all[:, :2]
        var_ratio = (
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        )
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_95 = max(2, min(int(np.searchsorted(cum_var, 0.95) + 1), max_comp))
        X_pca = X_pca_all[:, :n_95]
    else:
        coords_pca_2d = np.zeros((n_bouts, 2), dtype=np.float32)
        if X_repr.shape[1] >= 1:
            coords_pca_2d[:, 0] = X_repr[:, 0]
        var_ratio = (1.0, 0.0)
        X_pca = coords_pca_2d

    # 2. Optional RF Leaf-Proximity MDS 2D (classical PCoA on `fit_idx` + Nyström projection)
    coords_mds_2d: np.ndarray | None = None
    dist_rf_all: np.ndarray | None = None
    if diagnostics.rf_proximity_matrix is not None:
        dist_rf_all = np.sqrt(np.clip(1.0 - diagnostics.rf_proximity_matrix, 0.0, 1.0))
        if space == "rf_proximity" or compute_mds:
            coords_mds_2d = _compute_mds_coords(dist_rf_all, fit_idx, n_bouts)
            if space == "rf_proximity" and coords_mds_2d is not None:
                coords_pca_2d = coords_mds_2d

    # 3. Optional 2D t-SNE (fit on `fit_idx`, 3-NN interpolation for remaining bouts when scoped)
    coords_tsne_2d: np.ndarray | None = None
    if compute_tsne and n_fit >= 6:
        perp = float(min(30.0, max(5.0, (n_fit - 1) / 3.0)))
        X_tsne_in = X_pca[fit_idx, : min(12, X_pca.shape[1])]
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            method="exact" if n_fit < 180 else "barnes_hut",
            angle=0.6,
            max_iter=250,
            random_state=42,
            n_jobs=1,
        )
        coords_tsne_fit = tsne.fit_transform(X_tsne_in).astype(np.float32)
        if n_fit == n_bouts:
            coords_tsne_2d = coords_tsne_fit
        else:
            coords_tsne_2d = np.zeros((n_bouts, 2), dtype=np.float32)
            coords_tsne_2d[fit_idx] = coords_tsne_fit
            non_fit_mask = np.ones(n_bouts, dtype=bool)
            non_fit_mask[fit_idx] = False
            X_feat_rest = X_pca[non_fit_mask, : min(12, X_pca.shape[1])]
            d2_nn = np.sum((X_feat_rest[:, None, :] - X_tsne_in[None, :, :]) ** 2, axis=2)
            k_nn = min(3, n_fit)
            nn_idx = np.argpartition(d2_nn, kth=k_nn - 1, axis=1)[:, :k_nn]
            coords_tsne_2d[non_fit_mask] = np.mean(coords_tsne_fit[nn_idx], axis=1)

    # 4. Inlier-Seeded Single-Tree Ward.D2 Clustering (Euclidean PCA or true RF-Proximity)
    seed_idx = _resolve_seed_indices(fit_idx, diagnostics.outlier_votes)
    if space == "rf_proximity" and dist_rf_all is not None:
        raw_labels, best_k, best_sil = _select_ward_clusters_rf_proximity(
            dist_rf_all, seed_idx, eff_k_range, forced_k
        )
    else:
        raw_labels, best_k, best_sil = _select_ward_clusters_euclidean(
            X_pca, seed_idx, eff_k_range, forced_k
        )

    # Relabel clusters in descending `fit_idx` size order (`C1` = largest subtype)
    fit_raw = raw_labels[fit_idx]
    fit_unique = [int(c) for c in np.unique(fit_raw)]
    ordered_raw = sorted(
        fit_unique,
        key=lambda cid: (-int(np.sum(fit_raw == cid)), cid),
    )
    old_to_new = {old_cid: new_cid for new_cid, old_cid in enumerate(ordered_raw, start=1)}
    best_labels = np.array([old_to_new.get(int(c), 1) for c in raw_labels], dtype=np.int32)
    best_k = len(ordered_raw)
    unique_clusters = list(range(1, best_k + 1))
    cluster_counts = {cid: int(np.sum(best_labels[fit_idx] == cid)) for cid in unique_clusters}

    # 5. Vectorized 1-way ANOVA eta-squared on `fit_idx` + Deduplicated Heatmap Matrix
    active_stat_names = [feature_matrix.stat_names[i] for i in active_stat_indices]
    active_base_ids = [int(feature_matrix.stat_to_base_idx[i]) for i in active_stat_indices]
    w_active_all = diagnostics.stat_feature_weights[std_mask]
    X_fit_scaled = X_scaled[fit_idx]
    fit_labels = best_labels[fit_idx]

    cluster_means = np.zeros((best_k, X_scaled.shape[1]), dtype=np.float32)
    for c_pos, c_id in enumerate(unique_clusters):
        c_fit_mask = fit_labels == c_id
        c_rows = X_fit_scaled[c_fit_mask] if np.any(c_fit_mask) else X_scaled[best_labels == c_id]
        cluster_means[c_pos] = np.median(c_rows, axis=0).astype(np.float32)

    if best_k >= 2 and n_fit >= 3:
        grand_mean = np.mean(X_fit_scaled, axis=0, dtype=np.float64)
        ss_total = np.sum((X_fit_scaled - grand_mean[None, :]) ** 2, axis=0, dtype=np.float64)
        ss_between = np.zeros(X_scaled.shape[1], dtype=np.float64)
        for c_id in unique_clusters:
            m_c = fit_labels == c_id
            n_c = int(np.sum(m_c))
            if n_c > 0:
                c_mean = np.mean(X_fit_scaled[m_c], axis=0, dtype=np.float64)
                ss_between += float(n_c) * (c_mean - grand_mean) ** 2
        eta2_vec = np.where(
            ss_total > 1e-12,
            np.clip(ss_between / np.maximum(ss_total, 1e-12), 0.0, 1.0),
            0.0,
        )
    else:
        eta2_vec = np.zeros(X_scaled.shape[1], dtype=np.float64)

    bias_classifier_weights = space == "classifier" or feature_lens == "classifier_weighted"
    weight_factor = (
        np.sqrt(np.maximum(w_active_all, 1e-4))
        if bias_classifier_weights
        else np.ones_like(w_active_all)
    )
    has_eta2 = best_k >= 2 and bool(np.any(eta2_vec > 1e-9))

    cluster_top_features: dict[int, list[tuple[str, float]]] = {}
    for c_pos, c_id in enumerate(unique_clusters):
        c_med = cluster_means[c_pos]
        base_score = np.abs(c_med) * (np.maximum(eta2_vec, 0.02) if has_eta2 else 1.0)
        scored = base_score * weight_factor
        sorted_c_idx = np.argsort(scored)[::-1]
        c_top_idx: list[int] = []
        seen_c_bases: set[int] = set()
        for j in sorted_c_idx:
            b_id = active_base_ids[int(j)]
            if b_id >= 0 and b_id in seen_c_bases:
                continue
            if b_id >= 0:
                seen_c_bases.add(b_id)
            c_top_idx.append(int(j))
            if len(c_top_idx) >= min(8, X_scaled.shape[1]):
                break
        if len(c_top_idx) < min(8, X_scaled.shape[1]):
            for j in sorted_c_idx:
                if int(j) not in c_top_idx:
                    c_top_idx.append(int(j))
                if len(c_top_idx) >= min(8, X_scaled.shape[1]):
                    break
        cluster_top_features[int(c_id)] = [
            (active_stat_names[j], round(float(c_med[j]), 3)) for j in c_top_idx
        ]

    col_spread = (
        np.ptp(cluster_means, axis=0) if best_k >= 2 else np.max(np.abs(cluster_means), axis=0)
    )
    col_rank_score = (eta2_vec if has_eta2 else col_spread) * weight_factor
    sorted_cols = np.argsort(col_rank_score)[::-1]

    dedup_cols: list[int] = []
    seen_bases: set[int] = set()
    for j in sorted_cols:
        b_id = active_base_ids[int(j)]
        if b_id >= 0 and b_id in seen_bases:
            continue
        if b_id >= 0:
            seen_bases.add(b_id)
        dedup_cols.append(int(j))
        if len(dedup_cols) >= min(40, X_scaled.shape[1]):
            break

    top_20_feat_idx_list = dedup_cols[: min(20, X_scaled.shape[1])]
    if len(top_20_feat_idx_list) < min(20, X_scaled.shape[1]):
        for j in sorted_cols:
            if int(j) not in top_20_feat_idx_list:
                top_20_feat_idx_list.append(int(j))
            if len(top_20_feat_idx_list) >= min(20, X_scaled.shape[1]):
                break

    top_20_feat_idx = np.asarray(top_20_feat_idx_list, dtype=np.int32)
    cluster_z_matrix = cluster_means[:, top_20_feat_idx]
    cluster_z_feature_names = [active_stat_names[j] for j in top_20_feat_idx]

    # Store unbiased eta2 values ordered descending by pure eta2 (covering top_20 + dedup_cols)
    eta2_pool = list(dict.fromkeys([*top_20_feat_idx_list, *dedup_cols]))
    eta2_pool.sort(key=lambda j: -float(eta2_vec[j]))
    cluster_feature_eta2 = {active_stat_names[j]: round(float(eta2_vec[j]), 4) for j in eta2_pool}

    # 6. Per-Bout Top Feature Z-Score Deviations (relative to the bout's own class median)
    top_z_diffs_per_bout: list[list[tuple[str, float]]] = []
    class_medians: dict[str, np.ndarray] = {}
    for cls_name in sorted({r.class_name for r in records}):
        cls_rows = [
            i for i, r in enumerate(records) if r.class_name == cls_name and r.source == "labeled"
        ]
        if not cls_rows:
            cls_rows = [i for i, r in enumerate(records) if r.class_name == cls_name]
        class_medians[cls_name] = np.median(X_scaled[cls_rows], axis=0)

    for idx, rec in enumerate(records):
        diff = X_scaled[idx] - class_medians[rec.class_name]
        top3 = np.argsort(np.abs(diff))[::-1][:4]
        top_z_diffs_per_bout.append(
            [(active_stat_names[j], round(float(diff[j]), 2)) for j in top3]
        )

    # 7. Cage-Skew / Batch Effect Audit (< 1 ms)
    cage_skew_p, cage_skew_cramers_v, dominant_cage_by_cluster = _compute_cage_skew(
        records, fit_idx, best_labels, best_k
    )

    return ClusterAndEmbeddingResult(
        coords_pca_2d=coords_pca_2d,
        pca_explained_variance_ratio=var_ratio,
        coords_tsne_2d=coords_tsne_2d,
        coords_mds_2d=coords_mds_2d,
        cluster_labels=best_labels,
        optimal_k=best_k,
        silhouette=best_sil,
        cluster_top_features=cluster_top_features,
        top_z_diffs_per_bout=top_z_diffs_per_bout,
        cluster_z_matrix=cluster_z_matrix,
        cluster_z_feature_names=cluster_z_feature_names,
        cluster_feature_eta2=cluster_feature_eta2,
        cluster_counts=cluster_counts,
        cage_skew_p=cage_skew_p,
        cage_skew_cramers_v=cage_skew_cramers_v,
        dominant_cage_by_cluster=dominant_cage_by_cluster,
        subtype_scope=subtype_scope,
        feature_lens=feature_lens,
        forced_k=forced_k,
        target_behavior=target_behavior,
    )
