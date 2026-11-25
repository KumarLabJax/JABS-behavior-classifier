"""Out-of-fold Confident Learning confusion, transition, and consensus outlier detection.

Provides within-class whitened-PCA 5-metric consensus outlier scoring and RF proximity
diagnostics for `jabs.bale`.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from .models import BoutFeatureMatrix, ConfusionAndOutlierResult

logger = logging.getLogger(__name__)


def _fast_robust_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute column-wise median and IQR/SD floor via C-level `np.sort(X, axis=0)` (`< 0.5 ms`)."""
    n = X.shape[0]
    if n == 0:
        d = X.shape[1] if X.ndim == 2 else 1
        return np.zeros(d, dtype=np.float32), np.ones(d, dtype=np.float32)
    s = np.sort(X, axis=0)
    med = s[n // 2] if (n % 2 == 1) else 0.5 * (s[n // 2 - 1] + s[n // 2])
    q25 = s[int(0.25 * (n - 1))]
    q75 = s[int(0.75 * (n - 1))]
    std_col = np.std(X, axis=0)
    scale = np.maximum(q75 - q25, 0.5 * std_col)
    iqr = np.where(scale > 1e-5, scale, 1.0).astype(np.float32)
    return med.astype(np.float32), iqr


def _fast_robust_transform(X: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """Scale `X` using precomputed median and IQR, clip extreme spikes, and sanitize non-finite entries."""
    scaled = (X - med[None, :]) / iqr[None, :]
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(scaled, -10.0, 10.0).astype(np.float32, copy=False)


def _rank_percentile(arr: np.ndarray) -> np.ndarray:
    """Convert a 1D array of distances into `[0, 1]` rank percentiles."""
    n = arr.size
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32) / float(n - 1)
    return ranks


def compute_within_class_consensus_outliers(
    X_bout: np.ndarray,
    class_names: list[str],
    sources: list[str],
    *,
    vote_percentile: float = 95.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute within-class 5-metric consensus outliers (Clipper parity) + LOF + IsoForest + OOD.

    Uses the whitened-PCA Mahalanobis identity (`ddof=1` via `pca.explained_variance_` with
    eigenvalue regularization `max(1e-8, 1e-6 * lambda_1)`) separately within each human-labeled
    class so multimodal class mixtures never distort covariance geometry.

    Returns:
        `(outlier_scores, outlier_votes_5, lof_scores, isoforest_scores, ood_scores)`
    """
    n_total = X_bout.shape[0]
    outlier_scores = np.zeros(n_total, dtype=np.float32)
    outlier_votes = np.zeros(n_total, dtype=np.int32)
    lof_scores = np.zeros(n_total, dtype=np.float32)
    iso_scores = np.zeros(n_total, dtype=np.float32)
    ood_scores = np.zeros(n_total, dtype=np.float32)

    if n_total == 0:
        return outlier_scores, outlier_votes, lof_scores, iso_scores, ood_scores

    cls_arr = np.asarray(class_names, dtype=object)
    src_arr = np.asarray(sources, dtype=object)
    unique_classes = sorted(set(class_names))

    for cls_val in unique_classes:
        cls_mask = cls_arr == cls_val
        lab_mask = cls_mask & (src_arr == "labeled")
        pred_mask = cls_mask & (src_arr == "predicted")

        lab_idx = np.flatnonzero(lab_mask)
        pred_idx = np.flatnonzero(pred_mask)

        # Fallback if there are only predicted bouts for this class
        ref_idx = lab_idx if lab_idx.size >= 4 else np.flatnonzero(cls_mask)
        n_ref = ref_idx.size
        if n_ref < 4:
            continue

        X_ref_raw = np.nan_to_num(X_bout[ref_idx], nan=0.0, posinf=0.0, neginf=0.0)
        std_mask = np.std(X_ref_raw, axis=0) > 1e-8
        if not np.any(std_mask):
            continue

        med_ref, iqr_ref = _fast_robust_fit(X_ref_raw[:, std_mask])
        X_ref_scaled = _fast_robust_transform(X_ref_raw[:, std_mask], med_ref, iqr_ref)

        max_comp = min(n_ref - 2, X_ref_scaled.shape[1], 50)
        if max_comp < 1:
            continue

        pca_full = PCA(n_components=max_comp, svd_solver="full", random_state=42)
        X_ref_pca_full = pca_full.fit_transform(X_ref_scaled)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        d_95 = int(np.searchsorted(cum_var, 0.95) + 1)
        n_comp = max(1, min(d_95, max_comp))

        X_ref_pca = X_ref_pca_full[:, :n_comp]
        eigvals = pca_full.explained_variance_[:n_comp]  # ddof=1 sample variance
        lam_floor = max(1e-8, float(eigvals[0]) * 1e-6)
        eigvals_reg = np.maximum(eigvals, lam_floor)
        inv_sqrt_eig = (1.0 / np.sqrt(eigvals_reg)).astype(np.float32)

        # Whitened PCA space: Euclidean distance in X_ref_white == exact Mahalanobis distance!
        X_ref_white = X_ref_pca * inv_sqrt_eig
        center_white = np.mean(X_ref_white, axis=0, keepdims=True)
        center_pca = np.mean(X_ref_pca, axis=0, keepdims=True)

        # 5 Clipper consensus metrics on reference (labeled) bouts:
        # 1. Mahalanobis to center
        m1_mahal_center = np.linalg.norm(X_ref_white - center_white, axis=1)
        # 2. Euclidean to center (in PCA space)
        m2_euclid_center = np.linalg.norm(X_ref_pca - center_pca, axis=1)
        # 3. Mean pairwise Mahalanobis distance
        D_mahal = squareform(pdist(X_ref_white, metric="euclidean"))
        m3_mean_mahal_pair = np.sum(D_mahal, axis=1) / max(1, n_ref - 1)
        # 4. Mean pairwise Euclidean distance
        D_euclid = squareform(pdist(X_ref_pca, metric="euclidean"))
        m4_mean_euclid_pair = np.sum(D_euclid, axis=1) / max(1, n_ref - 1)
        # 5. k-NN distance in PCA space
        k_nn = max(1, min(5, int(np.sqrt(n_ref))))
        D_euclid_sorted = np.sort(D_euclid, axis=1)
        m5_knn_dist = np.mean(D_euclid_sorted[:, 1 : k_nn + 1], axis=1)

        metrics_5 = [
            m1_mahal_center,
            m2_euclid_center,
            m3_mean_mahal_pair,
            m4_mean_euclid_pair,
            m5_knn_dist,
        ]
        votes_ref = np.zeros(n_ref, dtype=np.int32)
        pct_sum_ref = np.zeros(n_ref, dtype=np.float32)
        for m_arr in metrics_5:
            thresh = np.percentile(m_arr, vote_percentile)
            votes_ref += (m_arr >= thresh).astype(np.int32)
            pct_sum_ref += _rank_percentile(m_arr)
        pct_mean_ref = pct_sum_ref / 5.0

        # LOF & IsolationForest on reference bouts
        if n_ref >= 8:
            n_neighbors_lof = min(20, max(3, n_ref // 3))
            lof = LocalOutlierFactor(n_neighbors=n_neighbors_lof, novelty=True)
            lof.fit(X_ref_pca)
            lof_raw_ref = -lof.decision_function(X_ref_pca)
            lof_pct_ref = _rank_percentile(lof_raw_ref)
        else:
            lof = None
            lof_pct_ref = pct_mean_ref.copy()

        if n_ref >= 10:
            iso = IsolationForest(n_estimators=80, random_state=42, n_jobs=1)
            iso.fit(X_ref_pca)
            iso_raw_ref = -iso.decision_function(X_ref_pca)
            iso_pct_ref = _rank_percentile(iso_raw_ref)
        else:
            iso = None
            iso_pct_ref = pct_mean_ref.copy()

        combined_ref = 0.60 * pct_mean_ref + 0.20 * lof_pct_ref + 0.20 * iso_pct_ref
        outlier_scores[ref_idx] = combined_ref
        outlier_votes[ref_idx] = votes_ref
        lof_scores[ref_idx] = lof_pct_ref
        iso_scores[ref_idx] = iso_pct_ref
        ood_scores[ref_idx] = _rank_percentile(m1_mahal_center)

        # Evaluate predicted bouts of the same class against the labeled reference cloud
        if pred_idx.size > 0 and lab_idx.size >= 4:
            X_pred_raw = np.nan_to_num(X_bout[pred_idx], nan=0.0, posinf=0.0, neginf=0.0)
            X_pred_scaled = _fast_robust_transform(X_pred_raw[:, std_mask], med_ref, iqr_ref)
            X_pred_pca = pca_full.transform(X_pred_scaled)[:, :n_comp]
            X_pred_white = X_pred_pca * inv_sqrt_eig

            p_m1 = np.linalg.norm(X_pred_white - center_white, axis=1)
            p_m2 = np.linalg.norm(X_pred_pca - center_pca, axis=1)
            p_m3 = np.mean(cdist(X_pred_white, X_ref_white, metric="euclidean"), axis=1)
            D_pr_euc = cdist(X_pred_pca, X_ref_pca, metric="euclidean")
            p_m4 = np.mean(D_pr_euc, axis=1)
            p_m5 = np.mean(np.sort(D_pr_euc, axis=1)[:, :k_nn], axis=1)

            pred_metrics = [p_m1, p_m2, p_m3, p_m4, p_m5]
            p_votes = np.zeros(pred_idx.size, dtype=np.int32)
            p_pct_sum = np.zeros(pred_idx.size, dtype=np.float32)
            for p_arr, ref_arr in zip(pred_metrics, metrics_5, strict=True):
                thresh = np.percentile(ref_arr, vote_percentile)
                p_votes += (p_arr >= thresh).astype(np.int32)
                # Empirical CDF against reference distribution
                cdf_vals = np.searchsorted(np.sort(ref_arr), p_arr, side="right") / float(n_ref)
                p_pct_sum += cdf_vals.astype(np.float32)

            p_pct_mean = p_pct_sum / 5.0
            if lof is not None:
                p_lof_raw = -lof.decision_function(X_pred_pca)
                p_lof_pct = (
                    np.searchsorted(np.sort(lof_raw_ref), p_lof_raw, side="right") / float(n_ref)
                ).astype(np.float32)
            else:
                p_lof_pct = p_pct_mean

            if iso is not None:
                p_iso_raw = -iso.decision_function(X_pred_pca)
                p_iso_pct = (
                    np.searchsorted(np.sort(iso_raw_ref), p_iso_raw, side="right") / float(n_ref)
                ).astype(np.float32)
            else:
                p_iso_pct = p_pct_mean

            outlier_scores[pred_idx] = 0.60 * p_pct_mean + 0.20 * p_lof_pct + 0.20 * p_iso_pct
            outlier_votes[pred_idx] = p_votes
            lof_scores[pred_idx] = p_lof_pct
            iso_scores[pred_idx] = p_iso_pct

            # OOD score: Mahalanobis distance relative to 95th percentile of labeled reference
            q95_mahal = max(1e-6, float(np.percentile(m1_mahal_center, 95.0)))
            ood_scores[pred_idx] = np.clip(p_m1 / (1.35 * q95_mahal), 0.0, 1.0).astype(np.float32)

    return outlier_scores, outlier_votes, lof_scores, iso_scores, ood_scores


def _compute_bout_transition_and_bounds(
    prob_curve: np.ndarray,
    start_frame: int,
    end_frame: int,
    *,
    min_discordant_frames: int = 5,
    min_core_frames: int = 8,
) -> tuple[float, tuple[int, int] | None, int | None]:
    """Compute `transition_score`, `suggested_trim_bounds`, and `suggested_split_frame`."""
    n = prob_curve.size
    if n < 2 * min_discordant_frames:
        return 0.0, None, None

    k_med = min(5, n if n % 2 == 1 else n - 1)
    p_smooth = median_filter(prob_curve.astype(np.float32), size=max(1, k_med), mode="nearest")

    raw_swing = float(np.max(p_smooth) - np.min(p_smooth))
    if raw_swing < 0.25:
        return round(raw_swing * 0.5, 4), None, None

    low_mask = p_smooth < 0.45
    high_mask = p_smooth >= 0.50

    # Find contiguous low-probability (discordant) runs
    padded = np.concatenate([[0], low_mask.astype(np.int8), [0]])
    diffs = np.diff(padded)
    low_starts = np.flatnonzero(diffs == 1)
    low_ends = np.flatnonzero(diffs == -1) - 1
    valid_low_runs = [
        (int(s), int(e))
        for s, e in zip(low_starts, low_ends, strict=True)
        if (e - s + 1) >= min_discordant_frames
    ]

    if not valid_low_runs or int(np.count_nonzero(high_mask)) < min_core_frames:
        return round(raw_swing * 0.65, 4), None, None

    transition_score = float(np.clip(raw_swing, 0.0, 1.0))

    # Check prefix and suffix discordant runs for Trim suggestion
    trim_s = start_frame
    trim_e = end_frame
    trimmed = False

    if valid_low_runs[0][0] <= 2:
        cand_s = start_frame + valid_low_runs[0][1] + 1
        if (trim_e - cand_s + 1) >= min_core_frames:
            trim_s = cand_s
            trimmed = True

    if valid_low_runs[-1][1] >= (n - 3):
        cand_e = start_frame + valid_low_runs[-1][0] - 1
        if (cand_e - trim_s + 1) >= min_core_frames:
            trim_e = cand_e
            trimmed = True

    if trimmed and (trim_s > start_frame or trim_e < end_frame):
        return transition_score, (int(trim_s), int(trim_e)), None

    # Otherwise check interior discordant run or sharpest step for Split suggestion
    grad = np.abs(np.diff(p_smooth))
    if grad.size >= 6:
        margin = max(3, min(min_discordant_frames, n // 4))
        interior_grad = grad[margin : n - margin]
        if interior_grad.size > 0:
            rel_idx = int(np.argmax(interior_grad)) + margin
            split_frame = int(start_frame + rel_idx)
            if (split_frame - start_frame) >= min_discordant_frames and (
                end_frame - split_frame
            ) >= min_discordant_frames:
                return transition_score, None, split_frame

    return transition_score, None, None


def compute_confusion_and_outliers(
    feature_matrix: BoutFeatureMatrix,
    *,
    frame_stride: int = 3,
    n_cv_splits: int = 5,
    knn_k: int = 7,
) -> ConfusionAndOutlierResult:
    """Compute Confident Learning OOF confusion, transition, and consensus outlier scores.

    Computes feature weights, RF leaf proximity matrix, `oof_top_class`, and
    `knn_majority_class` diagnostics.
    """
    records = feature_matrix.records
    X_bout = feature_matrix.X_bout
    n_bouts, n_stats = X_bout.shape
    n_base = len(feature_matrix.base_feature_names)

    if n_bouts == 0:
        return ConfusionAndOutlierResult(
            confusion_scores=np.zeros(0, dtype=np.float32),
            oof_mean_probs=np.zeros(0, dtype=np.float32),
            knn_disagreement=np.zeros(0, dtype=np.float32),
            ood_scores=np.zeros(0, dtype=np.float32),
            transition_scores=np.zeros(0, dtype=np.float32),
            suggested_trim_bounds=[],
            suggested_split_frame=[],
            outlier_scores=np.zeros(0, dtype=np.float32),
            outlier_votes=np.zeros(0, dtype=np.int32),
            lof_scores=np.zeros(0, dtype=np.float32),
            isoforest_scores=np.zeros(0, dtype=np.float32),
            base_feature_weights=np.ones(max(1, n_base), dtype=np.float32) / max(1, n_base),
            stat_feature_weights=np.ones(max(1, n_stats), dtype=np.float32),
            rf_proximity_matrix=None,
            oof_top_class=[],
            knn_majority_class=[],
        )

    class_names = [r.class_name for r in records]
    sources = [r.source for r in records]

    # 1. Within-class 5-metric consensus outlier detection + OOD scores
    outlier_scores, outlier_votes, lof_scores, iso_scores, ood_scores = (
        compute_within_class_consensus_outliers(X_bout, class_names, sources)
    )

    labeled_indices = [i for i, r in enumerate(records) if r.source == "labeled"]
    pred_indices = [i for i, r in enumerate(records) if r.source == "predicted"]

    unique_labeled_classes = sorted({records[i].class_name for i in labeled_indices})
    cls_to_int = {c: idx for idx, c in enumerate(unique_labeled_classes)}

    oof_mean_probs = np.full(n_bouts, 0.85, dtype=np.float32)
    oof_top_class: list[str] = [r.class_name for r in records]
    knn_majority_class: list[str] = [r.class_name for r in records]
    n_classes = max(1, len(unique_labeled_classes))
    oof_probs = np.zeros((n_bouts, n_classes), dtype=np.float32)
    has_frame_oof = np.zeros(n_bouts, dtype=bool)
    has_bout_oof = np.zeros(n_bouts, dtype=bool)
    base_weights = np.ones(max(1, n_base), dtype=np.float32) / max(1, n_base)
    stat_weights = np.ones(max(1, n_stats), dtype=np.float32)
    rf_proximity: np.ndarray | None = None

    # 2. If >= 2 labeled classes exist, run 5-fold StratifiedGroupKFold OOF probe
    if len(unique_labeled_classes) >= 2 and len(labeled_indices) >= 6:
        # 2A. Frame-level stride-3 train -> stride-1 predict OOF probe
        X_sub_list: list[np.ndarray] = []
        y_sub_list: list[np.ndarray] = []
        g_sub_list: list[np.ndarray] = []

        track_keys = sorted({records[i].track_key for i in labeled_indices})
        track_to_gid = {tk: g_idx for g_idx, tk in enumerate(track_keys)}

        total_lab_frames = sum(records[i].duration_frames for i in labeled_indices)
        eff_stride = max(frame_stride, 5) if total_lab_frames > 12_000 else max(1, frame_stride)

        for i in labeled_indices:
            rec = records[i]
            f_arr = feature_matrix.frame_features_by_bout.get(rec.bout_id)
            if f_arr is None or f_arr.shape[0] == 0:
                continue
            step = eff_stride if f_arr.shape[0] >= 8 else 1
            sub_f = f_arr[::step]
            c_idx = cls_to_int[rec.class_name]
            g_id = track_to_gid[rec.track_key]
            X_sub_list.append(sub_f)
            y_sub_list.append(np.full(sub_f.shape[0], c_idx, dtype=np.int32))
            g_sub_list.append(np.full(sub_f.shape[0], g_id, dtype=np.int32))

        if X_sub_list and len(track_keys) >= 2:
            # Sanitize NaNs before tree fitting so sklearn uses its fast non-missing Cython splitter
            X_sub = np.nan_to_num(np.vstack(X_sub_list), nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32, copy=False
            )
            y_sub = np.concatenate(y_sub_list)
            g_sub = np.concatenate(g_sub_list)

            n_splits_actual = min(n_cv_splits, len(track_keys))
            try:
                splitter = StratifiedGroupKFold(n_splits=n_splits_actual)
                splits = list(splitter.split(X_sub, y_sub, g_sub))
            except ValueError:
                splitter = GroupKFold(n_splits=n_splits_actual)
                splits = list(splitter.split(X_sub, y_sub, g_sub))

            # Map each track group_id to its held-out fold
            gid_to_bouts: dict[int, list[int]] = {g: [] for g in range(len(track_keys))}
            for i in labeled_indices:
                gid_to_bouts[track_to_gid[records[i].track_key]].append(i)

            imp_accum = np.zeros(n_base, dtype=np.float64)
            valid_folds = 0
            rf_jobs = min(4, os.cpu_count() or 1)

            for train_idx, test_idx in splits:
                y_tr = y_sub[train_idx]
                if np.unique(y_tr).size < 2:
                    continue
                rf_frame = ExtraTreesClassifier(
                    n_estimators=60,
                    max_depth=10,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    n_jobs=rf_jobs,
                    random_state=42 + valid_folds,
                )
                rf_frame.fit(X_sub[train_idx], y_tr)
                if rf_frame.feature_importances_.size == n_base:
                    imp_accum += rf_frame.feature_importances_
                valid_folds += 1

                rf_classes = [int(c) for c in rf_frame.classes_]
                heldout_gids = np.unique(g_sub[test_idx])

                # Batch all held-out bouts in this fold into a single predict_proba call
                fold_bouts: list[tuple[int, int, int | None, int]] = []
                fold_arrays: list[np.ndarray] = []
                for gid in heldout_gids:
                    for b_idx in gid_to_bouts.get(int(gid), []):
                        rec = records[b_idx]
                        target_c = cls_to_int[rec.class_name]
                        col_pos = rf_classes.index(target_c) if target_c in rf_classes else None
                        f_full = feature_matrix.frame_features_by_bout.get(rec.bout_id)
                        if f_full is None or f_full.shape[0] == 0:
                            continue
                        n_f = f_full.shape[0]
                        fold_bouts.append((b_idx, rec.bout_id, col_pos, n_f))
                        fold_arrays.append(f_full)

                if fold_arrays:
                    X_fold_eval = np.nan_to_num(
                        np.vstack(fold_arrays), nan=0.0, posinf=0.0, neginf=0.0
                    ).astype(np.float32, copy=False)
                    proba_fold_all = rf_frame.predict_proba(X_fold_eval).astype(np.float32)
                    cur = 0
                    for b_idx, bout_id, col_pos, n_f in fold_bouts:
                        bout_prob_chunk = proba_fold_all[cur : cur + n_f]
                        cur += n_f
                        mean_vec = np.mean(bout_prob_chunk, axis=0)
                        for local_col, cls_idx in enumerate(rf_classes):
                            if 0 <= cls_idx < n_classes:
                                oof_probs[b_idx, cls_idx] = float(mean_vec[local_col])
                        has_frame_oof[b_idx] = True
                        if col_pos is not None:
                            proba_full = bout_prob_chunk[:, col_pos]
                            feature_matrix.frame_probs_by_bout[bout_id] = proba_full
                            oof_mean_probs[b_idx] = float(mean_vec[col_pos])

            if valid_folds > 0 and np.sum(imp_accum) > 0:
                base_weights = (imp_accum / np.sum(imp_accum)).astype(np.float32)

        # 2B. Bout-level ExtraTrees RF for OOF bout probabilities, direct bout-stat weights,
        # and Breiman Leaf-Proximity Matrix
        X_lab = np.nan_to_num(X_bout[labeled_indices], nan=0.0, posinf=0.0, neginf=0.0)
        y_lab = np.array(
            [cls_to_int[records[i].class_name] for i in labeled_indices], dtype=np.int32
        )
        g_lab = np.array(
            [track_to_gid[records[i].track_key] for i in labeled_indices], dtype=np.int32
        )

        rf_bout_jobs = min(4, os.cpu_count() or 1)
        if len(track_keys) >= 2:
            n_splits_bout = min(n_cv_splits, len(track_keys))
            try:
                bout_splitter = StratifiedGroupKFold(n_splits=n_splits_bout)
                bout_splits = list(bout_splitter.split(X_lab, y_lab, g_lab))
            except ValueError:
                bout_splitter = GroupKFold(n_splits=n_splits_bout)
                bout_splits = list(bout_splitter.split(X_lab, y_lab, g_lab))

            for b_tr, b_te in bout_splits:
                y_b_tr = y_lab[b_tr]
                if np.unique(y_b_tr).size < 2:
                    continue
                rf_bout_cv = ExtraTreesClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=rf_bout_jobs,
                    random_state=142,
                )
                rf_bout_cv.fit(X_lab[b_tr], y_b_tr)
                rf_b_classes = [int(c) for c in rf_bout_cv.classes_]
                proba_b_te = rf_bout_cv.predict_proba(X_lab[b_te]).astype(np.float32)
                for local_pos, lab_pos in enumerate(b_te):
                    global_b_idx = labeled_indices[int(lab_pos)]
                    bout_vec = np.zeros(n_classes, dtype=np.float32)
                    for local_col, cls_idx in enumerate(rf_b_classes):
                        if 0 <= cls_idx < n_classes:
                            bout_vec[cls_idx] = float(proba_b_te[local_pos, local_col])
                    if has_frame_oof[global_b_idx]:
                        oof_probs[global_b_idx] = 0.65 * bout_vec + 0.35 * oof_probs[global_b_idx]
                    else:
                        oof_probs[global_b_idx] = bout_vec
                    has_bout_oof[global_b_idx] = True

                    target_c = int(y_lab[lab_pos])
                    if target_c in rf_b_classes:
                        col_pos = rf_b_classes.index(target_c)
                        p_bout_oof = float(proba_b_te[local_pos, col_pos])
                        # Blend bout-level OOF probability (which captures multi-frame IQR/MAD rhythm)
                        # with frame-level OOF mean probability
                        oof_mean_probs[global_b_idx] = float(
                            0.65 * p_bout_oof + 0.35 * oof_mean_probs[global_b_idx]
                        )

        rf_bout = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=rf_bout_jobs,
            random_state=42,
        )
        rf_bout.fit(X_lab, y_lab)
        bout_imp = rf_bout.feature_importances_.astype(np.float32)

        # Combine direct bout-stat RF importance with parent base-feature importance
        parent_imp = np.zeros(n_stats, dtype=np.float32)
        for s_idx, b_idx in enumerate(feature_matrix.stat_to_base_idx):
            if 0 <= b_idx < base_weights.size:
                parent_imp[s_idx] = base_weights[b_idx] / 7.0
            else:
                parent_imp[s_idx] = 1.0 / max(1, n_stats)

        if np.sum(bout_imp) > 0:
            bout_imp = bout_imp / np.sum(bout_imp)
        if np.sum(parent_imp) > 0:
            parent_imp = parent_imp / np.sum(parent_imp)

        blended = 0.60 * bout_imp + 0.40 * parent_imp
        # Floor tiny weights so no feature collapses to 0, and normalize mean weight to 1.0
        blended = np.maximum(blended, 0.05 / max(1, n_stats))
        stat_weights = (blended / np.mean(blended)).astype(np.float32)

        # Breiman Random Forest Leaf Co-Occurrence Proximity Matrix across all bouts (N x N)
        X_all_clean = np.nan_to_num(X_bout, nan=0.0, posinf=0.0, neginf=0.0)
        leaf_ids = rf_bout.apply(X_all_clean)  # shape: (N_bouts, n_trees)
        n_trees = int(leaf_ids.shape[1])
        if n_bouts > 0 and n_trees > 0:
            max_leaves = np.maximum(1, leaf_ids.max(axis=0).astype(np.int64) + 1)
            col_offsets = np.zeros(n_trees, dtype=np.int64)
            if n_trees > 1:
                col_offsets[1:] = np.cumsum(max_leaves[:-1])
            total_cols = int(col_offsets[-1] + max_leaves[-1])
            cols = (leaf_ids.astype(np.int64, copy=False) + col_offsets[None, :]).ravel()
            rows = np.repeat(np.arange(n_bouts, dtype=np.int32), n_trees)
            vals = np.ones(n_bouts * n_trees, dtype=np.float32)
            leaf_mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_bouts, total_cols))
            rf_proximity = (leaf_mat @ leaf_mat.T).toarray().astype(np.float32, copy=False)
            rf_proximity *= np.float32(1.0 / float(n_trees))
            np.fill_diagonal(rf_proximity, np.float32(1.0))
        else:
            rf_proximity = np.eye(n_bouts, dtype=np.float32)

        if pred_indices:
            rf_bout_classes = [int(c) for c in rf_bout.classes_]
            proba_pred_all = rf_bout.predict_proba(X_all_clean[pred_indices]).astype(np.float32)
            for local_pos, p_idx in enumerate(pred_indices):
                for local_col, cls_idx in enumerate(rf_bout_classes):
                    if 0 <= cls_idx < n_classes:
                        oof_probs[p_idx, cls_idx] = float(proba_pred_all[local_pos, local_col])
                has_bout_oof[p_idx] = True

        for idx in range(n_bouts):
            if has_frame_oof[idx] or has_bout_oof[idx]:
                oof_top_class[idx] = str(unique_labeled_classes[int(np.argmax(oof_probs[idx]))])

    # Populate stored prediction probabilities for predicted bouts
    for p_idx in pred_indices:
        rec = records[p_idx]
        if not np.isnan(rec.mean_pred_prob):
            oof_mean_probs[p_idx] = float(rec.mean_pred_prob)

    # 3. k-NN Disagreement in Classifier-Weighted (sqrt(w_d)) Space against Labeled Reference Cloud
    knn_disagreement = np.zeros(n_bouts, dtype=np.float32)
    if len(labeled_indices) >= 3:
        X_lab_raw = np.nan_to_num(X_bout[labeled_indices], nan=0.0, posinf=0.0, neginf=0.0)
        med_all, iqr_all = _fast_robust_fit(X_lab_raw)
        X_lab_std = _fast_robust_transform(X_lab_raw, med_all, iqr_all)
        X_all_raw = np.nan_to_num(X_bout, nan=0.0, posinf=0.0, neginf=0.0)
        X_all_std = _fast_robust_transform(X_all_raw, med_all, iqr_all)

        sqrt_w = np.sqrt(stat_weights)[None, :]
        top_d = min(64, n_stats)
        top_w_cols = np.argsort(stat_weights)[::-1][:top_d]
        X_lab_w = (X_lab_std * sqrt_w)[:, top_w_cols]
        X_all_w = (X_all_std * sqrt_w)[:, top_w_cols]

        k_eff = min(knn_k, len(labeled_indices) - 1)
        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
        nn.fit(X_lab_w)
        _, neigh_pos = nn.kneighbors(X_all_w)

        lab_classes = np.array([records[i].class_name for i in labeled_indices], dtype=object)
        for idx, rec in enumerate(records):
            row_neigh = neigh_pos[idx]
            # If this bout is in labeled_indices, drop self-match (first neighbor)
            if rec.source == "labeled":
                actual_global_idx = [
                    labeled_indices[p] for p in row_neigh if labeled_indices[p] != idx
                ][:k_eff]
                neigh_cls = np.array(
                    [records[g].class_name for g in actual_global_idx], dtype=object
                )
            else:
                neigh_cls = lab_classes[row_neigh[:k_eff]]
            if neigh_cls.size > 0:
                knn_disagreement[idx] = float(np.mean(neigh_cls != rec.class_name))
                if len(unique_labeled_classes) >= 2:
                    cls_counts: dict[str, int] = {}
                    for nc in neigh_cls:
                        c_str = str(nc)
                        cls_counts[c_str] = cls_counts.get(c_str, 0) + 1
                    # Preserve nearest-neighbor insertion order so ties break by closest neighbor
                    knn_majority_class[idx] = max(cls_counts.items(), key=lambda kv: kv[1])[0]

    # 4. Final Confusion Score (Labeled) & Prediction Doubt Score (Predicted) + Transition Bounds
    confusion_scores = np.zeros(n_bouts, dtype=np.float32)
    transition_scores = np.zeros(n_bouts, dtype=np.float32)
    suggested_trim_bounds: list[tuple[int, int] | None] = [None] * n_bouts
    suggested_split_frame: list[int | None] = [None] * n_bouts

    for idx, rec in enumerate(records):
        if rec.source == "labeled":
            if len(unique_labeled_classes) >= 2:
                confusion_scores[idx] = float(
                    np.clip(
                        0.50 * (1.0 - oof_mean_probs[idx]) + 0.50 * knn_disagreement[idx],
                        0.0,
                        1.0,
                    )
                )
            else:
                confusion_scores[idx] = float(outlier_scores[idx])
        else:
            # Predicted bout doubt score combines margin uncertainty, labeled k-NN disagreement, and OOD
            margin_unc = float(np.clip(1.0 - oof_mean_probs[idx], 0.0, 1.0))
            confusion_scores[idx] = float(
                np.clip(
                    0.40 * margin_unc + 0.40 * knn_disagreement[idx] + 0.20 * ood_scores[idx],
                    0.0,
                    1.0,
                )
            )

        p_curve = feature_matrix.frame_probs_by_bout.get(rec.bout_id)
        if p_curve is not None and p_curve.size >= 10:
            t_score, trim_b, split_f = _compute_bout_transition_and_bounds(
                p_curve, rec.start_frame, rec.end_frame
            )
            transition_scores[idx] = t_score
            suggested_trim_bounds[idx] = trim_b
            suggested_split_frame[idx] = split_f

    return ConfusionAndOutlierResult(
        confusion_scores=confusion_scores,
        oof_mean_probs=oof_mean_probs,
        knn_disagreement=knn_disagreement,
        ood_scores=ood_scores,
        transition_scores=transition_scores,
        suggested_trim_bounds=suggested_trim_bounds,
        suggested_split_frame=suggested_split_frame,
        outlier_scores=outlier_scores,
        outlier_votes=outlier_votes,
        lof_scores=lof_scores,
        isoforest_scores=iso_scores,
        base_feature_weights=base_weights,
        stat_feature_weights=stat_weights,
        rf_proximity_matrix=rf_proximity,
        oof_top_class=oof_top_class,
        knn_majority_class=knn_majority_class,
    )
