"""Data models for the BALE - Bout & Label Explorer (`jabs.bale`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

BoutSource = Literal["labeled", "predicted"]
RepresentationSpace = Literal["classifier", "unsupervised", "rf_proximity"]
ProjectionMethod = Literal["pca", "tsne", "mds", "spatial"]
SubtypeScope = Literal["pos_only", "all_labeled"]
FeatureLens = Literal["all", "intrinsic", "locomotion", "posture", "social", "classifier_weighted"]


@dataclass
class BoutRecord:
    """Metadata for a single labeled or predicted bout."""

    bout_id: int
    source: BoutSource
    video: str
    identity: int
    behavior: str
    class_name: str
    is_positive: bool
    start_frame: int
    end_frame: int
    duration_frames: int
    valid_pose_fraction: float = 1.0
    mean_pred_prob: float = float("nan")
    postproc_fragility: float = 0.0
    origin_source: BoutSource | None = None
    origin_behavior: str = ""
    is_label_halo: bool = False
    halo_side: Literal["before", "after", "between", ""] = ""
    halo_distance_frames: int | None = None

    def __post_init__(self) -> None:
        if self.origin_source is None:
            self.origin_source = self.source
        if not self.origin_behavior:
            self.origin_behavior = self.behavior

    @property
    def track_key(self) -> tuple[str, int]:
        """Return the `(video, identity)` cross-validation track key."""
        return (self.video, self.identity)


@dataclass
class BoutFeatureMatrix:
    """Extracted bout-level feature matrix and per-bout frame probability curves."""

    records: list[BoutRecord]
    X_bout: np.ndarray  # shape: (N_bouts, D_stats), dtype=float32
    stat_names: list[str]
    base_feature_names: list[str]
    stat_to_base_idx: np.ndarray  # shape: (D_stats,), maps each stat col to base_feature_names
    frame_probs_by_bout: dict[int, np.ndarray] = field(default_factory=dict)
    raw_preds_by_bout: dict[int, np.ndarray] = field(default_factory=dict)
    frame_features_by_bout: dict[int, np.ndarray] = field(default_factory=dict)
    skipped_stale_cache_tracks: int = 0
    skipped_missing_cache_tracks: int = 0

    @property
    def skipped_feature_cache_tracks(self) -> int:
        """Total `(video, identity)` tracks skipped due to stale or missing feature caches."""
        return int(self.skipped_stale_cache_tracks + self.skipped_missing_cache_tracks)


@dataclass
class ConfusionAndOutlierResult:
    """Diagnostic scores for label confusion, chimeric transitions, and kinematic outliers."""

    confusion_scores: (
        np.ndarray
    )  # shape: (N_bouts,), [0, 1] (confusion for labeled, doubt for pred)
    oof_mean_probs: np.ndarray  # shape: (N_bouts,), [0, 1]
    knn_disagreement: np.ndarray  # shape: (N_bouts,), [0, 1]
    ood_scores: np.ndarray  # shape: (N_bouts,), [0, 1]
    transition_scores: np.ndarray  # shape: (N_bouts,), [0, 1]
    suggested_trim_bounds: list[tuple[int, int] | None]
    suggested_split_frame: list[int | None]
    outlier_scores: np.ndarray  # shape: (N_bouts,), [0, 1]
    outlier_votes: np.ndarray  # shape: (N_bouts,), int in 0..5 (Clipper 5-metric consensus)
    lof_scores: np.ndarray  # shape: (N_bouts,), [0, 1]
    isoforest_scores: np.ndarray  # shape: (N_bouts,), [0, 1]
    base_feature_weights: np.ndarray  # shape: (F_base,), normalized sum=1
    stat_feature_weights: np.ndarray  # shape: (D_stats,), normalized mean=1
    rf_proximity_matrix: np.ndarray | None = None
    oof_top_class: list[str] = field(default_factory=list)
    knn_majority_class: list[str] = field(default_factory=list)


@dataclass
class ClusterAndEmbeddingResult:
    """2D representation embeddings and sub-type clustering results."""

    coords_pca_2d: np.ndarray  # shape: (N_bouts, 2)
    pca_explained_variance_ratio: tuple[float, float]
    coords_tsne_2d: np.ndarray | None
    coords_mds_2d: np.ndarray | None
    cluster_labels: np.ndarray  # shape: (N_bouts,), 1..k
    optimal_k: int
    silhouette: float
    cluster_top_features: dict[int, list[tuple[str, float]]]
    top_z_diffs_per_bout: list[list[tuple[str, float]]]
    cluster_z_matrix: np.ndarray | None = None  # shape: (optimal_k, top_m)
    cluster_z_feature_names: list[str] = field(default_factory=list)
    cluster_feature_eta2: dict[str, float] = field(default_factory=dict)
    cluster_counts: dict[int, int] = field(default_factory=dict)
    cage_skew_p: float | None = None
    cage_skew_cramers_v: float | None = None
    dominant_cage_by_cluster: dict[int, tuple[str, float]] = field(default_factory=dict)
    subtype_scope: SubtypeScope = "all_labeled"
    feature_lens: FeatureLens = "all"
    forced_k: int | None = None
    target_behavior: str | None = None


@dataclass
class CVGroupImpact:
    """Impact of a proposed relabel/delete/promote edit on JABS CV group invariants."""

    before_groups: int
    after_groups: int
    jabs_threshold_met_before: bool
    jabs_threshold_met_after: bool
    dual_class_cv_met_before: bool
    dual_class_cv_met_after: bool
    safe_to_apply: bool
    warning_message: str | None = None
    suggested_pairing_bout: BoutRecord | None = None


@dataclass
class BoutSpatialResult:
    """2D arena coordinates (`(x, y)`) and optional background landmarks for bouts."""

    coords_xy: npt.NDArray[np.float32]  # shape: (N_bouts, 2), median body centroid
    nose_xy: npt.NDArray[np.float32]  # shape: (N_bouts, 2), median nose position
    valid_mask: npt.NDArray[np.bool_]  # shape: (N_bouts,)
    frame_width: float | None = None
    frame_height: float | None = None
    normalized: bool = False
    reference_frame_gray: npt.NDArray[np.uint8] | None = None
    reference_video: str | None = None
    static_objects: dict[str, npt.NDArray[np.float32]] = field(default_factory=dict)
    cm_per_pixel: float | None = None
    elapsed_ms: float = 0.0


@dataclass
class BaleReport:
    """Complete analysis output for `jabs.bale`."""

    project_dir: str
    behavior: str
    is_multiclass: bool
    space: RepresentationSpace
    feature_matrix: BoutFeatureMatrix
    diagnostics: ConfusionAndOutlierResult
    embedding: ClusterAndEmbeddingResult
    cv_groups_count: int
    jabs_threshold_met: bool
    dual_class_cv_met: bool
    timings_ms: dict[str, float] = field(default_factory=dict)
    cv_track_counts: dict[tuple[str, int], tuple[int, int]] = field(default_factory=dict)
    include_all_multiclass: bool = False
    spatial: BoutSpatialResult | None = None
    cv_grouping_regex: str | None = None
    mc_counts_by_behavior: dict[str, dict[str, dict[int, dict[str, tuple[int, int]]]]] | None = (
        None
    )

    def to_dataframe(self, exclude_bout_ids: set[int] | None = None) -> pd.DataFrame:
        """Convert all bout records, coordinates, and diagnostic scores to a DataFrame."""
        rows: list[dict[str, Any]] = []
        fm = self.feature_matrix
        diag = self.diagnostics
        emb = self.embedding

        for idx, rec in enumerate(fm.records):
            if exclude_bout_ids and rec.bout_id in exclude_bout_ids:
                continue
            trim = diag.suggested_trim_bounds[idx]
            split_f = diag.suggested_split_frame[idx]
            top_feats = (
                emb.top_z_diffs_per_bout[idx] if idx < len(emb.top_z_diffs_per_bout) else []
            )
            top_feat_str = "; ".join(f"{name} ({z:+.2f}SD)" for name, z in top_feats[:3])

            if rec.source == "predicted":
                if rec.is_label_halo:
                    side_lbl = (
                        "onset"
                        if rec.halo_side == "before"
                        else ("offset" if rec.halo_side == "after" else "bridge")
                    )
                    action = f"Label Edge ({side_lbl} halo, {rec.start_frame}-{rec.end_frame})"
                elif diag.confusion_scores[idx] >= 0.70 and diag.knn_disagreement[idx] >= 0.50:
                    action = f"Mark False Pos -> Add Neg ({rec.start_frame}-{rec.end_frame})"
                elif rec.postproc_fragility >= 0.35:
                    action = "Split Stitched Gap"
                elif diag.ood_scores[idx] >= 0.85:
                    action = "Inspect OOD Novelty"
                else:
                    action = f"Confirm -> Add Pos ({rec.start_frame}-{rec.end_frame})"
            else:
                if trim is not None and diag.transition_scores[idx] >= 0.55:
                    action = f"Trim Ends -> {trim[0]}-{trim[1]}"
                elif split_f is not None and diag.transition_scores[idx] >= 0.55:
                    action = f"Split @ {split_f}"
                elif diag.confusion_scores[idx] >= 0.70:
                    target = "Neg" if rec.is_positive else "Pos"
                    action = f"Flip -> {target} (Inspect)"
                elif diag.outlier_scores[idx] >= 0.80:
                    action = f"Inspect Outlier ({int(diag.outlier_votes[idx])}/5 votes)"
                else:
                    action = "Keep (Clean)"

            row_dict: dict[str, Any] = {
                "bout_id": rec.bout_id,
                "source": rec.source,
                "video": rec.video,
                "identity": rec.identity,
                "behavior": rec.behavior,
                "class_name": rec.class_name,
                "is_positive": rec.is_positive,
                "start_frame": rec.start_frame,
                "end_frame": rec.end_frame,
                "duration_frames": rec.duration_frames,
                "is_label_halo": bool(rec.is_label_halo),
                "halo_side": rec.halo_side,
                "halo_distance_frames": rec.halo_distance_frames,
                "valid_pose_fraction": round(float(rec.valid_pose_fraction), 4),
                "confusion_score": round(float(diag.confusion_scores[idx]), 4),
                "confusion_or_doubt_score": round(float(diag.confusion_scores[idx]), 4),
                "oof_or_pred_prob": round(float(diag.oof_mean_probs[idx]), 4),
                "knn_disagreement": round(float(diag.knn_disagreement[idx]), 4),
                "transition_score": round(float(diag.transition_scores[idx]), 4),
                "postproc_fragility": round(float(rec.postproc_fragility), 4),
                "outlier_score": round(float(diag.outlier_scores[idx]), 4),
                "outlier_votes_5": int(diag.outlier_votes[idx]),
                "ood_score": round(float(diag.ood_scores[idx]), 4),
                "cluster_id": int(emb.cluster_labels[idx]),
                "pc1": round(float(emb.coords_pca_2d[idx, 0]), 4),
                "pc2": round(float(emb.coords_pca_2d[idx, 1]), 4),
                "tsne1": (
                    round(float(emb.coords_tsne_2d[idx, 0]), 4)
                    if emb.coords_tsne_2d is not None
                    else float("nan")
                ),
                "tsne2": (
                    round(float(emb.coords_tsne_2d[idx, 1]), 4)
                    if emb.coords_tsne_2d is not None
                    else float("nan")
                ),
                "suggested_trim_start": trim[0] if trim is not None else None,
                "suggested_trim_end": trim[1] if trim is not None else None,
                "suggested_split_frame": split_f,
                "top_divergent_features": top_feat_str,
                "recommended_action": action,
            }
            if self.spatial is not None and idx < len(self.spatial.coords_xy):
                sx = float(self.spatial.coords_xy[idx, 0])
                sy = float(self.spatial.coords_xy[idx, 1])
                row_dict["spatial_x"] = round(sx, 4) if np.isfinite(sx) else float("nan")
                row_dict["spatial_y"] = round(sy, 4) if np.isfinite(sy) else float("nan")
            rows.append(row_dict)
        return pd.DataFrame(rows)

    def to_dict(self, top_n: int = 50) -> dict[str, Any]:
        """Serialize summary metrics and top flagged bouts to a JSON-friendly dictionary."""
        df = self.to_dataframe()
        labeled_df = df[df["source"] == "labeled"]
        pred_df = df[df["source"] == "predicted"]

        top_confusing = (
            labeled_df.sort_values("confusion_or_doubt_score", ascending=False)
            .head(top_n)
            .to_dict(orient="records")
            if not labeled_df.empty
            else []
        )
        top_chimeric = (
            labeled_df.sort_values("transition_score", ascending=False)
            .head(top_n)
            .to_dict(orient="records")
            if not labeled_df.empty
            else []
        )
        top_outliers = (
            labeled_df.sort_values("outlier_score", ascending=False)
            .head(top_n)
            .to_dict(orient="records")
            if not labeled_df.empty
            else []
        )
        top_doubt_predictions = (
            pred_df.sort_values("confusion_or_doubt_score", ascending=False)
            .head(top_n)
            .to_dict(orient="records")
            if not pred_df.empty
            else []
        )

        return {
            "project_dir": self.project_dir,
            "behavior": self.behavior,
            "is_multiclass": self.is_multiclass,
            "space": self.space,
            "total_bouts": len(df),
            "labeled_bouts": len(labeled_df),
            "predicted_bouts": len(pred_df),
            "skipped_stale_cache_tracks": int(self.feature_matrix.skipped_stale_cache_tracks),
            "skipped_missing_cache_tracks": int(self.feature_matrix.skipped_missing_cache_tracks),
            "high_confusion_labeled_count": int(
                (labeled_df["confusion_or_doubt_score"] >= 0.70).sum()
            )
            if not labeled_df.empty
            else 0,
            "chimeric_labeled_count": int((labeled_df["transition_score"] >= 0.55).sum())
            if not labeled_df.empty
            else 0,
            "outlier_labeled_count": int((labeled_df["outlier_score"] >= 0.80).sum())
            if not labeled_df.empty
            else 0,
            "high_doubt_predicted_count": int((pred_df["confusion_or_doubt_score"] >= 0.70).sum())
            if not pred_df.empty
            else 0,
            "cv_groups_count": self.cv_groups_count,
            "jabs_threshold_met": self.jabs_threshold_met,
            "dual_class_cv_met": self.dual_class_cv_met,
            "optimal_k_clusters": self.embedding.optimal_k,
            "cluster_silhouette": round(float(self.embedding.silhouette), 4),
            "pca_explained_variance_ratio": [
                round(float(v), 4) for v in self.embedding.pca_explained_variance_ratio
            ],
            "timings_ms": {k: round(float(v), 2) for k, v in self.timings_ms.items()},
            "top_confusing_labeled_bouts": top_confusing,
            "top_chimeric_labeled_bouts": top_chimeric,
            "top_outlier_labeled_bouts": top_outliers,
            "top_doubt_predicted_bouts": top_doubt_predictions,
        }
