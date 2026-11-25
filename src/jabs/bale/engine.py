"""Main orchestration engine for `jabs.bale`."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from jabs.core.enums import ClassifierMode

from .bout_extractor import collect_labeled_bout_records, extract_bout_feature_matrix
from .clustering_and_embedding import compute_clustering_and_embedding
from .models import BaleReport, FeatureLens, RepresentationSpace, SubtypeScope
from .outlier_and_confusion import compute_confusion_and_outliers
from .prediction_extractor import collect_predicted_bout_records
from .relabel_guard import count_dual_class_cv_groups
from .spatial_localization import extract_bout_spatial_locations

if TYPE_CHECKING:
    from jabs.project import Project


def run_bale(
    project: Project,
    behavior: str | None = None,
    *,
    include_positive: bool = True,
    include_negative: bool = True,
    include_all_multiclass: bool = False,
    include_predictions: bool = False,
    max_predicted_bouts: int = 500,
    use_postprocessed: bool = True,
    synthesize_predictions_if_missing: bool = False,
    space: RepresentationSpace = "classifier",
    subtype_scope: SubtypeScope = "all_labeled",
    feature_lens: FeatureLens = "all",
    target_behavior: str | None = None,
    compute_tsne: bool = False,
    compute_mds: bool = True,
    frame_stride: int = 3,
    n_cv_splits: int = 5,
    knn_k: int = 7,
    forced_k: int | None = None,
    compute_spatial: bool = False,
) -> BaleReport:
    """Run the complete BALE - Bout & Label Explorer pipeline.

    Args:
        project: Open JABS `Project` instance.
        behavior: Target behavior name (defaults to the first project behavior).
        include_positive: Include positive (`present: true`) labeled bouts.
        include_negative: Include negative (`present: false` / `"None"`) labeled bouts.
        include_all_multiclass: In multi-class mode, include all behavior classes simultaneously.
        include_predictions: Also extract unlabeled predicted bouts from `jabs/predictions/*.h5`.
        max_predicted_bouts: Maximum number of predicted bouts to sample in Stage B (`70%` top doubt
            + `30%` stratified across probability deciles).
        use_postprocessed: Use `predictions_post_processed` (when present) to segment predicted bouts.
        synthesize_predictions_if_missing: If True and no `jabs/predictions/*.h5` files exist,
            allow cross-validated ExtraTrees synthesis of predicted bouts as a fallback.
        space: Representation geometry (`"classifier"`, `"unsupervised"`, or `"rf_proximity"`).
        subtype_scope: Cohort scope (`"pos_only"` or `"all_labeled"`) for clustering/PCA fit.
        feature_lens: Ethological feature lens (`"all"`, `"intrinsic"`, `"locomotion"`,
            `"posture"`, `"social"`, or `"classifier_weighted"`).
        target_behavior: Optional multi-class behavior filter when fitting sub-type clusters.
        compute_tsne: Whether to compute 2D PCA-initialized t-SNE coordinates.
        compute_mds: Whether to compute 2D Classical PCoA / MDS coordinates on the RF proximity matrix.
        frame_stride: Temporal stride for training the out-of-fold frame probe (`default=3`).
        n_cv_splits: Number of animal-track CV folds for the OOF probe (`default=5`).
        knn_k: Number of nearest labeled neighbors for disagreement scoring (`default=7`).
        forced_k: Optional fixed cluster count `k` for Ward.D2 sub-type clustering.
        compute_spatial: Whether to extract per-bout 2D arena coordinates (`report.spatial`).

    Returns:
        Populated `BaleReport` containing bout records, feature matrices, diagnostic scores,
        2D embeddings, sub-type clusters, CV group counts, and per-stage `timings_ms`.
    """
    t0 = time.perf_counter()
    timings_ms: dict[str, float] = {}

    sm = project.settings_manager
    is_multiclass = sm.classifier_mode == ClassifierMode.MULTICLASS
    behaviors = list(sm.behavior_names)
    active_behavior = behavior or (behaviors[0] if behaviors else "")

    # Stage 1: Collect labeled (+ optional predicted) bout records
    t_stage = time.perf_counter()
    labeled_records = collect_labeled_bout_records(
        project,
        active_behavior,
        include_positive=include_positive,
        include_negative=include_negative,
        include_all_multiclass=include_all_multiclass,
    )

    pred_records = []
    frame_probs_pred: dict[int, object] = {}
    raw_preds_pred: dict[int, object] = {}
    if include_predictions and max_predicted_bouts > 0:
        pred_beh = None if (is_multiclass and include_all_multiclass) else active_behavior
        pred_records, frame_probs_pred, raw_preds_pred = collect_predicted_bout_records(
            project,
            pred_beh,
            use_postprocessed=use_postprocessed,
            max_predicted_bouts=max_predicted_bouts,
            synthesize_if_missing=synthesize_predictions_if_missing,
        )

    all_records = [*labeled_records, *pred_records]
    timings_ms["collect_bouts_ms"] = (time.perf_counter() - t_stage) * 1000.0

    # Stage 2: Track-grouped Parquet feature extraction + fast order-statistic quantiles
    t_stage = time.perf_counter()
    # Map temporary predicted bout_ids before re-indexing in extract_bout_feature_matrix
    pred_temp_key_to_probs = {
        (r.video, r.identity, r.start_frame, r.end_frame, r.class_name): (
            frame_probs_pred.get(r.bout_id),
            raw_preds_pred.get(r.bout_id),
        )
        for r in pred_records
    }
    feature_matrix = extract_bout_feature_matrix(
        project,
        all_records,
        active_behavior,
        retain_frame_features=True,
    )
    for rec in feature_matrix.records:
        if rec.source == "predicted":
            p_arr, r_arr = pred_temp_key_to_probs.get(
                (rec.video, rec.identity, rec.start_frame, rec.end_frame, rec.class_name),
                (None, None),
            )
            if p_arr is not None:
                feature_matrix.frame_probs_by_bout[rec.bout_id] = p_arr  # type: ignore[assignment]
            if r_arr is not None:
                feature_matrix.raw_preds_by_bout[rec.bout_id] = r_arr  # type: ignore[assignment]
    timings_ms["extract_features_ms"] = (time.perf_counter() - t_stage) * 1000.0

    # Stage 3: Out-of-fold Confident Learning confusion, transition scoring & consensus outliers
    t_stage = time.perf_counter()
    diagnostics = compute_confusion_and_outliers(
        feature_matrix,
        frame_stride=frame_stride,
        n_cv_splits=n_cv_splits,
        knn_k=knn_k,
    )
    timings_ms["diagnostics_ms"] = (time.perf_counter() - t_stage) * 1000.0

    # Stage 4: 2D embeddings (PCA, t-SNE, RF-Proximity PCoA) & Ward.D2 sub-type clustering
    t_stage = time.perf_counter()
    embedding = compute_clustering_and_embedding(
        feature_matrix,
        diagnostics,
        space=space,
        subtype_scope=subtype_scope,
        feature_lens=feature_lens,
        target_behavior=target_behavior,
        compute_tsne=compute_tsne,
        compute_mds=compute_mds,
        forced_k=forced_k,
    )
    timings_ms["embedding_ms"] = (time.perf_counter() - t_stage) * 1000.0

    spatial_res = None
    if compute_spatial:
        t_stage = time.perf_counter()
        spatial_res = extract_bout_spatial_locations(project, feature_matrix.records)
        timings_ms["spatial_ms"] = (time.perf_counter() - t_stage) * 1000.0

    # Stage 5: Cross-validation dual-class group audit (`Dual-Class CV rule`)
    t_stage = time.perf_counter()
    cv_groups_count, jabs_met, dual_class_cv_met, cv_track_counts = count_dual_class_cv_groups(
        project, active_behavior
    )
    timings_ms["cv_guard_ms"] = (time.perf_counter() - t_stage) * 1000.0
    timings_ms["total_ms"] = (time.perf_counter() - t0) * 1000.0

    return BaleReport(
        project_dir=str(Path(project.dir).resolve()),
        behavior=active_behavior,
        is_multiclass=is_multiclass,
        space=space,
        feature_matrix=feature_matrix,
        diagnostics=diagnostics,
        embedding=embedding,
        cv_groups_count=cv_groups_count,
        jabs_threshold_met=jabs_met,
        dual_class_cv_met=dual_class_cv_met,
        timings_ms=timings_ms,
        cv_track_counts=cv_track_counts,
        include_all_multiclass=bool(is_multiclass and include_all_multiclass),
        spatial=spatial_res,
    )


def recompute_embedding_only(
    report: BaleReport,
    *,
    space: RepresentationSpace | None = None,
    subtype_scope: SubtypeScope | None = None,
    feature_lens: FeatureLens | None = None,
    target_behavior: str | None = None,
    compute_tsne: bool = False,
    compute_mds: bool = True,
    forced_k: int | None = None,
) -> BaleReport:
    """Warm-recompute only the 2D embeddings and Ward.D2 clustering (`< 60 ms`).

    Used when the user switches between Classifier-Weighted, Unsupervised, RF-Proximity
    spaces, ethological feature lenses, subtype cohort scope, or cluster count `k`
    in the GUI without changing the underlying bout set.
    """
    t0 = time.perf_counter()
    new_space: RepresentationSpace = space if space is not None else report.space
    new_scope: SubtypeScope = (
        subtype_scope if subtype_scope is not None else report.embedding.subtype_scope
    )
    new_lens: FeatureLens = (
        feature_lens if feature_lens is not None else report.embedding.feature_lens
    )
    new_emb = compute_clustering_and_embedding(
        report.feature_matrix,
        report.diagnostics,
        space=new_space,
        subtype_scope=new_scope,
        feature_lens=new_lens,
        target_behavior=target_behavior,
        compute_tsne=compute_tsne,
        compute_mds=compute_mds,
        forced_k=forced_k,
    )
    warm_ms = (time.perf_counter() - t0) * 1000.0
    new_timings = dict(report.timings_ms)
    new_timings["warm_recompute_ms"] = warm_ms
    return BaleReport(
        project_dir=report.project_dir,
        behavior=report.behavior,
        is_multiclass=report.is_multiclass,
        space=new_space,
        feature_matrix=report.feature_matrix,
        diagnostics=report.diagnostics,
        embedding=new_emb,
        cv_groups_count=report.cv_groups_count,
        jabs_threshold_met=report.jabs_threshold_met,
        dual_class_cv_met=report.dual_class_cv_met,
        timings_ms=new_timings,
        cv_track_counts=dict(report.cv_track_counts),
        include_all_multiclass=report.include_all_multiclass,
        spatial=report.spatial,
    )
