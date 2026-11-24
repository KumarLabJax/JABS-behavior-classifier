"""BALE - Bout & Label Explorer (`jabs.bale`).

Provides interactive and headless CLI tools to inspect human-labeled bouts and
unlabeled predicted bouts in classifier-weighted, unsupervised, and RF-proximity
representation spaces, rank cross-class confusion and boundary transitions, and
perform 1-click relabeling and active-learning promotion with `Dual-Class CV rule` CV group guards.
"""

from .bout_extractor import (
    STAT_SUFFIXES,
    clear_bout_extractor_cache,
    collect_labeled_bout_records,
    compute_fast_bout_quantiles,
    extract_bout_feature_matrix,
    invalidate_track_cache,
    resolve_project_video_name,
)
from .clustering_and_embedding import (
    compute_clustering_and_embedding,
    select_feature_lens_mask,
)
from .engine import recompute_embedding_only, run_bale
from .examined_store import ExaminedBoutStore
from .models import (
    BaleReport,
    BoutFeatureMatrix,
    BoutRecord,
    BoutSource,
    ClusterAndEmbeddingResult,
    ConfusionAndOutlierResult,
    CVGroupImpact,
    FeatureLens,
    ProjectionMethod,
    RepresentationSpace,
    SubtypeScope,
)
from .outlier_and_confusion import (
    compute_confusion_and_outliers,
    compute_within_class_consensus_outliers,
)
from .prediction_extractor import collect_predicted_bout_records
from .relabel_guard import (
    BaleUndoStack,
    BoutEditSnapshot,
    EditAction,
    apply_bout_edit,
    count_dual_class_cv_groups,
    evaluate_cv_group_impact,
    find_dual_class_pairing_candidate,
    simulate_track_count_delta,
    summarize_cv_track_counts,
)
from .trajectory_and_export import (
    BoutPoseTrajectory,
    export_bale_summary_figure,
    load_bout_pose_trajectory,
    load_pose_cached,
)
from .video_screener import (
    VideoFocalEvent,
    VideoMetricSource,
    VideoScreenerReport,
    VideoScreeningPreset,
    VideoSummaryRecord,
    classify_video_outliers,
    filter_video_records,
    summarize_project_videos,
)

__all__ = [
    "STAT_SUFFIXES",
    "BaleReport",
    "BaleUndoStack",
    "BoutEditSnapshot",
    "BoutFeatureMatrix",
    "BoutPoseTrajectory",
    "BoutRecord",
    "BoutSource",
    "CVGroupImpact",
    "ClusterAndEmbeddingResult",
    "ConfusionAndOutlierResult",
    "EditAction",
    "ExaminedBoutStore",
    "FeatureLens",
    "ProjectionMethod",
    "RepresentationSpace",
    "SubtypeScope",
    "VideoFocalEvent",
    "VideoMetricSource",
    "VideoScreenerReport",
    "VideoScreeningPreset",
    "VideoSummaryRecord",
    "apply_bout_edit",
    "classify_video_outliers",
    "clear_bout_extractor_cache",
    "collect_labeled_bout_records",
    "collect_predicted_bout_records",
    "compute_clustering_and_embedding",
    "compute_fast_bout_quantiles",
    "compute_within_class_consensus_outliers",
    "count_dual_class_cv_groups",
    "evaluate_cv_group_impact",
    "export_bale_summary_figure",
    "extract_bout_feature_matrix",
    "filter_video_records",
    "find_dual_class_pairing_candidate",
    "invalidate_track_cache",
    "load_bout_pose_trajectory",
    "load_pose_cached",
    "recompute_embedding_only",
    "resolve_project_video_name",
    "run_bale",
    "select_feature_lens_mask",
    "simulate_track_count_delta",
    "summarize_cv_track_counts",
    "summarize_project_videos",
]
