"""
The `jabs.classifier` package provides tools for training, evaluating, saving, and loading machine learning classifiers for behavioral data analysis.

It includes the `Classifier` class, which supports multiple classification algorithms (such as Random Forest,
Gradient Boosting, and XGBoost), utilities for feature management, data splitting, model evaluation, and serialization.`
"""

from .classifier import Classifier
from .cross_validation import run_leave_one_group_out_cv
from .cv_postprocessing import (
    FoldPostprocessingEvaluation,
    enabled_stage_configs,
    evaluate_group_with_postprocessing,
)
from .inference import IdentityPrediction, predict_identity
from .mlflow_logging import (
    MlflowLoggingError,
    log_cross_validation_to_mlflow,
    mlflow_available,
    parse_kv_tags,
)
from .multi_class_classifier import MultiClassClassifier
from .protocols import ClassifierProtocol
from .training_report import (
    BinaryCVResult,
    CrossValidationResult,
    MultiClassCVResult,
    PostprocessedMetrics,
    TrainingReportData,
    generate_markdown_report,
    save_training_report,
)

__all__ = [
    "BinaryCVResult",
    "Classifier",
    "ClassifierProtocol",
    "CrossValidationResult",
    "FoldPostprocessingEvaluation",
    "IdentityPrediction",
    "MlflowLoggingError",
    "MultiClassCVResult",
    "MultiClassClassifier",
    "PostprocessedMetrics",
    "TrainingReportData",
    "enabled_stage_configs",
    "evaluate_group_with_postprocessing",
    "generate_markdown_report",
    "log_cross_validation_to_mlflow",
    "mlflow_available",
    "parse_kv_tags",
    "predict_identity",
    "run_leave_one_group_out_cv",
    "save_training_report",
]
