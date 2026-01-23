"""
The `jabs.classifier` package provides tools for training, evaluating, saving, and loading machine learning classifiers for behavioral data analysis.

It includes the `Classifier` class, which supports multiple classification algorithms (such as Random Forest,
Gradient Boosting, and XGBoost), utilities for feature management, data splitting, model evaluation, and serialization.`
"""

from .classifier import Classifier
from .cross_validation import run_leave_one_group_out_cv
from .training_report import (
    CrossValidationResult,
    TrainingReportData,
    generate_markdown_report,
    save_training_report,
)

__all__ = [
    "Classifier",
    "CrossValidationResult",
    "TrainingReportData",
    "generate_markdown_report",
    "run_leave_one_group_out_cv",
    "save_training_report",
]
