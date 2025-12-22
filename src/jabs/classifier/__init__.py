"""
The `jabs.classifier` package provides tools for training, evaluating, saving, and loading machine learning classifiers for behavioral data analysis.

It includes the `Classifier` class, which supports multiple classification algorithms (such as Random Forest,
Gradient Boosting, and XGBoost), utilities for feature management, data splitting, model evaluation, and serialization.`
"""

from .classifier import Classifier

__all__ = [
    "Classifier",
]
