"""
The `jabs.classifier` package provides tools for training, evaluating, saving, and loading machine learning classifiers for behavioral data analysis.

It includes the `Classifier` class, which supports multiple classification algorithms (such as Random Forest,
Gradient Boosting, and XGBoost), utilities for feature management, data splitting, model evaluation, and serialization.`
"""

import pathlib

from .classifier import Classifier

HYPERPARAMETER_PATH = pathlib.Path(__file__).parent / "hyperparameters.json"

__all__ = [
    "Classifier",
]
