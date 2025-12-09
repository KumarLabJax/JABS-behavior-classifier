from enum import Enum


class ClassifierType(str, Enum):
    """Classifier type for the project."""

    RANDOM_FOREST = "Random Forest"
    XGBOOST = "XGBoost"
