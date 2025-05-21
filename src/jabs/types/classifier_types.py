from enum import IntEnum


class ClassifierType(IntEnum):
    """Classifier type for the project."""

    RANDOM_FOREST = 1
    GRADIENT_BOOSTING = 2
    XGBOOST = 3
