"""Module for defining enums used in JABS"""

from .classifier_types import ClassifierType
from .cv_grouping import DEFAULT_CV_GROUPING_STRATEGY, CrossValidationGroupingStrategy
from .prediction_type import PredictionType
from .storage_format import StorageFormat
from .units import ProjectDistanceUnit

__all__ = [
    "DEFAULT_CV_GROUPING_STRATEGY",
    "ClassifierType",
    "CrossValidationGroupingStrategy",
    "PredictionType",
    "ProjectDistanceUnit",
    "StorageFormat",
]
