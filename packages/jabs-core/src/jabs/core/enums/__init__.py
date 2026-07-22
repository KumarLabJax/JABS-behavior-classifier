"""Module for defining enums used in JABS"""

from .cache_format import CacheFormat
from .classifier_mode import DEFAULT_CLASSIFIER_MODE, ClassifierMode
from .classifier_types import ClassifierType
from .cv_grouping import (
    DEFAULT_CV_GROUPING_STRATEGY,
    CrossValidationGroupingStrategy,
    compile_grouping_regex,
    filename_group_key,
)
from .inference import ConfidenceMetric, Method, SamplingStrategy
from .pose_version import JabsPoseVersion
from .prediction_type import PredictionType
from .storage_format import StorageFormat
from .units import ProjectDistanceUnit

__all__ = [
    "DEFAULT_CLASSIFIER_MODE",
    "DEFAULT_CV_GROUPING_STRATEGY",
    "CacheFormat",
    "ClassifierMode",
    "ClassifierType",
    "ConfidenceMetric",
    "CrossValidationGroupingStrategy",
    "JabsPoseVersion",
    "Method",
    "PredictionType",
    "ProjectDistanceUnit",
    "SamplingStrategy",
    "StorageFormat",
    "compile_grouping_regex",
    "filename_group_key",
]
