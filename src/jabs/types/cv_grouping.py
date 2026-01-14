from enum import Enum


class CrossValidationGroupingStrategy(str, Enum):
    """Cross-validation grouping type for the project.

    Inheriting from str allows for easy serialization to/from JSON (the enum will
    automatically be serialized using the enum value).
    """

    INDIVIDUAL = "Individual Animal"
    VIDEO = "Video"


DEFAULT_CV_GROUPING_STRATEGY = CrossValidationGroupingStrategy.INDIVIDUAL
