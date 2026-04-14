"""Module for classifier mode enum."""

from enum import Enum


class ClassifierMode(str, Enum):
    """Classifier mode for the project.

    Inheriting from str allows for easy serialization to/from JSON (the enum
    will automatically be serialized using the enum value).
    """

    BINARY = "binary"
    MULTICLASS = "multiclass"


DEFAULT_CLASSIFIER_MODE = ClassifierMode.BINARY
