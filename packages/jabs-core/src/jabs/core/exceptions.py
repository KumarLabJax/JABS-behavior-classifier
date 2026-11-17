class PoseHashException(Exception):
    """Exception raised when the hash of a pose file does not match the expected value."""

    pass


class PoseIdEmbeddingException(Exception):
    """Exception raised for invalid instance_embed_id values in pose file."""

    pass


class MissingBehaviorError(Exception):
    """Exception raised when a behavior is not found in the prediction file."""

    pass


class FeatureVersionException(Exception):
    """exception raised when the version of the features in the h5 file is not compatible with the current version of JABS"""

    pass


class DistanceScaleException(Exception):
    """exception raised when the distance scale factor in the h5 file don't match what the classifier expects"""

    pass


class MissingOptionalDependencyError(ImportError):
    """Exception raised when a feature needs an optional dependency that is not installed.

    Subclasses ``ImportError``, so existing ``except ImportError`` handlers still catch it.
    The message is written for whoever installed the raising package; an application
    that bundles that package can catch this and use ``extra`` to give install advice
    that fits how the application itself was installed.

    Attributes:
        extra: Name of the optional extra that provides the missing dependency.
    """

    def __init__(self, message: str, *, extra: str) -> None:
        super().__init__(message)
        self.extra = extra
