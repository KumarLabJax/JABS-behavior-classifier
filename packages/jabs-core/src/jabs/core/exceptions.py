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
