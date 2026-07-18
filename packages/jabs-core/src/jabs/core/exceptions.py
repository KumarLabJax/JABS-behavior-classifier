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


class EmbeddingProvenanceException(Exception):
    """Raised when a feature cache's embedding-sidecar provenance does not match.

    Feature-group membership is otherwise a pure function of the pose file (covered
    by ``pose_hash``). The embedding group's membership additionally depends on the
    external sidecar's presence/content, which ``pose_hash`` does not capture; this
    signals that the cached feature set was built under different sidecar state and
    must be recomputed.
    """

    pass
