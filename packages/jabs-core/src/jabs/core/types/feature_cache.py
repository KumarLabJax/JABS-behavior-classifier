"""Types for feature cache data exchanged between feature extraction and jabs-io."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class FeatureCacheMetadata:
    """Versioning and validation metadata for a single identity's feature cache.

    Stores the fields written to HDF5 file attributes or to ``metadata.json``
    in the Parquet cache layout. Both ``distance_scale_factor`` and
    ``avg_wall_length`` are omitted (left as ``None``) when the corresponding
    features were not computed.

    ``cached_window_sizes`` uses ``frozenset`` to enforce uniqueness and allow
    O(1) membership testing. When serializing to JSON, convert to a sorted list
    for a stable, human-readable representation and reconstruct with
    ``frozenset`` on read.

    Attributes:
        feature_version: Value of ``FEATURE_VERSION`` from ``features.py`` at
            the time the cache was written.
        identity: Identity index this cache entry belongs to.
        num_frames: Total frame count in the source video.
        pose_hash: Hash of the pose file content; used to detect stale caches.
        distance_scale_factor: Pixels-to-cm conversion factor; ``None`` when
            not using cm units.
        avg_wall_length: Mean arena wall length in distance units; ``None``
            when no landmark features were computed.
        cached_window_sizes: Set of window sizes whose features have been
            written to this cache. Updated each time a new window size is
            written by replacing the instance:
            ``replace(meta, cached_window_sizes=meta.cached_window_sizes | {size})``.
    """

    feature_version: int
    identity: int
    num_frames: int
    pose_hash: str
    distance_scale_factor: float | None = None
    avg_wall_length: float | None = None
    cached_window_sizes: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class PerFrameCacheData:
    """Per-frame feature data for a single identity.

    Holds all data that forms the per-frame portion of the cache — the feature
    arrays plus auxiliary arrays produced alongside them. Feature columns use
    the flat, merged key format (``"module_name feature_name"``) that matches
    both the existing HDF5 dataset key format and the future Parquet column
    names, so the I/O layer never needs to merge or split the nested dict.
    That merging responsibility stays in ``IdentityFeatures`` via
    ``merge_per_frame_features()``.

    ``closest_identities``, ``closest_fov_identities``, ``closest_corners``,
    ``closest_lixit``, and ``wall_distances`` are absent (``None`` or empty)
    when the corresponding features were not computed for this identity.

    Attributes:
        frame_valid: Boolean presence mask, shape ``(n_frames,)``, dtype uint8.
        features: Flat dict mapping ``"module_name feature_name"`` to a
            shape-``(n_frames,)`` array. Produced by
            ``IdentityFeatures.merge_per_frame_features()``.
        closest_identities: Nearest-neighbor identity index per frame, shape
            ``(n_frames,)``. Present for pose v3+ (social features),
            ``None`` otherwise.
        closest_fov_identities: Nearest-neighbor identity within the field of
            view per frame, shape ``(n_frames,)``. Same conditions as
            ``closest_identities``.
        closest_corners: Distance to the nearest arena corner per frame, shape
            ``(n_frames,)``. ``None`` when no landmark features were computed.
        closest_lixit: Distance to the nearest lixit per frame, shape
            ``(n_frames,)``. ``None`` when no landmark features were computed.
        wall_distances: Per-wall distance arrays keyed by wall direction (e.g.
            ``"top"``, ``"bottom"``), each shape ``(n_frames,)``. Empty dict
            when no landmark features were computed.
    """

    frame_valid: npt.NDArray[np.uint8]
    features: dict[str, npt.NDArray[np.float64]]
    closest_identities: npt.NDArray[np.int64] | None = None
    closest_fov_identities: npt.NDArray[np.int64] | None = None
    closest_corners: npt.NDArray[np.float64] | None = None
    closest_lixit: npt.NDArray[np.float64] | None = None
    wall_distances: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
