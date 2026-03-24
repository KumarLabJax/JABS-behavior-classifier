"""Abstract base classes for feature cache readers and writers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt

from jabs.core.exceptions import DistanceScaleException, FeatureVersionException
from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
from jabs.pose_estimation import PoseHashException

logger = logging.getLogger(__name__)


class FeatureCacheReader(ABC):
    """Abstract reader for a single identity's feature cache directory.

    Validation parameters are supplied at construction so a single reader
    instance can serve repeated reads for one identity session without
    re-passing them on every call.

    Args:
        expected_feature_version: Value of ``FEATURE_VERSION`` from
            ``features.py``; used to detect stale caches.
        expected_pose_hash: Hash of the pose file at the time features were
            requested; used to detect caches built from a different pose file.
        expected_distance_scale_factor: Pixels-to-cm scale factor, or ``None``
            when not using cm units.
    """

    def __init__(
        self,
        expected_feature_version: int,
        expected_pose_hash: str,
        expected_distance_scale_factor: float | None,
    ) -> None:
        self._expected_feature_version = expected_feature_version
        self._expected_pose_hash = expected_pose_hash
        self._expected_distance_scale_factor = expected_distance_scale_factor

    def _validate(self, metadata: FeatureCacheMetadata) -> None:
        """Raise if metadata fails any validation check.

        Subclasses may extend validation by calling ``super()._validate(metadata)``
        and then adding format-specific checks (e.g. ``format_version`` for
        the Parquet reader).

        Args:
            metadata: Metadata read from the cache to validate against the
                expected values supplied at construction.

        Raises:
            FeatureVersionException: ``feature_version`` does not match
                ``expected_feature_version``.
            PoseHashException: ``pose_hash`` does not match
                ``expected_pose_hash``.
            DistanceScaleException: ``distance_scale_factor`` does not match
                ``expected_distance_scale_factor``.
        """
        if metadata.feature_version != self._expected_feature_version:
            logger.debug(
                "Feature version mismatch: expected %d, got %d",
                self._expected_feature_version,
                metadata.feature_version,
            )
            raise FeatureVersionException
        if metadata.pose_hash != self._expected_pose_hash:
            logger.debug(
                "Pose hash mismatch: expected %s, got %s",
                self._expected_pose_hash,
                metadata.pose_hash,
            )
            raise PoseHashException
        if metadata.distance_scale_factor != self._expected_distance_scale_factor:
            logger.debug(
                "Distance scale factor mismatch: expected %s, got %s",
                self._expected_distance_scale_factor,
                metadata.distance_scale_factor,
            )
            raise DistanceScaleException

    @abstractmethod
    def read_metadata(self, identity_dir: Path) -> FeatureCacheMetadata:
        """Read and return validated cache metadata.

        Implementations must call ``self._validate(metadata)`` before
        returning.

        Args:
            identity_dir: Directory for this identity's cache.

        Raises:
            FeatureVersionException: ``feature_version`` mismatch.
            PoseHashException: ``pose_hash`` mismatch.
            DistanceScaleException: ``distance_scale_factor`` mismatch.
            OSError: If the cache files cannot be opened.
        """

    @abstractmethod
    def read_per_frame(self, identity_dir: Path) -> PerFrameCacheData:
        """Read per-frame features and auxiliary arrays.

        Args:
            identity_dir: Directory for this identity's cache.

        Raises:
            FeatureVersionException: ``feature_version`` mismatch.
            PoseHashException: ``pose_hash`` mismatch.
            DistanceScaleException: ``distance_scale_factor`` mismatch.
            OSError: If the cache files cannot be opened.
        """

    @abstractmethod
    def read_window(
        self, identity_dir: Path, window_size: int
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Read window features for a specific window size.

        Returns a flat dict mapping ``"module_name window_op feature_name"``
        to shape-``(n_frames,)`` arrays, matching the format produced by
        ``IdentityFeatures.merge_window_features()``.

        Args:
            identity_dir: Directory for this identity's cache.
            window_size: Window size to load.

        Raises:
            AttributeError: If ``window_size`` is not in
                ``cached_window_sizes``.
            OSError: If the cache files cannot be opened.
        """


class FeatureCacheWriter(ABC):
    """Abstract writer for a single identity's feature cache directory."""

    @abstractmethod
    def write_per_frame(
        self,
        identity_dir: Path,
        metadata: FeatureCacheMetadata,
        data: PerFrameCacheData,
    ) -> None:
        """Write per-frame features and auxiliary arrays.

        Creates ``identity_dir`` and any missing parents if they do not exist.

        Args:
            identity_dir: Directory for this identity's cache.
            metadata: Versioning and validation metadata to persist alongside
                the feature data.
            data: Per-frame feature arrays and auxiliary arrays to write.
        """

    @abstractmethod
    def write_window(
        self,
        identity_dir: Path,
        metadata: FeatureCacheMetadata,
        window_size: int,
        data: dict[str, npt.NDArray[np.float64]],
    ) -> None:
        """Write window features for one window size.

        Appends ``window_size`` to the persisted ``cached_window_sizes`` after
        writing the feature data.

        Args:
            identity_dir: Directory for this identity's cache.
            metadata: Current cache metadata; used to update
                ``cached_window_sizes`` on disk.
            window_size: Window size these features were computed for.
            data: Flat dict mapping ``"module_name window_op feature_name"``
                to shape-``(n_frames,)`` arrays.
        """
