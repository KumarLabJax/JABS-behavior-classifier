"""HDF5 feature cache reader and writer."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from jabs.core.constants import COMPRESSION, COMPRESSION_OPTS_DEFAULT
from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
from jabs.io.feature_cache.base import FeatureCacheReader, FeatureCacheWriter

logger = logging.getLogger(__name__)


class HDF5FeatureCacheReader(FeatureCacheReader):
    """Reads per-frame and window features from an HDF5 feature cache file.

    The cache is stored as a single ``features.h5`` file in the identity
    directory. Metadata (version, pose hash, distance scale) is stored in
    HDF5 file-level attributes; cached window sizes are inferred by scanning
    group names rather than being stored explicitly.
    """

    _FILENAME = "features.h5"

    def _read_attrs(self, f: h5py.File) -> FeatureCacheMetadata:
        """Build a FeatureCacheMetadata from HDF5 file attributes.

        Args:
            f: Open HDF5 file in read mode.

        Returns:
            Metadata populated from file attributes and group structure.
        """
        scale = f.attrs.get("distance_scale_factor", None)
        cached_window_sizes: frozenset[int] = frozenset()
        if "features" in f:
            cached_window_sizes = frozenset(
                int(key[len("window_features_") :])
                for key in f["features"]
                if key.startswith("window_features_")
            )
        return FeatureCacheMetadata(
            feature_version=int(f.attrs["version"]),
            identity=int(f.attrs["identity"]),
            num_frames=int(f.attrs["num_frames"]),
            pose_hash=str(f.attrs["pose_hash"]),
            distance_scale_factor=float(scale) if scale is not None else None,
            avg_wall_length=(float(f["avg_wall_length"][...]) if "avg_wall_length" in f else None),
            cached_window_sizes=cached_window_sizes,
        )

    def read_metadata(self, identity_dir: Path) -> FeatureCacheMetadata:
        """Read and return validated cache metadata.

        Args:
            identity_dir: Directory for this identity's cache.

        Raises:
            FeatureVersionException: ``feature_version`` mismatch.
            PoseHashException: ``pose_hash`` mismatch.
            DistanceScaleException: ``distance_scale_factor`` mismatch.
            OSError: If ``features.h5`` cannot be opened.
        """
        path = identity_dir / self._FILENAME
        logger.debug("Reading HDF5 cache metadata from %s", path)
        with h5py.File(path, "r") as f:
            metadata = self._read_attrs(f)
        self._validate(metadata)
        return metadata

    def read_per_frame(self, identity_dir: Path) -> PerFrameCacheData:
        """Read per-frame features and auxiliary arrays.

        Args:
            identity_dir: Directory for this identity's cache.

        Raises:
            FeatureVersionException: ``feature_version`` mismatch.
            PoseHashException: ``pose_hash`` mismatch.
            DistanceScaleException: ``distance_scale_factor`` mismatch.
            OSError: If ``features.h5`` cannot be opened.
        """
        path = identity_dir / self._FILENAME
        logger.info("Loading HDF5 per-frame feature cache from %s", path)

        with h5py.File(path, "r") as f:
            metadata = self._read_attrs(f)
            self._validate(metadata)

            frame_valid: npt.NDArray[np.uint8] = f["frame_valid"][:]
            assert len(frame_valid) == metadata.num_frames

            closest_identities: npt.NDArray[np.int64] | None = None
            closest_fov_identities: npt.NDArray[np.int64] | None = None
            if "closest_identities" in f:
                closest_identities = f["closest_identities"][:]
                closest_fov_identities = f["closest_fov_identities"][:]

            closest_corners: npt.NDArray[np.float64] | None = None
            if "closest_corners" in f:
                closest_corners = f["closest_corners"][:]

            wall_distances: dict[str, npt.NDArray[np.float64]] = {}
            if "wall_distances" in f:
                wall_distances = {key: f["wall_distances"][key][:] for key in f["wall_distances"]}

            closest_lixit: npt.NDArray[np.uint8] | None = None
            if "closest_lixit" in f:
                closest_lixit = f["closest_lixit"][:]

            features: dict[str, npt.NDArray[np.generic]] = {}
            for feature_key in f["features/per_frame"]:
                values = f[f"features/per_frame/{feature_key}"][:]
                assert len(values) == metadata.num_frames
                features[feature_key] = values

        logger.debug("Loaded %d per-frame feature columns from %s", len(features), path)
        return PerFrameCacheData(
            frame_valid=frame_valid,
            features=features,
            closest_identities=closest_identities,
            closest_fov_identities=closest_fov_identities,
            closest_corners=closest_corners,
            closest_lixit=closest_lixit,
            wall_distances=wall_distances,
        )

    def read_window(
        self, identity_dir: Path, window_size: int
    ) -> dict[str, npt.NDArray[np.generic]]:
        """Read window features for a specific window size.

        Args:
            identity_dir: Directory for this identity's cache.
            window_size: Window size to load.

        Raises:
            AttributeError: If ``window_size`` is not in ``cached_window_sizes``.
            FeatureVersionException: ``feature_version`` mismatch.
            PoseHashException: ``pose_hash`` mismatch.
            DistanceScaleException: ``distance_scale_factor`` mismatch.
            OSError: If ``features.h5`` cannot be opened.
        """
        path = identity_dir / self._FILENAME
        logger.info("Loading HDF5 window-%d feature cache from %s", window_size, path)

        with h5py.File(path, "r") as f:
            metadata = self._read_attrs(f)
            self._validate(metadata)

            if window_size not in metadata.cached_window_sizes:
                raise AttributeError(f"Window size {window_size} not found in cache at {path}")

            group_key = f"features/window_features_{window_size}"
            window_features: dict[str, npt.NDArray[np.generic]] = {}
            for feature_key in f[group_key]:
                values = f[f"{group_key}/{feature_key}"][:]
                assert len(values) == metadata.num_frames
                window_features[feature_key] = values

        logger.debug(
            "Loaded %d window-%d feature columns from %s",
            len(window_features),
            window_size,
            path,
        )
        return window_features


class HDF5FeatureCacheWriter(FeatureCacheWriter):
    """Writes per-frame and window features to an HDF5 feature cache file.

    Per-frame features and auxiliary arrays are written to a new
    ``features.h5`` file. Window features are appended to the same file
    in separate groups named ``features/window_features_{size}``.

    Compression uses gzip at ``COMPRESSION_OPTS_DEFAULT`` level
    (from ``jabs.core.constants``).
    """

    _FILENAME = "features.h5"

    def _write_attrs(self, f: h5py.File, metadata: FeatureCacheMetadata) -> None:
        """Write versioning and validation attributes to an HDF5 file.

        Args:
            f: Open HDF5 file in write or append mode.
            metadata: Metadata whose fields are written as file attributes.
        """
        f.attrs["num_frames"] = metadata.num_frames
        f.attrs["identity"] = metadata.identity
        f.attrs["version"] = metadata.feature_version
        f.attrs["pose_hash"] = metadata.pose_hash
        if metadata.distance_scale_factor is not None:
            f.attrs["distance_scale_factor"] = metadata.distance_scale_factor

    def write_per_frame(
        self,
        identity_dir: Path,
        metadata: FeatureCacheMetadata,
        data: PerFrameCacheData,
    ) -> None:
        """Write per-frame features and auxiliary arrays to ``features.h5``.

        Creates ``identity_dir`` and any missing parents. Overwrites any
        existing ``features.h5``.

        Args:
            identity_dir: Directory for this identity's cache.
            metadata: Versioning and validation metadata to persist.
            data: Per-frame feature arrays and auxiliary arrays to write.
        """
        identity_dir.mkdir(mode=0o775, exist_ok=True, parents=True)
        path = identity_dir / self._FILENAME
        logger.info("Writing HDF5 per-frame feature cache to %s", path)

        with h5py.File(path, "w") as f:
            self._write_attrs(f, metadata)

            f.create_dataset(
                "frame_valid",
                data=data.frame_valid,
                compression=COMPRESSION,
                compression_opts=COMPRESSION_OPTS_DEFAULT,
            )

            if (data.closest_identities is None) != (data.closest_fov_identities is None):
                raise ValueError(
                    "closest_identities and closest_fov_identities must both be provided "
                    "or both be None; got one without the other"
                )
            if data.closest_identities is not None:
                f.create_dataset(
                    "closest_identities",
                    data=data.closest_identities,
                    compression=COMPRESSION,
                    compression_opts=COMPRESSION_OPTS_DEFAULT,
                )
                f.create_dataset(
                    "closest_fov_identities",
                    data=data.closest_fov_identities,
                    compression=COMPRESSION,
                    compression_opts=COMPRESSION_OPTS_DEFAULT,
                )

            if data.closest_corners is not None:
                if metadata.avg_wall_length is None:
                    raise ValueError(
                        "metadata.avg_wall_length must be set when data.closest_corners is present"
                    )
                f.create_dataset(
                    "closest_corners",
                    data=data.closest_corners,
                    compression=COMPRESSION,
                    compression_opts=COMPRESSION_OPTS_DEFAULT,
                )
                f.create_dataset("avg_wall_length", data=metadata.avg_wall_length)
                wall_grp = f.require_group("wall_distances")
                for direction, values in data.wall_distances.items():
                    wall_grp.create_dataset(
                        direction,
                        data=values,
                        compression=COMPRESSION,
                        compression_opts=COMPRESSION_OPTS_DEFAULT,
                    )

            if data.closest_lixit is not None:
                f.create_dataset(
                    "closest_lixit",
                    data=data.closest_lixit,
                    compression=COMPRESSION,
                    compression_opts=COMPRESSION_OPTS_DEFAULT,
                )

            per_frame_grp = f.require_group("features/per_frame")
            for feature_key, values in data.features.items():
                per_frame_grp.create_dataset(
                    feature_key,
                    data=values,
                    compression=COMPRESSION,
                    compression_opts=COMPRESSION_OPTS_DEFAULT,
                )

        logger.debug("Wrote %d per-frame feature columns to %s", len(data.features), path)

    def write_window(
        self,
        identity_dir: Path,
        metadata: FeatureCacheMetadata,
        window_size: int,
        data: dict[str, npt.NDArray[np.generic]],
    ) -> None:
        """Append window features for one window size to ``features.h5``.

        For HDF5 caches, ``cached_window_sizes`` is not stored explicitly —
        it is inferred from group names on read. The ``metadata`` parameter
        is used only to refresh file-level attributes.

        Args:
            identity_dir: Directory for this identity's cache.
            metadata: Current cache metadata; attributes are refreshed on disk.
            window_size: Window size these features were computed for.
            data: Flat dict mapping ``"module_name window_op feature_name"``
                to shape-``(n_frames,)`` arrays.
        """
        path = identity_dir / self._FILENAME
        logger.info("Writing HDF5 window-%d feature cache to %s", window_size, path)

        with h5py.File(path, "a") as f:
            self._write_attrs(f, metadata)
            group_key = f"features/window_features_{window_size}"
            if group_key in f:
                del f[group_key]
            window_grp = f.create_group(group_key)
            for feature_key, values in data.items():
                window_grp.create_dataset(
                    feature_key,
                    data=values,
                    compression=COMPRESSION,
                    compression_opts=COMPRESSION_OPTS_DEFAULT,
                )

        logger.debug(
            "Wrote %d window-%d feature columns to %s",
            len(data),
            window_size,
            path,
        )
