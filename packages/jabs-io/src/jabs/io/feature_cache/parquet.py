"""Parquet feature cache reader and writer."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from jabs.core.exceptions import FeatureVersionException
from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
from jabs.io.feature_cache.base import FeatureCacheReader, FeatureCacheWriter

logger = logging.getLogger(__name__)

# Increment when the Parquet cache layout changes in a backward-incompatible way.
PARQUET_FORMAT_VERSION: int = 1

# Column name constants for auxiliary (non-feature) columns in per_frame.parquet.
_COL_FRAME_VALID = "_jabs_frame_valid"
_COL_CLOSEST_IDENTITIES = "_jabs_closest_identities"
_COL_CLOSEST_FOV_IDENTITIES = "_jabs_closest_fov_identities"
_COL_CLOSEST_CORNERS = "_jabs_closest_corners"
_COL_CLOSEST_LIXIT = "_jabs_closest_lixit"
_COL_WALL_PREFIX = "_jabs_wall_"

_METADATA_FILENAME = "metadata.json"
_PER_FRAME_FILENAME = "per_frame.parquet"
_WINDOW_FILENAME_TEMPLATE = "window_{size}.parquet"


def _metadata_to_dict(metadata: FeatureCacheMetadata) -> dict[str, object]:
    """Serialize ``FeatureCacheMetadata`` to a JSON-compatible dict.

    ``cached_window_sizes`` is converted to a sorted list for a stable,
    human-readable representation. ``format_version`` is always
    ``PARQUET_FORMAT_VERSION``.

    Args:
        metadata: Metadata to serialize.

    Returns:
        Dict suitable for writing to ``metadata.json``.
    """
    return {
        "feature_version": metadata.feature_version,
        "format_version": PARQUET_FORMAT_VERSION,
        "identity": metadata.identity,
        "num_frames": metadata.num_frames,
        "pose_hash": metadata.pose_hash,
        "distance_scale_factor": metadata.distance_scale_factor,
        "avg_wall_length": metadata.avg_wall_length,
        "cached_window_sizes": sorted(metadata.cached_window_sizes),
    }


def _write_metadata_json(identity_dir: Path, metadata: FeatureCacheMetadata) -> None:
    """Write ``metadata.json`` into ``identity_dir``.

    Args:
        identity_dir: Directory for this identity's cache.
        metadata: Metadata to serialize and write.
    """
    path = identity_dir / _METADATA_FILENAME
    path.write_text(json.dumps(_metadata_to_dict(metadata), indent=2))


class ParquetFeatureCacheWriter(FeatureCacheWriter):
    """Writes per-frame and window features to Parquet files with LZ4 compression.

    Per-frame features and auxiliary arrays are written to ``per_frame.parquet``.
    Window features for each window size are written to
    ``window_{size}.parquet``. Cache metadata (versions, hashes, window size
    bookkeeping) is stored in ``metadata.json``.

    Write ordering guarantees:
    - ``write_per_frame``: Parquet file first, then ``metadata.json``. An
      incomplete write therefore leaves no sentinel file and the partial cache
      is ignored by ``detect_cache_format()``.
    - ``write_window``: Parquet file first, then ``metadata.json`` is updated
      with the new window size. A crash between the two steps leaves an
      unregistered window file that is simply ignored on the next read.
    """

    def write_per_frame(
        self,
        identity_dir: Path,
        metadata: FeatureCacheMetadata,
        data: PerFrameCacheData,
    ) -> None:
        """Write per-frame features and auxiliary arrays to ``per_frame.parquet``.

        Creates ``identity_dir`` and any missing parents. Writes
        ``per_frame.parquet`` first, then ``metadata.json`` with an empty
        ``cached_window_sizes`` list so that an incomplete write leaves no
        sentinel file.

        Column ordering: ``_jabs_frame_valid`` first, then other ``_jabs_*``
        auxiliary columns, then feature columns sorted alphabetically.

        Args:
            identity_dir: Directory for this identity's cache.
            metadata: Versioning and validation metadata to persist in
                ``metadata.json``.
            data: Per-frame feature arrays and auxiliary arrays to write.

        Raises:
            ValueError: If exactly one of ``closest_identities`` /
                ``closest_fov_identities`` is provided, or if
                ``closest_corners`` is present but ``metadata.avg_wall_length``
                is ``None``.
        """
        if (data.closest_identities is None) != (data.closest_fov_identities is None):
            raise ValueError(
                "closest_identities and closest_fov_identities must both be provided "
                "or both be None; got one without the other"
            )
        if data.closest_corners is not None and metadata.avg_wall_length is None:
            raise ValueError(
                "metadata.avg_wall_length must be set when data.closest_corners is present"
            )

        identity_dir.mkdir(mode=0o775, exist_ok=True, parents=True)
        parquet_path = identity_dir / _PER_FRAME_FILENAME
        logger.info("Writing Parquet per-frame feature cache to %s", parquet_path)

        # Build columns in the specified order first _jabs_frame_valid followed by other _jabs_*,
        columns: dict[str, npt.NDArray[np.generic]] = {}
        columns[_COL_FRAME_VALID] = data.frame_valid
        if data.closest_identities is not None:
            columns[_COL_CLOSEST_IDENTITIES] = data.closest_identities
            columns[_COL_CLOSEST_FOV_IDENTITIES] = data.closest_fov_identities
        if data.closest_corners is not None:
            columns[_COL_CLOSEST_CORNERS] = data.closest_corners
        if data.closest_lixit is not None:
            columns[_COL_CLOSEST_LIXIT] = data.closest_lixit
        for direction, values in sorted(data.wall_distances.items()):
            columns[f"{_COL_WALL_PREFIX}{direction}"] = values

        # then add  feature columns sorted alphabetically.
        for feature_key in sorted(data.features):
            columns[feature_key] = data.features[feature_key]

        table = pa.table({k: pa.array(v) for k, v in columns.items()})
        pq.write_table(table, parquet_path, compression="lz4")
        logger.debug("Wrote %d per-frame feature columns to %s", len(data.features), parquet_path)

        # metadata.json is written last; its presence is the sentinel for a valid cache.
        initial_metadata = replace(metadata, cached_window_sizes=frozenset())
        _write_metadata_json(identity_dir, initial_metadata)
        logger.debug("Wrote metadata.json for identity %d", metadata.identity)

    def write_window(
        self,
        identity_dir: Path,
        metadata: FeatureCacheMetadata,
        window_size: int,
        data: dict[str, npt.NDArray[np.generic]],
    ) -> None:
        """Write window features for one window size to ``window_{size}.parquet``.

        Writes the Parquet file first, then reads ``metadata.json``, appends
        ``window_size`` to ``cached_window_sizes``, and writes ``metadata.json``
        back. Columns are sorted alphabetically for a stable output layout.

        Args:
            identity_dir: Directory for this identity's cache.
            metadata: Current cache metadata. The ``cached_window_sizes`` field
                is ignored; ``metadata.json`` is read from disk to obtain the
                accumulated set of previously written window sizes, because
                callers construct metadata fresh for each call without carrying
                that set forward.
            window_size: Window size these features were computed for.
            data: Flat dict mapping ``"module_name window_op feature_name"``
                to shape-``(n_frames,)`` arrays.
        """
        parquet_path = identity_dir / _WINDOW_FILENAME_TEMPLATE.format(size=window_size)
        logger.info("Writing Parquet window-%d feature cache to %s", window_size, parquet_path)

        table = pa.table({k: pa.array(data[k]) for k in sorted(data)})
        pq.write_table(table, parquet_path, compression="lz4")
        logger.debug(
            "Wrote %d window-%d feature columns to %s",
            len(data),
            window_size,
            parquet_path,
        )

        # Read cached_window_sizes from disk rather than from the caller-supplied
        # metadata, because callers construct metadata fresh for each call and do
        # not carry forward the accumulated set of previously written window sizes.
        metadata_path = identity_dir / _METADATA_FILENAME
        with metadata_path.open() as f:
            raw: dict[str, object] = json.load(f)
        existing_sizes: frozenset[int] = frozenset(int(s) for s in raw["cached_window_sizes"])  # type: ignore[arg-type]
        updated_metadata = replace(metadata, cached_window_sizes=existing_sizes | {window_size})
        _write_metadata_json(identity_dir, updated_metadata)
        logger.debug(
            "Updated metadata.json cached_window_sizes for identity %d: %s",
            metadata.identity,
            sorted(updated_metadata.cached_window_sizes),
        )


class ParquetFeatureCacheReader(FeatureCacheReader):
    """Reads per-frame and window features from Parquet feature cache files.

    The cache consists of ``per_frame.parquet``, zero or more
    ``window_{size}.parquet`` files, and ``metadata.json`` in the identity
    directory. ``metadata.json`` is the sentinel whose presence indicates a
    complete cache.

    Validation order on every read: ``format_version`` is checked first
    (raises ``FeatureVersionException`` if it does not match
    ``PARQUET_FORMAT_VERSION``), then the base-class checks for
    ``feature_version``, ``pose_hash``, and ``distance_scale_factor``.

    Error recovery:
    - ``per_frame.parquet`` unreadable → all cache files are deleted and
      ``OSError`` is raised so the caller falls back to recomputing.
    - ``window_{size}.parquet`` unreadable → ``window_size`` is removed from
      ``cached_window_sizes`` in ``metadata.json`` and ``AttributeError`` is
      raised so only that window size is recomputed.
    """

    @staticmethod
    def _read_raw_metadata(identity_dir: Path) -> dict[str, object]:
        """Read and parse ``metadata.json``.

        Args:
            identity_dir: Directory for this identity's cache.

        Returns:
            Parsed JSON content as a plain dict.

        Raises:
            OSError: If ``metadata.json`` cannot be opened or contains invalid JSON.
        """
        path = identity_dir / _METADATA_FILENAME
        try:
            with path.open() as f:
                return json.load(f)  # type: ignore[return-value]
        except ValueError as exc:
            # json.JSONDecodeError is a subclass of ValueError; normalize to OSError
            # so callers treat malformed metadata the same as a missing file.
            raise OSError(f"Invalid JSON in cache metadata at {path}") from exc

    @staticmethod
    def _metadata_from_dict(raw: dict[str, object]) -> FeatureCacheMetadata:
        """Construct ``FeatureCacheMetadata`` from a parsed ``metadata.json`` dict.

        Args:
            raw: Dict as returned by ``_read_raw_metadata``.

        Returns:
            Populated ``FeatureCacheMetadata`` instance.

        Raises:
            FeatureVersionException: A required key is missing or has an
                incompatible type, indicating a schema change or corrupt file.
        """
        try:
            scale = raw["distance_scale_factor"]
            wall = raw["avg_wall_length"]
            return FeatureCacheMetadata(
                feature_version=int(raw["feature_version"]),  # type: ignore[arg-type]
                identity=int(raw["identity"]),  # type: ignore[arg-type]
                num_frames=int(raw["num_frames"]),  # type: ignore[arg-type]
                pose_hash=str(raw["pose_hash"]),
                distance_scale_factor=float(scale) if scale is not None else None,  # type: ignore[arg-type]
                avg_wall_length=float(wall) if wall is not None else None,  # type: ignore[arg-type]
                cached_window_sizes=frozenset(
                    int(s)
                    for s in raw["cached_window_sizes"]  # type: ignore[union-attr]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Failed to parse cache metadata: %s", exc)
            raise FeatureVersionException from exc

    @staticmethod
    def _check_format_version(raw: dict[str, object]) -> None:
        """Raise ``FeatureVersionException`` if ``format_version`` is unexpected.

        Args:
            raw: Dict as returned by ``_read_raw_metadata``.

        Raises:
            FeatureVersionException: ``format_version`` is missing, has an
                incompatible type, or does not equal ``PARQUET_FORMAT_VERSION``.
        """
        try:
            stored = int(raw["format_version"])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Missing or invalid format_version in cache metadata: %s", exc)
            raise FeatureVersionException from exc
        if stored != PARQUET_FORMAT_VERSION:
            logger.debug(
                "Parquet format version mismatch: expected %d, got %d",
                PARQUET_FORMAT_VERSION,
                stored,
            )
            raise FeatureVersionException

    @staticmethod
    def _delete_all_cache_files(identity_dir: Path) -> None:
        """Delete all Parquet files and ``metadata.json`` from ``identity_dir``.

        Called when ``per_frame.parquet`` is unreadable so the caller can
        safely recompute from scratch.

        Args:
            identity_dir: Directory for this identity's cache.
        """
        for path in identity_dir.glob("*.parquet"):
            path.unlink(missing_ok=True)

        (identity_dir / _METADATA_FILENAME).unlink(missing_ok=True)
        logger.debug("Deleted all Parquet cache files in %s", identity_dir)

    def read_metadata(self, identity_dir: Path) -> FeatureCacheMetadata:
        """Read and return validated cache metadata from ``metadata.json``.

        Checks ``format_version`` before delegating to the base-class
        ``_validate`` for the remaining fields.

        Args:
            identity_dir: Directory for this identity's cache.

        Raises:
            FeatureVersionException: ``feature_version`` or ``format_version``
                mismatch.
            PoseHashException: ``pose_hash`` mismatch.
            DistanceScaleException: ``distance_scale_factor`` mismatch.
            OSError: If ``metadata.json`` cannot be read.
        """
        logger.debug("Reading Parquet cache metadata from %s", identity_dir)

        raw = self._read_raw_metadata(identity_dir)
        self._check_format_version(raw)

        metadata = self._metadata_from_dict(raw)
        self._validate(metadata)

        return metadata

    def read_per_frame(self, identity_dir: Path) -> PerFrameCacheData:
        """Read per-frame features and auxiliary arrays from ``per_frame.parquet``.

        Validates metadata first. If ``per_frame.parquet`` is missing or
        unreadable, all cache files are deleted and ``OSError`` is raised so
        the caller falls back to recomputing all features.

        ``_jabs_*`` columns are mapped to their corresponding ``PerFrameCacheData``
        fields; all remaining columns are treated as feature columns.

        Args:
            identity_dir: Directory for this identity's cache.

        Raises:
            FeatureVersionException: ``feature_version`` or ``format_version``
                mismatch.
            PoseHashException: ``pose_hash`` mismatch.
            DistanceScaleException: ``distance_scale_factor`` mismatch.
            OSError: If ``per_frame.parquet`` is missing or unreadable (cache
                files are deleted before raising).
        """
        self.read_metadata(identity_dir)  # validate before touching data files
        parquet_path = identity_dir / _PER_FRAME_FILENAME
        logger.info("Loading Parquet per-frame feature cache from %s", parquet_path)

        try:
            table = pq.read_table(parquet_path)
        except Exception as exc:
            logger.warning(
                "Failed to read per-frame Parquet cache %s; deleting all cache files: %s",
                parquet_path,
                exc,
                exc_info=True,
            )
            self._delete_all_cache_files(identity_dir)
            raise OSError(f"Unreadable per-frame cache at {parquet_path}") from exc

        col_names: set[str] = set(table.schema.names)

        frame_valid: npt.NDArray[np.uint8] = (
            table.column(_COL_FRAME_VALID).to_numpy(zero_copy_only=False).astype(np.uint8)
        )

        closest_identities: npt.NDArray[np.int64] | None = None
        closest_fov_identities: npt.NDArray[np.int64] | None = None
        if _COL_CLOSEST_IDENTITIES in col_names:
            closest_identities = (
                table.column(_COL_CLOSEST_IDENTITIES)
                .to_numpy(zero_copy_only=False)
                .astype(np.int64)
            )
            closest_fov_identities = (
                table.column(_COL_CLOSEST_FOV_IDENTITIES)
                .to_numpy(zero_copy_only=False)
                .astype(np.int64)
            )

        closest_corners: npt.NDArray[np.float64] | None = None
        if _COL_CLOSEST_CORNERS in col_names:
            closest_corners = (
                table.column(_COL_CLOSEST_CORNERS)
                .to_numpy(zero_copy_only=False)
                .astype(np.float64)
            )

        closest_lixit: npt.NDArray[np.float64] | None = None
        if _COL_CLOSEST_LIXIT in col_names:
            closest_lixit = (
                table.column(_COL_CLOSEST_LIXIT).to_numpy(zero_copy_only=False).astype(np.float64)
            )

        wall_distances: dict[str, npt.NDArray[np.float64]] = {}
        for col_name in col_names:
            if col_name.startswith(_COL_WALL_PREFIX):
                direction = col_name[len(_COL_WALL_PREFIX) :]
                wall_distances[direction] = (
                    table.column(col_name).to_numpy(zero_copy_only=False).astype(np.float64)
                )

        _jabs_cols: set[str] = {
            _COL_FRAME_VALID,
            _COL_CLOSEST_IDENTITIES,
            _COL_CLOSEST_FOV_IDENTITIES,
            _COL_CLOSEST_CORNERS,
            _COL_CLOSEST_LIXIT,
        } | {c for c in col_names if c.startswith(_COL_WALL_PREFIX)}

        features: dict[str, npt.NDArray[np.generic]] = {}
        for col_name in table.schema.names:
            if col_name not in _jabs_cols:
                features[col_name] = table.column(col_name).to_numpy(zero_copy_only=False)

        logger.debug("Loaded %d per-frame feature columns from %s", len(features), parquet_path)
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

        If ``window_size`` is not recorded in ``cached_window_sizes``,
        ``AttributeError`` is raised immediately. If the Parquet file exists
        in the metadata but is unreadable, ``window_size`` is removed from
        ``cached_window_sizes`` in ``metadata.json`` and ``AttributeError`` is
        raised so only that window size is recomputed.

        Args:
            identity_dir: Directory for this identity's cache.
            window_size: Window size to load.

        Raises:
            AttributeError: If ``window_size`` is not cached, or if the
                corresponding Parquet file is unreadable (size is removed from
                ``metadata.json`` before raising).
            FeatureVersionException: ``feature_version`` or ``format_version``
                mismatch.
            PoseHashException: ``pose_hash`` mismatch.
            DistanceScaleException: ``distance_scale_factor`` mismatch.
            OSError: If ``metadata.json`` cannot be read.
        """
        metadata = self.read_metadata(identity_dir)

        if window_size not in metadata.cached_window_sizes:
            raise AttributeError(
                f"Window size {window_size} not found in Parquet cache at {identity_dir}"
            )

        parquet_path = identity_dir / _WINDOW_FILENAME_TEMPLATE.format(size=window_size)
        logger.info("Loading Parquet window-%d feature cache from %s", window_size, parquet_path)

        try:
            table = pq.read_table(parquet_path)
        except Exception as exc:
            logger.warning(
                "Failed to read window-%d Parquet cache %s; removing from cached_window_sizes: %s",
                window_size,
                parquet_path,
                exc,
                exc_info=True,
            )
            updated_metadata = replace(
                metadata,
                cached_window_sizes=metadata.cached_window_sizes - {window_size},
            )
            _write_metadata_json(identity_dir, updated_metadata)
            raise AttributeError(
                f"Unreadable window-{window_size} cache at {parquet_path}"
            ) from exc

        window_features: dict[str, npt.NDArray[np.generic]] = {}
        for col_name in table.schema.names:
            window_features[col_name] = table.column(col_name).to_numpy(zero_copy_only=False)

        logger.debug(
            "Loaded %d window-%d feature columns from %s",
            len(window_features),
            window_size,
            parquet_path,
        )
        return window_features
