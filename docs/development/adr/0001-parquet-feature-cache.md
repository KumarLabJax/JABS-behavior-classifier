# ADR 0001: Parquet Feature Cache

## Status

Proposed

## Context

Feature cache files are currently stored as compressed HDF5 (gzip level 6). The current I/O path
has two compounding bottlenecks: each of the ~253 per-frame feature columns and ~4529 window
feature columns is stored as its own HDF5 dataset and must be read individually, and each dataset
is gzip-compressed — so workers pay both the per-dataset open/seek overhead and the decompression
cost for every column. During classification of large projects (200+ videos) this leaves CPUs
underutilized waiting on I/O.

Parquet with LZ4 compression provides equivalent or better compression ratios with 3–5× faster
decompression, and pyarrow's columnar reader is a natural fit for the DataFrame-shaped access
pattern already used throughout the feature extraction code. `pyarrow>=20.0.0` is already a
root dependency — no new packages are needed.

## Decision

Migrate the feature cache format from HDF5 to Parquet, initially for GUI-facing workflows, while
preserving HDF5 as the default for HPC CLI tools until such a change can be accommodated by
existing Nextflow workflows. Both formats coexist until possible HDF5 feature cache format
deprecation; the reader auto-detects which format is present.

### Design goals

- **New GUI-facing workflows default to Parquet.** The GUI and `jabs-init` write Parquet for new
  caches. `jabs-init` is the recommended way to pre-compute features for GUI projects, so it
  aligns with the GUI default.
- **Existing HDF5 caches are preserved as-is.** When the GUI opens a project that already has
  `features.h5` files, those are used without conversion. Users must explicitly clear the cache
  (delete the `jabs/features/` directory or pass `--force` to `jabs-init`) to switch an existing
  project to Parquet.
- **HPC CLI tools remain on HDF5.** `jabs-features` and `jabs-classify` continue to write HDF5
  by default, preserving compatibility with existing HPC workflows that may depend on
  `features.h5` being present.
- **Auto-detect on read.** `IdentityFeatures` checks for `metadata.json` (Parquet cache marker)
  first; if absent it falls back to `features.h5`. The `cache_format` parameter only controls
  what is written when no cache exists yet.
- **Format-agnostic `features.py`.** All format-specific I/O moves to `jabs-io`, making
  `features.py` a pure orchestrator.
- **Diverged version spaces.** HDF5 and Parquet caches maintain separate version constants
  (`HDF5_FEATURE_VERSION`, `PARQUET_FEATURE_VERSION`) so they can evolve independently.

### Write-format decision table

| Entry point                     | Default `cache_format` | Rationale                                            |
|---------------------------------|------------------------|------------------------------------------------------|
| GUI (classification / training) | `PARQUET`              | Speed benefit for interactive use                    |
| `jabs-init`                     | `PARQUET`              | Pre-computes for GUI projects                        |
| `jabs-features`                 | `HDF5`                 | HPC batch; preserve existing behavior                |
| `jabs-classify`                 | `HDF5`                 | HPC batch; preserve existing behavior                |
| `jabs-cli` subcommands          | `NA`                   | Currently no `jabs-cli` commands write feature files |

## Consequences

### Cache layout

#### Current layout (HDF5)

```
<project>/jabs/features/<video_stem>/<identity>/
└── features.h5    # single file: attrs + frame_valid + auxiliary arrays + per_frame datasets + window_features_{size} groups
```

#### New layout (Parquet)

```
<project>/jabs/features/<video_stem>/<identity>/
├── metadata.json        # version, pose_hash, scalars, list of cached window sizes
├── per_frame.parquet    # N rows × (auxiliary cols + ~253 feature cols), LZ4
└── window_{size}.parquet  # N rows × ~4529 feature cols, LZ4 (one per cached window size)
```

The presence of `metadata.json` is the signal used during auto-detection to select the Parquet
reader. Its absence causes the reader to fall back to `features.h5`.

#### metadata.json schema

```json
{
  "version": 1,
  "identity": 0,
  "num_frames": 1800,
  "pose_hash": "53f10af4...",
  "distance_scale_factor": 0.07441338,
  "avg_wall_length": 123.4,
  "cached_window_sizes": [5, 10, 15]
}
```

`distance_scale_factor` and `avg_wall_length` are omitted when not applicable.
`cached_window_sizes` is updated each time a new window size is written.

#### per_frame.parquet columns

| Column                         | Type    | Present when                    |
|--------------------------------|---------|---------------------------------|
| `_jabs_frame_valid`            | uint8   | always                          |
| `_jabs_closest_identities`     | int64   | pose v3+ (social features)      |
| `_jabs_closest_fov_identities` | int64   | pose v3+ (social features)      |
| `_jabs_closest_corners`        | float64 | landmark features               |
| `_jabs_closest_lixit`          | float64 | landmark features               |
| `_jabs_wall_{direction}`       | float64 | landmark features, one per wall |
| `{module_name} {feature_name}` | float64 | always (~253 feature columns)   |

Feature column names are identical to the existing HDF5 dataset key format, so no downstream
renaming is required.

#### window_{size}.parquet columns

One row per video frame. All columns are float64:

| Column                                     | Present when                           |
|--------------------------------------------|----------------------------------------|
| `{module_name} {window_op} {feature_name}` | always (~4529 columns per window size) |

Column names are identical to the existing HDF5 dataset key format. One file is written per
cached window size (e.g. `window_5.parquet`, `window_10.parquet`, `window_15.parquet`).

### Architecture

#### New `jabs-io` feature cache module

```
jabs/io/feature_cache/
├── __init__.py          # exports FeatureCacheReader, FeatureCacheWriter, CacheFormat
├── base.py              # abstract FeatureCacheReader / FeatureCacheWriter
├── hdf5.py              # HDF5FeatureCacheReader, HDF5FeatureCacheWriter (code moved from features.py)
└── parquet.py           # ParquetFeatureCacheReader, ParquetFeatureCacheWriter (new)
```

`CacheFormat` is an enum: `CacheFormat.HDF5` | `CacheFormat.PARQUET`.

Auto-detection helper:

```python
def detect_cache_format(identity_dir: Path) -> CacheFormat | None:
    """Return the format of an existing cache, or None if no cache exists."""
    if (identity_dir / "metadata.json").exists():
        return CacheFormat.PARQUET
    if (identity_dir / "features.h5").exists():
        return CacheFormat.HDF5
    return None
```

#### Version constants

```python
HDF5_FEATURE_VERSION = 16    # unchanged; existing HDF5 caches remain valid
PARQUET_FEATURE_VERSION = 1  # starts at 1; independent of HDF5 versioning
```

#### `IdentityFeatures` changes

A `cache_format` parameter controls which format is written for new caches:

```python
def __init__(
    self,
    ...,
    cache_format: CacheFormat = CacheFormat.HDF5,  # default preserves CLI behavior
) -> None:
```

**Read path** (auto-detect, ignores `cache_format`):
1. Call `detect_cache_format(identity_dir)`
2. `PARQUET` → delegate to `ParquetFeatureCacheReader`
3. `HDF5` → delegate to `HDF5FeatureCacheReader`
4. `None` → compute from pose, then write using `cache_format`

**Write path** (uses `cache_format`):
- `CacheFormat.PARQUET` → `ParquetFeatureCacheWriter`
- `CacheFormat.HDF5` → `HDF5FeatureCacheWriter`

### Files changed

| File                                                     | Change                                                                                                |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `src/jabs/feature_extraction/features.py`                | Remove all I/O code; delegate to `jabs-io` backends; add `cache_format` param; keep version constants |
| `packages/jabs-io/src/jabs/io/feature_cache/hdf5.py`     | HDF5 I/O moved here from `features.py`                                                                |
| `packages/jabs-io/src/jabs/io/feature_cache/parquet.py`  | New Parquet I/O                                                                                       |
| `packages/jabs-io/src/jabs/io/feature_cache/base.py`     | Abstract base classes                                                                                 |
| `packages/jabs-io/src/jabs/io/feature_cache/__init__.py` | Public exports + `detect_cache_format`                                                                |
| GUI wiring files (TBD; see open question 1)              | Pass `cache_format=CacheFormat.PARQUET`                                                               |
| `packages/jabs-core/src/jabs/core/constants.py`          | Keep `COMPRESSION`/`COMPRESSION_OPTS_DEFAULT` (still used by HDF5 writer and pose readers)            |

### Cache invalidation and migration

- **HDF5 caches** at version 16 remain valid — the HDF5 reader still accepts them.
- **Parquet caches** start at version 1. A mismatch raises `FeatureVersionException` → recompute.
- **Mixed directories**: if both `metadata.json` and `features.h5` exist, Parquet takes precedence.
- **No automatic migration.** Existing HDF5 projects are used as-is when opened in the GUI.
- **Switching an existing project to Parquet** requires clearing the feature cache:
  - Delete `<project>/jabs/features/` and re-run `jabs-init`, or
  - Re-run `jabs-init --force` (once `--force` propagates to feature recomputation).
- **Partial write safety.** `metadata.json` and `per_frame.parquet` are written as separate
  operations. If a crash occurs between them, `metadata.json` may exist without a corresponding
  `per_frame.parquet`. The reader should treat this state as a missing cache (i.e., delete the
  stale `metadata.json` and recompute), not as a read error.

### Open questions

1. **GUI wiring depth**: How many layers between the GUI and `IdentityFeatures` construction need
   to be updated to pass `cache_format`? Needs a search through `ClassificationThread`,
   `training_workers.py`, and `classify_workers.py`.
2. **Project setting**: Should `cache_format` be stored in `project.json` so that a project
   consistently uses one format, rather than being set at the call site?
3. **`compression_opts` param**: The existing `compression_opts: int` parameter on
   `IdentityFeatures.__init__` is HDF5-specific. Should it be removed from the signature or
   silently ignored when writing Parquet?
4. **Pose hash in cache path**: For HPC workflows where multiple jobs share a single feature
   cache directory, collisions on video stem alone are possible if different pose files happen
   to share the same stem. Including the pose hash as an optional path component —
   `<cache_root>/<video_stem>/<pose_hash>/<identity>/` — would make the cache key
   self-describing and allow safe concurrent use without a locking scheme. For standard GUI
   projects the video stem is already unique within the project directory, so the hash level
   is unnecessary overhead. A `use_pose_hash_in_path` flag (or a dedicated `shared_cache`
   mode) could enable the longer path only for HPC use. The trade-off is that paths become
   harder to inspect by hand and existing caches (keyed without a hash) would not be found
   by a reader expecting the hash level. Note that `detect_cache_format` itself would not need
   a signature change — it operates on the leaf `identity_dir` regardless — but the
   path-construction logic that derives `identity_dir` from a video stem would need to be
   aware of the flag. If this is landed, it should be done alongside the Parquet migration
   rather than as a separate flag day. This is discussed in an existing GitHub issue:
   https://github.com/KumarLabJax/JABS-behavior-classifier/issues/52.