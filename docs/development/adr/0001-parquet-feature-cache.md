# ADR 0001: Parquet Feature Cache

## Status

Proposed

## Context

Feature cache files are currently stored as compressed HDF5 (gzip level 6). The current I/O path
has two compounding bottlenecks: each of the ~253 per-frame feature columns and ~4529 window
feature columns is stored as its own HDF5 dataset and must be read individually, and each dataset
is gzip-compressed — so workers pay both the per-dataset open/seek overhead and the decompression
cost for every column. During classification of large projects this leaves CPUs underutilized
waiting on I/O.

Parquet with LZ4 compression provides equivalent or better compression ratios with 3–5× faster
decompression, and pyarrow's columnar reader is a natural fit for the DataFrame-shaped access
pattern already used throughout the feature extraction code. `pyarrow>=20.0.0` is already a
root dependency — no new packages are needed.

## Decision

Migrate the feature cache format from HDF5 to Parquet. Both formats coexist until possible
HDF5 feature cache format deprecation; the reader auto-detects which format is present.

### Design goals

- **New GUI-facing workflows default to Parquet.** The GUI and `jabs-init` write Parquet for new
  caches. `jabs-init` is the recommended way to pre-compute features for GUI projects, so it
  aligns with the GUI default.
- **Existing HDF5 caches are preserved as-is.** When the GUI opens a project that already has
  `features.h5` files, those are used without conversion. Users must explicitly opt in to Parquet
  through the project settings dialog and then clear the cache to switch an existing project.
- **HPC CLI tools migrate to Parquet.** `jabs-features` and `jabs-classify` will also be
  updated to write Parquet feature caches by default. If there are any incompatibilities with
  existing Nextflow workflows they will be addressed by updating the pipelines.
- **`cache_format` persisted in `project.json`.** Each project stores its intended cache format
  so that every read and write operation within that project uses a consistent format. New
  projects are created with `"cache_format": "parquet"`; projects opened for the first time after
  this change is deployed have `"cache_format": "hdf5"` written to `project.json` on open (inferred
  from the presence of existing `features.h5` files, or defaulted to `"hdf5"` when no cache exists
  yet but the project predates this change).
- **Auto-detect on read.** `IdentityFeatures` checks for `metadata.json` (Parquet cache marker)
  first; if absent it falls back to `features.h5`. The `cache_format` parameter only controls
  what is written when no cache exists yet.
- **Format-agnostic `features.py`.** All format-specific I/O moves to `jabs-io`, making
  `features.py` a pure orchestrator.
- **Diverged version spaces.** HDF5 and Parquet caches maintain separate version constants
  (`HDF5_FEATURE_VERSION`, `PARQUET_FEATURE_VERSION`) so they can evolve independently.

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
  "feature_version": 16,
  "format_version": 1,
  "identity": 0,
  "num_frames": 1800,
  "pose_hash": "53f10af4...",
  "distance_scale_factor": 0.07441338,
  "avg_wall_length": 123.4,
  "cached_window_sizes": [5, 10, 15]
}
```

`feature_version` is the value of `FEATURE_VERSION` from `features.py` at the time the cache was
written; it is passed in by the caller. `format_version` is the Parquet format version owned by
`jabs-io`. Both are validated independently on read — a mismatch on either triggers recomputation.

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

There are two distinct reasons a cached file may need to be discarded and recomputed:

1. **Feature calculation changed.** A new feature was added, an existing formula was corrected, or
   the set of window operations changed. The cached *values* are stale regardless of which format
   they are stored in. `features.py` is the natural owner of this signal — it is the only layer
   that knows what the feature set looks like.

2. **On-disk schema changed.** A new field was added to `metadata.json`, column names were
   renamed, or the Parquet encoding was altered. The cached *files* can no longer be read
   correctly even though the feature values themselves are still valid. `jabs-io` is the natural
   owner of this signal — `features.py` has no knowledge of how data is laid out on disk.

Conflating both concerns into a single version number forces false invalidations: a pure schema
fix would require bumping a version that `features.py` owns, misleading readers of the git
history into thinking feature calculations changed; conversely, adding a new feature would appear
to imply a schema change. Keeping them separate makes each bump's motivation unambiguous and
ensures the owning layer can evolve independently.

The two concerns that can require cache invalidation are therefore kept in separate constants
owned by the layer responsible for each:

| Constant                 | Owner         | Location                                        | Bump when                                                                              |
|--------------------------|---------------|-------------------------------------------------|----------------------------------------------------------------------------------------|
| `FEATURE_VERSION`        | `features.py` | `src/jabs/feature_extraction/`                  | Feature calculations change (new feature, formula fix, window op change)               |
| `PARQUET_FORMAT_VERSION` | `jabs-io`     | `packages/jabs-io/.../feature_cache/parquet.py` | Parquet schema changes (new required metadata field, column renaming, encoding change) |

```python
# src/jabs/feature_extraction/features.py
FEATURE_VERSION = 16    # shared by both HDF5 and Parquet paths; bump for calculation changes

# packages/jabs-io/src/jabs/io/feature_cache/parquet.py
PARQUET_FORMAT_VERSION = 1  # bump for schema/encoding changes independent of feature calculations
```

`FEATURE_VERSION` keeps its current name and value (16). It is format-agnostic: the same value is
written into HDF5 attributes and into `metadata.json`, and bumping it invalidates caches of both
formats.

`PARQUET_FORMAT_VERSION` is Parquet-only. The `ParquetFeatureCacheWriter` receives
`feature_version` as a parameter from the caller (`features.py`) rather than importing
`features.py` directly, keeping `jabs-io` free of upward dependencies.

**HDF5 versioning note:** HDF5 caches currently conflate feature and format versioning in the
single `FEATURE_VERSION` attribute. This is acceptable for now since the HDF5 format is stable.
If `FEATURE_VERSION` is bumped in the future for a reason that affects only the HDF5 format (not
feature calculations), the HDF5 backend will be updated at that time to adopt the same two-version
scheme — a separate `HDF5_FORMAT_VERSION` attribute stored alongside `FEATURE_VERSION` in the
HDF5 file attributes.

#### `IdentityFeatures` changes

A `cache_format` parameter is added and the `compression_opts` parameter is removed:

```python
def __init__(
    self,
    ...,
    cache_format: CacheFormat = CacheFormat.HDF5,  # default preserves CLI behavior
    # compression_opts removed — compression level stored as a constant in the HDF5 writer
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

| File                                                     | Change                                                                                                     |
|----------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `src/jabs/feature_extraction/features.py`                | Remove all I/O code; delegate to `jabs-io` backends; add `cache_format` param; keep version constants      |
| `packages/jabs-io/src/jabs/io/feature_cache/hdf5.py`     | HDF5 I/O moved here from `features.py`                                                                     |
| `packages/jabs-io/src/jabs/io/feature_cache/parquet.py`  | New Parquet I/O                                                                                            |
| `packages/jabs-io/src/jabs/io/feature_cache/base.py`     | Abstract base classes                                                                                      |
| `packages/jabs-io/src/jabs/io/feature_cache/__init__.py` | Public exports + `detect_cache_format`                                                                     |
| `project.json` schema                                    | Add `cache_format` field (`"parquet"` for new projects, `"hdf5"` for existing)                             |
| `src/jabs/project/project.py` (or equivalent)            | Read `cache_format` from `project.json`; pass `CacheFormat` to `IdentityFeatures`; write default on open   |
| GUI settings dialog                                      | Expose `cache_format` as a user-editable field                                                             |
| GUI menu / `MainWindow`                                  | Move *Clear Project Cache* → File menu as *Clear Pose Cache*; add *Clear Feature Cache…* action            |
| GUI wiring files (TBD; see open question 1)              | Thread `cache_format` from `Project` down to `IdentityFeatures` construction                               |
| `src/jabs/scripts/jabs_init.py` (or equivalent)          | Add `--cache-format` option; update `project.json` before feature extraction                               |
| `packages/jabs-core/src/jabs/core/constants.py`          | Keep `COMPRESSION`/`COMPRESSION_OPTS_DEFAULT` (still used by HDF5 writer and pose readers)                 |

### Project configuration

`project.json` gains a `cache_format` field that controls which format is written for new caches:

```json
{
  "cache_format": "parquet"
}
```

Allowed values: `"parquet"` | `"hdf5"`.

**Setting on project creation:** `jabs-init` writes `"cache_format": "parquet"`.

**Setting for existing projects opened after this change is deployed:** On first open, the GUI
inspects the `jabs/features/` directory. If any `features.h5` files are present, it writes
`"cache_format": "hdf5"` to `project.json`. If no feature cache exists yet (or the project
predates this change and has no `cache_format` key), it also defaults to `"hdf5"` to preserve
the behavior the user had before upgrading.

**GUI wiring:** `Project` reads `cache_format` from `project.json` and passes the corresponding
`CacheFormat` enum value when constructing `IdentityFeatures`. This is the single authoritative
source; no call site needs to decide the format independently.

### GUI menu changes

The existing **JABS → Clear Project Cache** menu action (which clears pose-related cache files)
is moved to the **File** menu and renamed **Clear Pose Cache**. A second, separate action —
**Clear Feature Cache...** — is added to the File menu alongside it. The two actions are kept
distinct because they have very different cost profiles: pose cache regenerates quickly on next
open, while feature cache regeneration can take 10s of minutes on large projects.

```
File
├── …
├── Clear Pose Cache
└── Clear Feature Cache…
```

The **Clear Feature Cache** action opens a confirmation dialog. When the project's `cache_format`
is `"hdf5"`, the dialog includes an upgrade hint:

> *This project is configured to use HDF5 feature cache. To switch to the faster Parquet format,
> update **Cache Format** in Project Settings before clearing.*

### Upgrade process for existing projects

**Via the GUI:**

1. Open **Project Settings** and change **Cache Format** from `HDF5` to `Parquet`. This updates
   `project.json` immediately.
2. Use **File → Clear Feature Cache** to delete the existing cache.
3. Trigger feature extraction (via the GUI or `jabs-init`). Features are recomputed and written
   as Parquet.

**Via the CLI:**

```bash
jabs-init --force --cache-format parquet /path/to/project
```

`--cache-format` updates `project.json` before feature extraction begins; `--force` overwrites the
existing cache. The combined invocation replaces steps 1–3 above in a single command.

`--cache-format` without `--force` updates `project.json` but does not regenerate previously cached 
features (regardless of format) — the new format takes effect on the next cache miss (i.e., after 
the cache is cleared or invalidated by a version bump).

**Clearing the cache without changing the format setting** regenerates features in whatever format
`project.json` currently specifies. If `cache_format` is still `"hdf5"`, the new cache will be
HDF5. There is no automatic promotion to Parquet on cache clear — the format change must be an
explicit user action.

### Cache invalidation

- **HDF5 caches** at `FEATURE_VERSION` 16 remain valid — the HDF5 reader still accepts them.
- **Parquet caches** are validated against two independent version fields in `metadata.json`:
  - `feature_version` mismatch (vs. `FEATURE_VERSION` in `features.py`) → recompute.
  - `format_version` mismatch (vs. `PARQUET_FORMAT_VERSION` in `parquet.py`) → recompute.
  - Either mismatch raises `FeatureVersionException`.
- **Mixed directories**: if both `metadata.json` and `features.h5` exist in the same identity
  directory, Parquet takes precedence (this state should not arise in normal use).
- **Partial write safety.** `metadata.json` and `per_frame.parquet` are written as separate
  operations. If a crash occurs between them, `metadata.json` may exist without a corresponding
  `per_frame.parquet`. The reader should treat this state as a missing cache (i.e., delete the
  stale `metadata.json` and recompute), not as a read error.

### Open questions

1. **GUI wiring depth**: `Project` is the authoritative source for `cache_format` (read from
   `project.json`). The remaining question is how many layers between `Project` and
   `IdentityFeatures` construction need to be updated to thread the value through — specifically
   whether `ClassificationThread` receives it directly from `Project` or via an intermediate
   object. Needs a search through `ClassificationThread`.
2. ~~**Project setting**: Should `cache_format` be stored in `project.json`?~~ **Resolved:** Yes.
   `project.json` stores `cache_format`; `Project` reads it and passes the value when constructing
   `IdentityFeatures`. See the *Project configuration* and *Upgrade process* sections above.
3. ~~**`compression_opts` param**: Should it be removed from the signature or silently ignored
   when writing Parquet?~~ **Resolved:** Removed from the `IdentityFeatures.__init__` signature.
   The default value is always used in practice and there is no mechanism to change it without
   editing code, so it provides no real configurability. The compression level is stored as a
   constant in the HDF5 writer and accessed directly.
4. **Pose hash in cache path**: For HPC workflows where multiple jobs share a single feature
   cache directory, collisions on video stem alone are possible if different pose files happen
   to share the same stem. Including the pose hash as an optional path component —
   `<cache_root>/<video_stem>/<pose_hash>/<identity>/` — would make the cache key
   self-describing. For standard GUI projects the video stem is already unique within the
   project directory, so the hash level is unnecessary overhead. A `use_pose_hash_in_path`
   flag (or a dedicated `shared_cache` mode) could enable the longer path only for HPC use.
   The trade-off is that paths become harder to inspect by hand and existing caches (keyed
   without a hash) would not be found by a reader expecting the hash level. Note that
   `detect_cache_format` itself would not need a signature change — it operates on the leaf
   `identity_dir` regardless — but the path-construction logic that derives `identity_dir` from
   a video stem would need to be aware of the flag. If this is landed, it should be done
   alongside the Parquet migration rather than as a separate flag day. This is discussed in an
   existing GitHub issue: https://github.com/KumarLabJax/JABS-behavior-classifier/issues/52.