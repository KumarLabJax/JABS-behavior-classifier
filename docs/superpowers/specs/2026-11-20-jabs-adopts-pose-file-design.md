# JABS adopts the unified pose file — design

Project 2 of 5 on the path to ViT-SPAL embeddings in JABS classifiers
(see `2026-11-19-embedding-components-design.md` for the roadmap). Builds on
#596 (unified pose file, ADR 0002) and #598 (legacy converter); branched from
#598's head.

## Goal

Every pose JABS reads goes through the unified pose file (`jabs.pose-file`,
ADR 0002). Legacy `*_pose_est_v2…v8.h5` files keep working without any change
on disk. The per-version pose classes and every version branch are deleted, so
project 4 has one place to add embedding access.

## Decisions (agreed 2026-11-20)

1. **No feature parity.** JABS adopts the format's semantics; `FEATURE_VERSION`
   goes 19 → 20 and classifiers are retrained.
2. **Legacy files are converted at open time.** Nothing next to the video is
   written or renamed.
3. **One concrete `Pose` class over `PoseFile`.** It replaces the
   `PoseEstimation` ABC and `PoseEstimationV2…V8`. Method names are kept where
   the meaning holds.

## Design

### 1. Opening a pose file

`open_pose_file(path, cache_dir=None) -> Pose` (`src/jabs/pose_estimation/__init__.py`):

- `*_pose.h5` (a `jabs.pose-file`): read directly.
- `*_pose_est_v{2..8}.h5`: converted by `convert_legacy`, then read.
- Anything else: `PoseFileError` naming the file and the two accepted patterns.

**Where a conversion lives.** The converter streams to a destination file and
deep-checks it; an hour of dense segmentation does not fit comfortably in RAM.
So "in memory" is implemented as **a converted file in the project's cache
directory**, `<cache_dir>/converted/<key>.h5`, reused while the legacy
file's hash **and the converter version** (jabs-io version) are unchanged; the
key is `<legacy-hash>-<jabs-io version>`, so a converter fix regenerates old
conversions. Two workers converting the same file at once both succeed (the
write is atomic); the second write is wasted work, which is acceptable. The video directory is never written. Without a
`cache_dir` (one-off CLI use) the conversion goes to a temporary directory that
is removed when the `Pose` is closed.

**v2/v3 support in the converter.** `convert_legacy` gains v2 (single mouse,
12 keypoints, no identity datasets) and v3 (`instance_count`,
`instance_embedding`, `instance_track_id`). jabs-vision still writes v2.

**Discovery.** Every place that globs `*_pose_est_v*.h5` or derives a video
stem from a pose name (`pose_file_stem`, the viewer's file filter, project
discovery, `get_pose_file_major_version` callers, ~8 sites) goes through one
helper, `find_pose_file(video_path) -> Path | None`. It accepts both patterns.
When a video has both, the unified file wins; among legacy files the highest
version wins (today's rule).

### 2. Semantics (FEATURE_VERSION 20)

| Concern | Today (v19) | After (v20) |
|---|---|---|
| Point validity | conf ≤ 0.3 → NaN, float64 compare | `jabs.pose.point_valid`; invalid points are NaN in `get_points`/`get_identity_poses` (coordinates stay in the file) |
| Identity present | ≥3 valid body keypoints (v3: 1) | `jabs.pose.slot_usable[:, identity]`; when the file has none (it is optional, ADR 0002), derived from `point_valid` with ≥3 valid keypoints excluding the skeleton's tail keypoints, by name, not position |
| Slots iterated | all | identity slots only (`dimensions.identity`); tail slots never become JABS identities |
| fps | always 30 in features | manifest `video.fps`; 30 only when absent (logged once) |
| `cm_per_pixel` | float32 attr | manifest value, Python float; `None` when absent |
| Static objects | per-version shapes | always `(P, 2)`; single-point objects (lixit with P=1) squeezed in one helper, `static_object(name)` |
| Segmentation pairing | `longterm_seg_id` | the format's instance→slot mapping |
| Segmentation access | whole array | `segmentation_window(identity, start, stop)`; whole-file reads are not offered |

Feature names do not change: keypoint names (`KeypointIndex`) stay the column
names.

### 3. The `Pose` class

`src/jabs/pose_estimation/pose.py`, one concrete class wrapping a `PoseFile`.

- **Kept, same meaning:** `num_frames`, `identities`, `num_identities`, `fps`,
  `cm_per_pixel`, `pose_file`, `hash`, `get_points`, `get_identity_poses`,
  `get_identity_point_mask`, `get_reduced_point_mask`, `get_connected_segments`,
  `identity_mask`, `identity_to_track`, `static_objects`, `dynamic_objects`,
  `get_dynamic_object`, `get_identity_convex_hulls`, `compute_bearing`,
  `compute_all_bearings`, `num_lixit_keypoints`, `external_identities`,
  `identity_index_to_display`, `get_bounding_boxes`.
- **Replaced:** `format_major_version` and every `>= N` check (feature
  manager, read/export training, lixit/corner features, CLI, UI overlays) by
  capability properties: `has_segmentation`, `has_identity_tracking`,
  `has_static_object(name)`, `has_bounding_boxes`. The project-level
  `_min_pose_version` becomes "all files have capability X".
- **Deleted:** `packages/jabs-core/src/jabs/core/abstract/pose_est.py`'s
  `PoseEstimation`, `src/jabs/pose_estimation/pose_est_v2.py`…`v8.py`,
  `identity_indexed_pose.py` if nothing else uses it.
- **Moved:** `KeypointIndex`, `FULL_CONNECTED_SEGMENTS` and `MINIMUM_CONFIDENCE`
  to a neutral jabs-core module (`jabs.core.pose_skeleton`). Their users outside
  `src/jabs` move with them: `jabs-io` (`pose_file/skeletons.py`), `jabs-vision`
  (`cli/pose.py`, which still writes v2) and `jabs-core` (`utils/pose_util.py`,
  whose `identity_centroids` takes the new `Pose` via a Protocol in jabs-core).
- **jabs-core is published:** `jabs.core.abstract.PoseEstimation` stays for one
  release as a deprecated name that only carries those constants and warns
  (`DeprecationWarning`) on import; its abstract methods are removed.
- `update_pose` (swaps a project's pose set) uses `find_pose_file`, so it accepts
  unified files; it never writes into pose files.

### 4. Caches, hashes and classifiers

- `Pose.hash` is the hash of **the file the user has** (the legacy file for a
  conversion), so feature caches, prediction links and project merge keep
  keying on files on disk.
- `FEATURE_VERSION = 20` invalidates all feature caches through the existing
  cache-version check.
- Classifiers record the `FEATURE_VERSION` they were trained on. Loading one
  with a different (or no recorded) version raises a clear "retrain this
  classifier" error instead of predicting on mismatched features.
- The convex-hull cache path mismatch is fixed while moving its code.

### 5. Testing

- Conformance: each legacy version (v2–v8 fixture) converts to a file that
  passes `validate(deep=True)` and opens as a `Pose`.
- New fixtures: confidences in (0, 0.3] (the current samples have none), a
  non-30 fps file, a P=1 lixit, a v4 file with zero identities (today's crash),
  a video directory holding both a unified and a legacy file.
- End to end: open project → compute features → train → predict, run on a
  legacy-only project and a unified-only project.
- Existing feature tests are updated to v20 values where semantics changed,
  each change explained by a row of the table in §2.

## Out of scope

Embedding features (project 4), ViT SPAL writing files (project 3), on-disk
conversion of projects and the Hub cutover, rebasing #603.

## Risks

- **Conversion time on first open** of a large legacy project. Mitigation:
  conversion runs inside the existing per-video feature/progress workers, and
  converted files are reused.
- **Cache disk use**: a converted copy per legacy file. Acceptable for now; the
  on-disk migration later removes the duplication.
- **Converter refusals** (duplicate identities, shared tracklets) on files JABS
  opens today become errors. The error names the file and the reason; such
  files are counted during the end-to-end run on real projects.
