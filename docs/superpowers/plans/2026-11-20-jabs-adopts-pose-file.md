# JABS Adopts the Unified Pose File — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every pose JABS reads goes through one `Pose` class over the unified pose file; legacy v2–v8 files are converted into a keyed cache copy when first needed.

**Architecture:** Callers first move to a version-free API (capabilities, segmentation windows, skeleton constants) that the legacy readers also implement, so the suite stays green throughout. `Pose` and the conversion cache are built beside the legacy readers. A final flip points `open_pose_file` at `Pose` and deletes the readers. `FEATURE_VERSION` becomes 20 and classifiers record the feature version of their training data.

**Tech Stack:** Python 3.10+, h5py, numpy, platformdirs, jabs-io `pose_file` (ADR 0002), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-20-jabs-adopts-pose-file-design.md`

## Global Constraints

- **Every task ends with the full suite green:** `uv run pytest` (root) plus `uv run pytest packages/<pkg>/tests` for each package the task touched, and `uv run ruff check . && uv run ruff format --check .`. Environment: `uv sync --all-packages --all-extras --all-groups`.
- Repo style (CLAUDE.md): full type hints, Google docstrings, `logger = logging.getLogger(__name__)`, lazy log formatting, no `print`, pytest functions with `monkeypatch`/`unittest.mock` (no `mocker`), no EN DASH in docstrings.
- Filenames: unified `<video stem>_pose.h5`; legacy `<video stem>_pose_est_v<N>.h5`, N in 2..8. Unified wins when both exist; among legacy, highest N wins. Dotfiles (`._*`) are never pose files.
- Conversion cache file: `<cache root>/converted/<video stem>.<legacy hash>-<jabs-io version>.h5`. `<cache root>` is the project's `cache_dir`, or `platformdirs.user_cache_dir("jabs")` when there is none. Legacy hash = `jabs.core.utils.hash_file(legacy path)`; version from a module-level helper `_converter_version()` in `conversion_cache.py` (wrapping `importlib.metadata.version("jabs-io")`).
- `Pose.hash` = `hash_file(<the file the user has>)`, computed once and passed along (never hash a file twice per open).
- Point validity = `jabs.pose.point_valid`; invalid points are NaN in returned arrays. Identity present = `jabs.pose.slot_usable` when present, else `>= 3` valid keypoints among skeleton body parts other than `MID_TAIL`, `TIP_TAIL` (by name).
- Identities = slots `0..dimensions["identity"]-1`. When `dimensions["identity"] == 0` **and** `jabs.pose.tracklet_id` is present (v3 conversions), every slot is an identity; otherwise `identity == 0` means no identities.
- fps = manifest `video.fps`, else 30 (INFO log once per `Pose`). Features read fps **only** from the pose object.
- Static objects stored `(O, P, 2)` float32 xy under `jabs.static_objects.<name>`; `Pose.static_objects[name]` squeezes P when P == 1; `num_lixit_keypoints` = P of `lixit`.
- `PoseCapability` (`jabs.core.enums`, `str, Enum`): `SOCIAL = "social"` (more than one identity), `SEGMENTATION = "segmentation"`, `BOUNDING_BOXES = "bounding_boxes"`. Static objects are checked by name.
- Return dtypes after the flip (no parity required): points float32, masks bool (`identity_mask` bool; callers comparing `== 1` are updated), static objects float32. Tests asserting old dtypes are updated, not treated as regressions.
- `FEATURE_VERSION = 20`. Classifier errors about feature versions raise `ClassifierFeatureVersionError(ValueError)` with `retrain` in the message.

## Review Focus

1. **Mixed legacy + unified project**: capabilities intersect and per-frame feature column names are identical across videos (Task 15 e2e asserts column equality).
2. **Legacy file replaced after conversion**: new hash, reconversion, no stale cache read; convex hulls recomputed (Tasks 7, 6).
3. **Converter refusal on a file JABS used to open**: the video is marked unusable with the reason; the project opens (Task 10).
4. **Hour-long file with dense segmentation**: `Pose` never calls `read_pose_file`; segmentation comes from chunk-aligned cached windows (Task 6).
5. **v3 track continuity**: each slot of a converted v3 file holds one track per run; per-identity points equal today's `PoseEstimationV3` (Task 5, golden arrays).

---

### Task 1: Pose skeleton constants and the capability enum in jabs-core

**Files:**
- Create: `packages/jabs-core/src/jabs/core/pose_skeleton.py`, `packages/jabs-core/src/jabs/core/enums/pose_capability.py`, `packages/jabs-core/tests/test_pose_skeleton.py`
- Modify: `packages/jabs-core/src/jabs/core/enums/__init__.py`, `packages/jabs-core/src/jabs/core/abstract/pose_est.py` (class attrs become aliases)

**Interfaces:**
- Produces: `jabs.core.pose_skeleton.{KeypointIndex, FULL_CONNECTED_SEGMENTS, NVSN_CONNECTED_SEGMENTS, MINIMUM_CONFIDENCE, TAIL_KEYPOINTS}` (`TAIL_KEYPOINTS = ("MID_TAIL", "TIP_TAIL")`); `jabs.core.enums.PoseCapability`; `jabs.core.types.SegmentationWindow` (frozen dataclass: `contours: NDArray[int32] (n, C, P, 2)`, `contour_count: NDArray[uint32] (n,)`, `contour_length: NDArray[uint32] (n, C)`, `external_flag: NDArray[bool] (n, C) | None`, `start: int`).
- `PoseEstimation.KeypointIndex`, `.FULL_CONNECTED_SEGMENTS`, `.NVSN_CONNECTED_SEGMENTS` are the same objects as the new module's.

- [ ] **Step 1:** Tests: names/values equal the ABC's; `PoseEstimation.KeypointIndex is pose_skeleton.KeypointIndex`; enum values; `SegmentationWindow` is frozen.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Commit `refactor(core): pose skeleton constants and capability enum`.

### Task 2: Move every skeleton-constant user to `jabs.core.pose_skeleton`

Mechanical. **Files:** every `src/`, `packages/*/src` and test module referencing `PoseEstimation.KeypointIndex`, `PoseEstimation.FULL_CONNECTED_SEGMENTS`, `PoseEstimation.NVSN_CONNECTED_SEGMENTS` or importing `MINIMUM_CONFIDENCE` from `jabs.core.abstract.pose_est` / `jabs.pose_estimation` — including `packages/jabs-io/src/jabs/io/internal/pose_file/skeletons.py:16,47-48`, `packages/jabs-io/src/jabs/io/internal/pose/nwb.py:41,65`, `packages/jabs-vision/src/jabs/vision/cli/pose.py:11-27`, `packages/jabs-core/src/jabs/core/utils/pose_util.py`.

- [ ] **Step 1:** Replace imports/uses. `pose_util.identity_centroids` takes a `PoseLike` Protocol defined in `pose_util.py` (the methods it calls).
- [ ] **Step 2:** Acceptance: `grep -rnE "PoseEstimation\.(KeypointIndex|FULL_CONNECTED|NVSN)" src packages tests --include='*.py'` is empty; full suite + all package suites → PASS.
- [ ] **Step 3:** Commit `refactor: import pose skeleton constants from jabs.core.pose_skeleton`.

### Task 3: Version-free API on the legacy readers (the bridge)

**Files:**
- Modify: `packages/jabs-core/src/jabs/core/abstract/pose_est.py`, `src/jabs/pose_estimation/pose_est_v{2..8}.py`
- Test: `tests/pose_estimation/test_bridge_api.py`

**Interfaces (implemented on the ABC, overridden where the data lives):**
- `capabilities: frozenset[PoseCapability]` — `SOCIAL` iff `num_identities > 1`; `SEGMENTATION` iff the reader has segmentation (v6+ with data); `BOUNDING_BOXES` iff v8 `has_bounding_boxes`.
- `has_segmentation`, `has_bounding_boxes`, `has_social` (bool properties); `has_static_object(name) -> bool`.
- `segmentation_window(identity, start, stop) -> SegmentationWindow` and `iter_segmentation(identity, block: int = 300) -> Iterator[SegmentationWindow]` on v6+ (from the in-memory arrays; `-1` padding converted to counts/lengths); raise `NotImplementedError` without segmentation.
- `source_format: str` (`"pose_est_v<N>"`), `model_metadata: dict | None` (from `poseest.attrs["model_metadata_json"]` when present).

- [ ] **Step 1:** Tests on `tests/data/sample_pose_est_v{2,4,6}.h5`: capability sets (v2 → empty; v4 sample `SOCIAL`; v6 `SOCIAL, SEGMENTATION`); `iter_segmentation` concatenated equals `get_segmentation_data` for one identity, with counts/lengths matching non-`-1` entries.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Commit `feat(pose): version-free capability and segmentation API on legacy readers`.

### Task 4: Converter accepts v2

**Files:** `packages/jabs-io/src/jabs/io/internal/pose_file/convert/{inspect,plan,sources,provenance,memory,cli}.py`; `packages/jabs-io/tests/data/legacy-pose/generate.py` (add `v2-single.h5`); test `packages/jabs-io/tests/internal/pose_file/convert/test_convert_v2.py`.

**Interfaces:** `LegacyLayout` gains `instance_axis: bool`. v2 output: `{"frame": F, "slot": 1, "identity": 1}`, `slot_occupied = any(confidence > 0)`, `point_valid = confidence > 0.3` (float64 compare, as v4+), no tracklets/segmentation/objects; `source_format` → `pose_est_v2`. Docstrings/CLI help "v4–v8" → "v2–v8".

- [ ] **Step 1:** Fixture: points `(F,12,2)` uint16 yx, confidence `(F,12)` with values in `(0, 0.3]` and one all-zero frame. Tests: deep-valid; xy flip; `point_valid` exactly `conf > 0.3`; all-zero frame unoccupied; repo `tests/data/sample_pose_est_v2.h5` converts (`requires_samples`).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Converter + full suites → PASS.
- [ ] **Step 5:** Commit `feat(pose-file): convert legacy pose_est_v2`.

### Task 5: Converter accepts v3 with track-continuous slots

**Files:** same converter modules; generator `v3-tracklets.h5`; test `convert/test_convert_v3.py`; golden arrays `packages/jabs-io/tests/data/legacy-pose/v3-sample-expected.npz` produced by a script that runs today's `PoseEstimationV3` (commit the script beside it).

**Interfaces:** v3 output: `identity = 0`; `slot` = the maximum number of concurrent tracks; each track occupies one slot for its whole run, assigned by porting the free-list of `src/jabs/pose_estimation/pose_est_v3.py:_build_identity_map` (216-253) into `plan.py`; `jabs.pose.tracklet_id` = legacy track id + 1 (offset recorded in the convert entry `extra`); `slot_occupied = any(confidence > 0)`; `instance_count` only bounds which legacy instances are read; `instance_embedding` dropped and listed in `dropped`.

- [ ] **Step 1:** Fixture with 2–3 tracks that end and start (forcing compaction shifts) and a real track id 0. Tests: deep-valid; `identity == 0`; within each slot `tracklet_id` changes only across an unoccupied frame; track 0 stored as 1; on `tests/data/sample_pose_est_v3.h5` and `identity_with_no_data_pose_est_v3.h5`, per-slot points/masks equal the golden arrays (NaN-aware).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Converter suite incl. `test_legacy_corpus.py` + full → PASS.
- [ ] **Step 5:** Commit `feat(pose-file): convert legacy pose_est_v3 with one track per slot`.

### Task 6: The `Pose` class

**Files:** create `src/jabs/pose_estimation/pose.py`, `tests/pose_estimation/test_pose.py`, `tests/pose_estimation/conftest.py` (builders writing tiny unified files with `write_pose_file`).

**Interfaces:**
- `Pose(pose_path: Path, *, source_path: Path | None = None, source_hash: str | None = None, cache_dir: Path | None = None)`. Opens one `h5py.File` for its lifetime (self-containment checked once, as `read_component` does), `read_manifest` once; eagerly reads points, point_valid, slot_occupied, slot_usable?, bbox?, tracklet_id?, external_ids?, static and dynamic objects. `close()`, context manager, and `weakref.finalize` closing the file. Never calls `read_pose_file`.
- Same names/contracts as the ABC: `num_frames, identities, num_identities, fps, cm_per_pixel, pose_file (= source_path), hash, identity_to_track, static_objects, dynamic_objects, num_lixit_keypoints, external_identities, get_points, get_identity_poses, get_identity_point_mask, get_reduced_point_mask, get_connected_segments (FULL vs NVSN selection ported from pose_est_v5.py:64-96), identity_mask, get_dynamic_object, compute_bearing, compute_all_bearings, identity_index_to_display, get_bounding_boxes, get_identity_convex_hulls`, plus the Task 3 API (`capabilities, has_*, segmentation_window, iter_segmentation, source_format, model_metadata`).
- `identity_to_track`: `(F, identities)` int, `tracklet_id - 1` where occupied, `-1` elsewhere; `None` without tracklets.
- `dynamic_objects`: `frame_index → sample_indices int64`, `points → float64`, `counts → int64` (`DynamicObjectData`, `core/types/pose.py:29-31`).
- `source_format` / `model_metadata` read from `/provenance` (the convert entry's source format and preserved `model_metadata_json` in `extra`); for native files, `source_format = "jabs.pose-file"`.
- Segmentation: an LRU (8 entries) of 300-frame, chunk-aligned windows read across all slots from the open file; `segmentation_window` slices it.
- Convex hulls cached at `<cache_dir>/convex_hulls/<video base>/<hash>/convex_hulls_<identity>.pickle`; copy the ABC's hull algorithm.

- [ ] **Step 1:** Tests: invalid point NaN/mask False; `slot_usable` used when present; derived mask excludes tail; tail slots not identities (`slot=3, identity=2`); `identity=0` + tracklets → slots are identities; `identity=0` without tracklets → `[]`; fps from manifest / default 30; corners `(4,1,2)`→`(4,2)`, lixit `(2,1,2)`→`(2,2)` with `num_lixit_keypoints == 1`, lixit `(1,3,2)` unchanged; capabilities (`SOCIAL` false for 1 identity + tail slots); NVSN selection; `read_pose_file` monkeypatched to raise is never hit; 10 consecutive single-frame windows for 2 identities perform one dataset read (count via a wrapper); hulls recomputed after the source hash changes; `hash` equals the passed `source_hash`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Commit `feat(pose): Pose class over the unified pose file`.

### Task 7: Discovery and the conversion cache

**Files:** create `src/jabs/pose_estimation/discovery.py`, `src/jabs/pose_estimation/conversion_cache.py`; modify `packages/jabs-core/src/jabs/core/utils/utilities.py:69-80` (`pose_file_stem`); add `platformdirs` with `uv add platformdirs`; tests `tests/pose_estimation/test_discovery.py`, `test_conversion_cache.py`, `test_conformance.py`.

**Interfaces:**
- `POSE_FILE_GLOBS = ("*_pose.h5", "*_pose_est_v*.h5")`; `is_pose_file(path) -> bool`; `find_pose_file(video_path, pose_dir=None) -> Path | None` (skips dotfiles).
- `pose_file_stem` strips `_pose` only for `.h5` paths, and `_pose_est_v\d+` as today.
- `cache_root(cache_dir: Path | None) -> Path`; `converted_path(source, source_hash, cache_dir) -> Path`; `ensure_converted(source: Path, source_hash: str, cache_dir: Path | None) -> Path` — mkdir, convert if missing, on `OSError` from the final replace use the destination if it now exists, on success delete same-stem same-hash files of other versions. Wraps `LegacyConversionError` in `PoseFileError(path, reason)` (new, `jabs.pose_estimation.errors`).
- `open_unified(path: Path, cache_dir: Path | None = None) -> Pose` — the future `open_pose_file`; hashes once.

- [ ] **Step 1:** Tests: unified beats legacy; highest legacy; dotfiles skipped; `pose_file_stem("trial_pose.mp4") == "trial_pose"`, `("trial_pose.h5") == "trial"`; key changes with legacy bytes and with `_converter_version` (monkeypatched); second `open_unified` converts zero times; other-version copy removed; `cache_dir=None` uses `user_cache_dir` (monkeypatched to tmp); `PoseFileError` on `v8-duplicate-embed.h5` copied to `vid_pose_est_v8.h5` names file and reason; conformance: repo `sample_pose_est_v{2..6}.h5` and generator v7/v8 fixtures copied to `<name>_pose_est_v<N>.h5` all open with `open_unified`, `validate(deep=True)` passes on the converted file; a v4 generator file with all instances masked (nonzero confidences) opens with `identities == []`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Commit `feat(pose): pose discovery and keyed conversion cache`.

### Task 8: Features on capabilities, segmentation windows and pose fps

**Files:** `src/jabs/feature_extraction/{feature_base_class,feature_group_base_class,features}.py`; `landmark_features/{lixit,food_hopper,corner,landmark_group}.py`; `social_features/{closest_fov_angles,pairwise_social_distances,closest_fov_distances,huddle_features,closest_distances,social_distance}.py`; `segmentation_features/*` (incl. `moment_cache.py`); callers of `IdentityFeatures(fps=...)` (`project/parallel_workers.py:270,308`, `scripts/classify.py:253`, `scripts/cli/compute_features.py:140`, `ui/classification_thread.py:150`); `project/feature_manager.py:127-131`, `scripts/cli/compute_features.py:110-120` (pass capabilities); tests under `tests/feature_extraction/`.

**Interfaces:**
- `Feature._requires: ClassVar[frozenset[PoseCapability]]` replaces `_min_pose` (`3` → `{SOCIAL}`, `5` → `frozenset()` keeping `_static_objects`, segmentation → `{SEGMENTATION}`).
- `is_supported(capabilities, static_objects, **kwargs)`, `get_supported_feature_modules(capabilities, static_objects, **kwargs)`, `get_available_extended_features(capabilities, static_objects, **kwargs)`.
- `IdentityFeatures` drops its `fps` parameter; callers stop passing it; `jabs-classify --fps` and `generate_features --fps` log a deprecation warning and are ignored.
- `moment_cache.py` and `social_distance.py` iterate `iter_segmentation`, using `contour_count`/`contour_length` and `external_flag` instead of `-1` padding and `get_segmentation_flags`.
- `FeatureManager` exposes `capabilities` (intersection over videos, computed from each video's pose metadata as today's min version is) and keeps `min_pose_version` as a thin deprecated property until Task 10.

- [ ] **Step 1:** Tests: social feature unsupported without `SOCIAL`; lixit supported when `lixit` present; segmentation group skipped without `SEGMENTATION`; segmentation features on `sample_pose_est_v6.h5` equal (allclose) the values before this task (golden from the pre-task code); a velocity on a fake pose with `fps=60` is twice the `fps=30` value.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement; `grep -rnE "_min_pose|format_major_version|get_segmentation_(data|flags)" src/jabs/feature_extraction` empty.
- [ ] **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Commit `refactor(features): gate on capabilities, read segmentation windows and pose fps`.

### Task 9: Project scan stays metadata-only

**Files:** `src/jabs/project/{parallel_workers,project,pose_attribute_cache,project_paths}.py`; tests `tests/project/test_pose_attribute_cache.py`, `tests/project/test_scan.py`.

**Interfaces:**
- `scan_pose_metadata(path: Path) -> PoseMetadata` (new, `src/jabs/pose_estimation/metadata.py`): `read_manifest` for unified files, `inspect_legacy` for legacy (no keypoint reads). `PoseMetadata` fields: `frames, num_identities, static_objects, lixit_keypoints, cm_per_pixel, capabilities, source_format`.
- Scan job uses it; `VideoScanJobSpec` drops `pose_major_version`; `pose_attribute_cache` `SCHEMA_VERSION` 2 → 3 with a `capabilities` field (old entries rescanned).
- `project_paths.py:128` glob uses `POSE_FILE_GLOBS`.

- [ ] **Step 1:** Tests: scan of a unified and a legacy file gives the same metadata for the same content; scan never opens keypoint datasets (monkeypatch `read_component` to raise); cache schema 2 entries rescanned.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Commit `refactor(project): metadata-only pose scan for both file kinds`.

### Task 10: Project features, conversion workers and training exports on capabilities

**Files:** `src/jabs/project/{feature_manager,project,video_manager,project_merge,export_training,read_training,parallel_workers}.py`, `src/jabs/ui/main_window/menu_handlers.py:19,222,375`, `src/jabs/ui/export_training_thread.py:35-79`, `src/jabs/scripts/cli/cli.py:137,159`; tests in `tests/project/`, `tests/ui/` where present.

**Interfaces:**
- `FeatureManager.min_pose_version` removed; `Project.settings_by_capabilities(capabilities, static_objects, ...)` replaces `settings_by_pose_version`.
- Feature workers call `ensure_converted` before computing (conversion happens here); `Project` gains `prepare_poses(progress: Callable[[int, int], None])` that converts missing ones in the existing worker pool, called by the GUI before the first pose load and showing progress.
- A `PoseFileError` marks the video unusable (`Project.unusable_videos: dict[str, str]` name → reason, logged at WARNING); the project opens and the video list shows it disabled with the reason.
- Training export writes `pose_capabilities` (sorted values); `read_training` reads it or maps an old `min_pose_version` (≥3 SOCIAL, ≥6 SEGMENTATION, ≥8 BOUNDING_BOXES), and returns `file_version`.
- `get_derived_file_paths` includes the video's converted files (`converted/<video stem>.*.h5`) and the dead legacy reader caches `cache/<base>_pose_est_v*_cache.h5`.

- [ ] **Step 1:** Tests: mixed project capability intersection; refusal (`v8-duplicate-embed.h5` as `vid_pose_est_v8.h5`) → project opens, video unusable with reason; old export `min_pose_version=6` → `{SOCIAL, SEGMENTATION}`; prune removes converted files.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Commit `refactor(project): capabilities, conversion in workers, capability-based training exports`.

### Task 11: CLI scripts on the version-free API

**Files:** `src/jabs/scripts/{classify,initialize_project,generate_features}.py`, `src/jabs/scripts/cli/{compute_features,export_video,convert_to_nwb,update_pose,update_labels,sample_pose_intervals,evaluate,label_remap_matching}.py`, `src/jabs/pose_estimation/__init__.py:112-150` helpers; tests under `tests/scripts/`.

**Interfaces / behavior:**
- Pose files are opened as `with open_pose_file(...) as pose:`; discovery through `find_pose_file`/`get_pose_path`.
- `get_frames_from_file`, `get_static_objects_in_file`, `get_points_per_lixit` deleted; callers use `scan_pose_metadata`.
- `initialize_project`: passes the project cache dir, catches `PoseFileError` and reports the file.
- `update_pose`: capability checks instead of `format_major_version`; metadata from `pose.model_metadata`.
- `convert_to_nwb`: contours via `iter_segmentation`; metadata `pose_source_format` from `pose.source_format`.
- `label_remap_matching`: `iter_segmentation`.
- `sample_pose_intervals`: refuses a unified input with "sample_pose_intervals supports legacy pose_est files only" (clipping unified files is out of scope).
- `classify.py:31` `_POSE_FILE_NAME_RE` accepts both patterns.

- [ ] **Step 1:** Tests: each CLI accepts a `_pose.h5` input where it accepted a legacy one (compute-features, classify, export-video, convert-to-nwb, update-pose); `sample_pose_intervals` refusal message; `jabs-init` on a refused file reports it.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Commit `refactor(cli): pose scripts use discovery, capabilities and segmentation windows`.

### Task 12: UI and overlays on the version-free API

**Files:** `src/jabs/overlay_drawing/{bounding_box,segmentation}.py`, `src/jabs/ui/player_widget/overlays/bounding_box.py`, `src/jabs/ui/main_window/central_widget.py:555-563`, `src/jabs/ui/viewer/viewer_window.py:23,246,396-445`, `src/jabs/ui/dialogs/video_info_dialog.py:117-137`; tests under `tests/overlay_drawing/`, `tests/ui/`.

**Behavior:** overlays check `pose.has_bounding_boxes` / `has_segmentation`; the segmentation overlay draws from `segmentation_window(identity, frame, frame + 1)`; viewer filter `"Pose Files (*_pose.h5 *_pose_est_v*.h5);;All Files (*)"`; the viewer opens (and converts) a pose in a worker thread with a busy indicator; the info dialog reads static objects and model metadata from `Pose`/`scan_pose_metadata`.

- [ ] **Step 1:** Tests: segmentation overlay on v6 sample draws the window's contour; bbox overlay disabled without `BOUNDING_BOXES`; dialog shows static objects.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement; `grep -rn "format_major_version" src/jabs/ui src/jabs/overlay_drawing` empty. **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Commit `refactor(ui): overlays and viewer use pose capabilities`.

### Task 13: Retype pose annotations and test fakes

Mechanical. **Files:** every `src/` module annotating `PoseEstimation` (≈56, incl. `project/{video_labels,track_labels,prediction_manager,video_manager}.py`, `classifier/cv_postprocessing.py`, `overlay_drawing/*`, `video_export/*`, `ui/player_widget/*`, `ui/viewer/*`, `feature_extraction/**`) and test fakes (`tests/video_export/_fakes.py`, `tests/feature_extraction/**/conftest.py`, ...).

- [ ] **Step 1:** Annotate with `Pose` (under `TYPE_CHECKING`); fakes implement the `Pose` surface; `isinstance(..., PoseEstimationV6)` checks replaced by capabilities.
- [ ] **Step 2:** Acceptance: `grep -rnwE "PoseEstimation(V[2-8])?" src --include='*.py'` returns only `src/jabs/pose_estimation/` and its `__init__`; full suite → PASS.
- [ ] **Step 3:** Commit `refactor: annotate poses as Pose`.

### Task 14: Flip `open_pose_file` and delete the legacy readers

**Files:** `src/jabs/pose_estimation/__init__.py`; delete `src/jabs/pose_estimation/pose_est_v{2..8}.py`, `identity_indexed_pose.py`, their tests; `packages/jabs-core/src/jabs/core/abstract/{pose_est,__init__}.py` (shim); keep `jabs.core.enums.JabsPoseVersion` (used by jabs-io `pose/hdf5.py` and jabs-vision); test `packages/jabs-core/tests/test_pose_est_deprecated.py`.

**Interfaces:** `open_pose_file = open_unified` semantics (`open_pose_file(path, cache_dir=None) -> Pose`). `jabs.core.abstract.pose_est` keeps `MINIMUM_CONFIDENCE` and a `PoseEstimation` class carrying only the skeleton constants; importing `PoseEstimation` from either `jabs.core.abstract` or `jabs.core.abstract.pose_est` emits `DeprecationWarning` via module `__getattr__` in both modules (the eager import in `abstract/__init__.py` is removed).

- [ ] **Step 1:** Tests: deprecation warning from both import paths; `open_pose_file` on legacy and unified samples returns `Pose`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement and delete. Acceptance: `grep -rnwE "PoseEstimation(V[2-8])?|identity_indexed_pose|format_major_version|get_pose_file_major_version" src packages --include='*.py'` returns only the shim and its test.
- [ ] **Step 4:** Full suite + all package suites → PASS.
- [ ] **Step 5:** Commit `refactor(pose): open every pose file as Pose; delete per-version readers`.

### Task 15: FEATURE_VERSION 20, classifier feature versions, end to end, docs

**Files:** `src/jabs/feature_extraction/features.py:39`; `src/jabs/classifier/{base,classifier,multi_class_classifier}.py`; `src/jabs/project/project.py:927-931,1837`; `src/jabs/ui/main_window/central_widget.py:2284-2288`; `src/jabs/resources/docs/user_guide/{features,file-formats,project-setup,viewer}.md`; `CLAUDE.md` (pose_estimation line and "Adding a new feature" `_min_pose` → `_requires`); `docs/development/development.md`; tests `tests/classifier/test_feature_version.py`, `tests/end_to_end/test_pose_file_projects.py`.

**Interfaces:**
- `FEATURE_VERSION = 20`, changelog paragraph in `features.md` citing spec §2.
- `ClassifierFeatureVersionError(ValueError)` in `jabs.classifier`. `_feature_version` is added to `_PERSISTED_REQUIRED`-style handling (persisted, copied by `load`). `train` stamps the training data's feature version (project features: `FEATURE_VERSION`; exported data: its `file_version`), and refuses exported data whose `file_version != FEATURE_VERSION` ("re-export the training data"). `from_pickle`/`load` raise the error when `_feature_version` is missing or differs (lazy import of `FEATURE_VERSION`).
- `central_widget` shows a `MessageDialog` with the message; `Project.rename_behavior` skips (and logs) a classifier that raises it; `Project.load_classifier` catches it alongside `OSError`.

- [ ] **Step 1:** Tests: old pickle refused (`retrain`); mismatched refused; round trip keeps `_feature_version`; v19 export refused for training; rename with a stale classifier succeeds and logs. End to end with a fixture writing a tiny video (`cv2.VideoWriter`, frames = pose frames) beside copied samples: legacy-only (v4), unified-only (v4 converted and renamed `_pose.h5`), mixed; each: open → features → train on synthetic labels → predict; legacy and unified projects give identical features for the same video; mixed project per-frame column names equal across videos.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement + docs. **Step 4:** Full suite + all package suites + ruff → PASS.
- [ ] **Step 5:** Commit `feat: feature version 20 on the unified pose file; classifiers record feature versions`.
