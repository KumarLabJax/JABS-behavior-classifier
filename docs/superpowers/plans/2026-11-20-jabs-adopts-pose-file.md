# JABS Adopts the Unified Pose File — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every pose JABS reads goes through one `Pose` class over the unified pose file; legacy v2–v8 files are converted into a project cache copy at open time.

**Architecture:** `open_pose_file` resolves a path to a unified file (directly, or by `convert_legacy` into `<cache_dir>/converted/`), and wraps it in `jabs.pose_estimation.pose.Pose`, which eagerly reads the small components with `read_component` and reads segmentation in frame windows. Version numbers disappear: features and UI ask `Pose` for capabilities. `FEATURE_VERSION` becomes 20 and classifiers record it.

**Tech Stack:** Python 3.10+, h5py, numpy, jabs-io `pose_file` (ADR 0002), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-20-jabs-adopts-pose-file-design.md`

## Global Constraints

- Run everything with `uv run`; tests: `uv run pytest`, `uv run pytest packages/jabs-io/tests`, `uv run pytest packages/jabs-core/tests`, `uv run pytest packages/jabs-vision/tests` (needs `uv sync --all-packages --all-extras --all-groups`).
- `uv run ruff check . && uv run ruff format --check .` clean after every task.
- Repo style (CLAUDE.md): full type hints, Google docstrings, `logger = logging.getLogger(__name__)`, lazy log formatting, no `print`, pytest functions with `monkeypatch`/`unittest.mock` (no `mocker`), no EN DASH in docstrings.
- Unified file pattern: `<video stem>_pose.h5`. Legacy pattern: `<video stem>_pose_est_v<N>.h5`, N in 2..8. Unified wins when both exist; among legacy, highest N wins.
- Conversion cache: `<cache_dir>/converted/<legacy-hash>-<jabs-io version>.h5`; `<legacy-hash>` is `jabs.core.utils.hash_file(legacy)`; jabs-io version is `importlib.metadata.version("jabs-io")`. Without a `cache_dir`, a `tempfile.TemporaryDirectory` owned by the `Pose` and removed by `Pose.close()`.
- `Pose.hash` is `hash_file(<the file the user has>)` (the legacy file for a conversion).
- Point validity is `jabs.pose.point_valid`; invalid points are NaN in returned arrays. Identity present is `jabs.pose.slot_usable` when present, else derived: `>= 3` valid keypoints among skeleton body parts other than `MID_TAIL` and `TIP_TAIL` (by name).
- Identities = slots `0..dimensions["identity"]-1`; when `dimensions["identity"] == 0`, every slot is an identity and `has_identity_tracking` is False.
- fps = manifest `video.fps`, else 30 with one INFO log per `Pose`.
- Static objects stored `(O, P, 2)` xy under `jabs.static_objects.<name>`; `Pose.static_objects[name]` squeezes P when P == 1.
- `FEATURE_VERSION = 20`. Classifiers store `_feature_version`; loading a classifier whose `_feature_version` is missing or differs raises `ValueError` whose message contains `retrain`.
- Capabilities (`jabs.core.enums.PoseCapability`, a `StrEnum`-style `str, Enum`): `SOCIAL` (slot count > 1), `SEGMENTATION` (`jabs.segmentation.contours` present), `BOUNDING_BOXES` (`jabs.pose.bbox` present), `IDENTITY_TRACKING` (`dimensions["identity"] > 0`). Static objects are checked by name, not by capability.
- `jabs.core.abstract.PoseEstimation` remains one release as a constants-only deprecated name (`DeprecationWarning` on attribute access through the module `__getattr__`).

## Review Focus

1. **A project mixing legacy and unified files** — capabilities intersect correctly and features have identical column sets across videos. Pinned by Task 9's end-to-end mixed-project test.
2. **A legacy file edited or replaced after conversion** — the new hash misses the cache and reconverts; a stale cache copy is never read. Pinned in Task 5.
3. **A converter refusal** (duplicate identity, shared tracklet) on a file JABS used to open — the project scan reports the file name and reason instead of crashing the GUI. Pinned in Task 7.
4. **An hour-long file with dense segmentation** — `Pose` never calls `read_pose_file` and never loads all contours; only windows. Pinned in Task 4 by asserting `read_pose_file` is not called (monkeypatch) and windows read the requested frames only.
5. **A v3 file (`identity = 0`) and a v2 file (1 slot)** — identities, `has_social` and feature availability match the rules above. Pinned in Tasks 2–3 conversions and Task 4.

---

### Task 1: Neutral home for the pose skeleton constants

**Files:**
- Create: `packages/jabs-core/src/jabs/core/pose_skeleton.py`, `packages/jabs-core/tests/test_pose_skeleton.py`
- Modify: `packages/jabs-core/src/jabs/core/enums/__init__.py` (+ new `pose_capability.py`), `packages/jabs-core/src/jabs/core/utils/pose_util.py`, `packages/jabs-io/src/jabs/io/internal/pose_file/skeletons.py:19`, `packages/jabs-vision/src/jabs/vision/cli/pose.py:11-27`, `packages/jabs-core/src/jabs/core/abstract/pose_est.py`

**Interfaces:**
- Produces: `jabs.core.pose_skeleton.KeypointIndex` (IntEnum, same members/values as `PoseEstimation.KeypointIndex`), `FULL_CONNECTED_SEGMENTS`, `MINIMUM_CONFIDENCE = 0.3`, `TAIL_KEYPOINTS = ("MID_TAIL", "TIP_TAIL")`; `jabs.core.enums.PoseCapability` with members `SOCIAL="social"`, `SEGMENTATION="segmentation"`, `BOUNDING_BOXES="bounding_boxes"`, `IDENTITY_TRACKING="identity_tracking"`; `jabs.core.utils.pose_util.PoseLike` Protocol (whatever `identity_centroids` uses).
- `PoseEstimation.KeypointIndex` / `.FULL_CONNECTED_SEGMENTS` become aliases of the new objects (the ABC itself is untouched until Task 8).

- [ ] **Step 1:** Tests: `test_keypoint_index_matches_legacy` (names and values equal `PoseEstimation.KeypointIndex`), `test_capability_values` (the four string values above), `test_identity_centroids_accepts_protocol` (a small fake object satisfying `PoseLike` works).
- [ ] **Step 2:** Run `uv run pytest packages/jabs-core/tests/test_pose_skeleton.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement; repoint jabs-io `skeletons.py`, jabs-vision `cli/pose.py`, `pose_util.py` imports to `jabs.core.pose_skeleton`.
- [ ] **Step 4:** Run jabs-core, jabs-io and jabs-vision suites → PASS.
- [ ] **Step 5:** Commit `refactor(core): move pose skeleton constants out of the PoseEstimation ABC`.

### Task 2: Converter accepts v2

**Files:**
- Modify: `packages/jabs-io/src/jabs/io/internal/pose_file/convert/{inspect,plan,sources,provenance,memory,cli}.py`
- Modify: `packages/jabs-io/tests/data/legacy-pose/generate.py` (add `v2-single.h5`)
- Test: `packages/jabs-io/tests/internal/pose_file/convert/test_convert_v2.py`

**Interfaces:**
- Consumes: `convert_legacy(source, destination, *, video=None, ...) -> ConversionReport` unchanged.
- Produces: `LegacyLayout` gains `instance_axis: bool` (False for v2). v2 output: `dimensions = {"frame": F, "slot": 1, "identity": 1}`, `slot_occupied = any(confidence > 0)` per frame, `point_valid = confidence > 0.3` (float64 compare, as for v4+), no tracklets, no segmentation, no objects. `source_format` reports `pose_est_v2`.

- [ ] **Step 1:** Generator fixture `v2-single.h5` (points `(F,12,2)` uint16 yx, confidence `(F,12)`), including frames with confidence in `(0, 0.3]` and an all-zero frame. Tests: converts and `validate(deep=True)` passes; points are xy-flipped legacy values; `point_valid` false exactly where `confidence <= 0.3`; `slot_occupied` false on the all-zero frame; `inspect_legacy` on repo-root `tests/data/sample_pose_est_v2.h5` (behind `requires_samples`) succeeds.
- [ ] **Step 2:** Run → FAIL ("not a pose_est_v4 or later file").
- [ ] **Step 3:** Implement: `_REQUIRED` splits into a v2 shape (`points`, `confidence` without instance axis) and the existing v4+ set; v2 is read as one instance in slot 0. Update docstrings/CLI help "v4–v8" → "v2–v8".
- [ ] **Step 4:** Run converter tests → PASS (existing v4–v8 tests unchanged).
- [ ] **Step 5:** Commit `feat(pose-file): convert legacy pose_est_v2`.

### Task 3: Converter accepts v3

**Files:** same modules as Task 2; fixture `v3-tracklets.h5`; test `convert/test_convert_v3.py`.

**Interfaces:**
- Produces: v3 output per ADR 0002: `identity = 0`, `slot = I` (legacy instance axis), each legacy instance stays in its slot, `jabs.pose.tracklet_id` from `instance_track_id`, `slot_occupied` from `instance_count` (`slot < instance_count[frame]`), `jabs.identity.embeddings` from `instance_embedding` when present.

- [ ] **Step 1:** Fixture with 2 instances, `instance_count` varying 0–2, a track id switch, confidences in `(0, 0.3]`. Tests: deep-valid; `dimensions["identity"] == 0`; occupancy follows `instance_count`; tracklet ids preserved; repo-root `sample_pose_est_v3.h5` and `identity_with_no_data_pose_est_v3.h5` convert (behind `requires_samples`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the v3 branch in `inspect`/`plan`/`sources`.
- [ ] **Step 4:** Run converter suite + `test_legacy_corpus.py` → PASS.
- [ ] **Step 5:** Commit `feat(pose-file): convert legacy pose_est_v3`.

### Task 4: The `Pose` class

**Files:**
- Create: `src/jabs/pose_estimation/pose.py`, `tests/pose_estimation/test_pose.py`, `tests/pose_estimation/conftest.py` (builders that write small unified files with `write_pose_file`)

**Interfaces:**
- Consumes: `read_manifest(path)`, `read_component(path, component_id, frames=None, *, manifest=...)` (jabs-io reader; dense only), `PoseCapability`, `KeypointIndex`, `TAIL_KEYPOINTS`, `hash_file`.
- Produces `class Pose`:
  - `Pose(pose_path: Path, *, source_path: Path | None = None, cache_dir: Path | None = None, owned_dir: tempfile.TemporaryDirectory | None = None)` — `pose_path` is the unified file read; `source_path` is the file the user has (defaults to `pose_path`); `close()` cleans `owned_dir`; context manager.
  - Properties with today's names/contracts: `num_frames`, `identities`, `num_identities`, `fps` (float), `cm_per_pixel` (`float | None`), `pose_file` (= `source_path`), `hash`, `identity_to_track`, `static_objects` (`dict[str, NDArray]`), `dynamic_objects`, `num_lixit_keypoints`, `external_identities`.
  - Methods with today's contracts: `get_points(frame_index, identity, scale=None) -> (points (12,2), mask (12,))`, `get_identity_poses(identity, scale=None) -> (points (F,12,2), mask (F,12))`, `get_identity_point_mask(identity)`, `get_reduced_point_mask()`, `get_connected_segments()`, `identity_mask(identity) -> NDArray[bool] (F,)`, `get_dynamic_object(name)`, `compute_bearing`, `compute_all_bearings`, `identity_index_to_display`, `get_bounding_boxes(identity) -> NDArray | None`, `get_identity_convex_hulls(identity)`.
  - New: `capabilities: frozenset[PoseCapability]`, `has_segmentation`, `has_bounding_boxes`, `has_social`, `has_identity_tracking`, `has_static_object(name) -> bool`, `segmentation_window(identity: int, start: int, stop: int) -> tuple[NDArray[np.int32], NDArray[np.uint32]]` returning contours `(stop-start, C, P, 2)` padded `-1` and lengths `(stop-start, C)`.
  - Convex hull cache at `<cache_dir>/convex_hulls/<pose_file_stem(source_path)>/convex_hulls_<identity>.pickle` — the same base name `project.get_derived_file_paths` deletes (fix its stem logic in this task if it differs: use the video base for both).
- Copy the bearing/convex-hull/display algorithms from the ABC; do not re-derive them.

- [ ] **Step 1:** Tests (each on a tiny written unified file):
  - `test_invalid_points_are_nan` — a point with `point_valid` false (coords stored) returns NaN, mask False.
  - `test_identity_mask_uses_slot_usable_when_present` and `test_identity_mask_derived_excludes_tail` (2 body + 2 tail valid → False; 3 body → True).
  - `test_tail_slots_not_identities` (`slot=3, identity=2` → `identities == [0, 1]`).
  - `test_identity_zero_treats_slots_as_identities` (`has_identity_tracking` False).
  - `test_fps_from_manifest` / `test_fps_defaults_to_30`.
  - `test_static_object_squeezes_single_point` (corners `(4,1,2)` → `(4,2)`; lixit `(1,3,2)` unchanged).
  - `test_capabilities` for seg/bbox/social presence and absence.
  - `test_segmentation_window_reads_only_window` (window values equal the slice; `read_pose_file` monkeypatched to raise, never called).
  - `test_hash_is_source_file_hash`, `test_scale_converts_to_cm`.
- [ ] **Step 2:** Run `uv run pytest tests/pose_estimation/test_pose.py -v` → FAIL.
- [ ] **Step 3:** Implement. Eager components via `read_component(..., manifest=m)`: points, point_valid, slot_occupied, slot_usable (optional), bbox (optional), tracklet_id (optional), external_ids (optional), static/dynamic objects. Segmentation only via `segmentation_window`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(pose): one Pose class over the unified pose file`.

### Task 5: Opening, discovery and the conversion cache

**Files:**
- Modify: `src/jabs/pose_estimation/__init__.py`, `packages/jabs-core/src/jabs/core/utils/utilities.py:69-80` (`pose_file_stem`)
- Create: `src/jabs/pose_estimation/discovery.py`, `src/jabs/pose_estimation/conversion_cache.py`
- Test: `tests/pose_estimation/test_discovery.py`, `tests/pose_estimation/test_conversion_cache.py`, `tests/pose_estimation/test_open_pose_file.py`

**Interfaces:**
- Produces:
  - `find_pose_file(video_path: Path, pose_dir: Path | None = None) -> Path | None` and `get_pose_path(video_path, pose_dir=None) -> Path` (raises `ValueError("Video does not have pose file")` as today) built on it; `is_pose_file(path: Path) -> bool`; `POSE_FILE_GLOBS = ("*_pose.h5", "*_pose_est_v*.h5")`.
  - `pose_file_stem` strips both `_pose` and `_pose_est_v\d+`.
  - `converted_path(source: Path, cache_dir: Path) -> Path` and `ensure_converted(source: Path, cache_dir: Path) -> Path` (converts with `convert_legacy` if the keyed file is missing; returns the path).
  - `open_pose_file(path: Path, cache_dir: Path | None = None) -> Pose`; unknown names raise `PoseFileError` (new, in `jabs.pose_estimation`) naming both patterns; `LegacyConversionError` propagates wrapped in `PoseFileError` with the original reason.

- [ ] **Step 1:** Tests: unified wins over legacy; highest legacy wins; `_pose.h5` stem; cache key changes when the legacy bytes change (Review Focus 2) and when the jabs-io version changes (monkeypatch `importlib.metadata.version`); second open reuses the cached file (`convert_legacy` called once); no `cache_dir` → temp dir removed after `close()`; legacy `sample_pose_est_v4.h5` opens with `Pose.pose_file` equal to the legacy path and `hash` equal to its `hash_file`; a v4 file with zero identities (generator fixture) opens with `identities == []` instead of crashing.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. `open_pose_file` no longer returns legacy classes (they are deleted in Task 8; until then nothing else changes).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(pose): open legacy files through a keyed conversion cache`.

### Task 6: Capabilities replace pose versions in features

**Files:**
- Modify: `src/jabs/feature_extraction/feature_base_class.py:73,148-163`, `feature_group_base_class.py:128-141`, `features.py:163-191,882-906`, `landmark_features/{lixit,food_hopper,corner,landmark_group}.py`, `social_features/{closest_fov_angles,pairwise_social_distances,closest_fov_distances,huddle_features,closest_distances,social_distance}.py`, segmentation feature group
- Test: `tests/feature_extraction/test_capabilities.py` + update existing feature tests that build `PoseEstimation` fakes

**Interfaces:**
- `Feature._requires: ClassVar[frozenset[PoseCapability]] = frozenset()` replaces `_min_pose` (`_min_pose = 3` → `{SOCIAL}`; `= 5` → `frozenset()` plus its existing `_static_objects`; segmentation features → `{SEGMENTATION}`).
- `Feature.is_supported(capabilities: frozenset[PoseCapability], static_objects: Iterable[str], **kwargs) -> bool`; `get_supported_feature_modules(capabilities, static_objects, **kwargs)`; `get_available_extended_features(capabilities, static_objects, **kwargs)`.
- `IdentityFeatures` uses `pose_est.capabilities` and passes `pose_est.fps` into features (it is stored at `features.py:165`; make it the value features read for per-second quantities).

- [ ] **Step 1:** Tests: a social feature is unsupported without `SOCIAL`; lixit feature supported with `lixit` present regardless of capability; segmentation group skipped without `SEGMENTATION`; a velocity feature computed at fps 60 is twice the fps 30 value for the same pixel motion.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement; grep `_min_pose\|format_major_version\|pose_version` under `src/jabs/feature_extraction` must return nothing.
- [ ] **Step 4:** Run `uv run pytest tests/feature_extraction` → PASS.
- [ ] **Step 5:** Commit `refactor(features): gate features on pose capabilities, read fps from the pose`.

### Task 7: Project, scan, training exports and CLI on `Pose`

**Files:**
- Modify: `src/jabs/project/{feature_manager,project,parallel_workers,project_paths,project_merge,video_manager,export_training,read_training}.py`, `src/jabs/scripts/{classify,initialize_project,generate_features}.py`, `src/jabs/scripts/cli/{cli,compute_features,export_video,convert_to_nwb,update_pose,update_labels,sample_pose_intervals,evaluate,convert_parquet}.py`
- Test: matching files under `tests/project/`, `tests/scripts/`

**Interfaces:**
- `FeatureManager.capabilities: frozenset[PoseCapability]` (intersection over videos) replaces `min_pose_version`; `Project.settings_by_capabilities(capabilities, static_objects, ...)` replaces `settings_by_pose_version`.
- The scan worker (`parallel_workers.py:106-196`) calls `open_pose_file(path, cache_dir)` (this is where first-open conversion happens) and reports frames, identities, static objects, lixit keypoints, `cm_per_pixel` and capabilities from `Pose`; a `PoseFileError` becomes a per-video scan error naming file and reason.
- Training export writes attr `pose_capabilities` (sorted list of values); `read_training` reads it, or maps an old `min_pose_version` (≥3 SOCIAL, ≥6 SEGMENTATION, ≥8 BOUNDING_BOXES).
- Every filename-pattern site listed in the spec uses `find_pose_file`/`is_pose_file`/`POSE_FILE_GLOBS`/`pose_file_stem`; `convert_parquet` keeps writing `_pose_est_v8.h5` (it produces legacy files; out of scope).
- `convert_to_nwb` metadata `pose_format_version` → `pose_source_format` from the file's provenance.

- [ ] **Step 1:** Tests: project with one legacy and one unified video gets the capability intersection (Review Focus 1); scan reports a converter refusal as an error naming the file (Review Focus 3; fixture `v8-duplicate-embed.h5`); old training export with `min_pose_version=6` reads as `{SOCIAL, SEGMENTATION}`; `update_pose` discovers `_pose.h5`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement; `grep -rn "format_major_version\|min_pose_version\|get_pose_file_major_version\|_pose_est_v" src/jabs --include='*.py'` returns only `convert_parquet`, `read_training`'s legacy mapping, and `discovery.py`.
- [ ] **Step 4:** Run `uv run pytest tests/project tests/scripts` → PASS.
- [ ] **Step 5:** Commit `refactor(project): use Pose capabilities and pose discovery everywhere`.

### Task 8: UI and overlays on `Pose`; delete the legacy readers

**Files:**
- Modify: `src/jabs/overlay_drawing/{bounding_box,segmentation}.py`, `src/jabs/ui/player_widget/overlays/bounding_box.py`, `src/jabs/ui/main_window/central_widget.py:555-563`, `src/jabs/ui/viewer/viewer_window.py:23,246,396-445`, `src/jabs/ui/dialogs/video_info_dialog.py:117-125`, `src/jabs/video_export/*`, `src/jabs/ui/behavior_timeline/*`
- Delete: `src/jabs/pose_estimation/pose_est_v{2..8}.py`, `identity_indexed_pose.py`, their tests under `tests/pose_estimation/`
- Modify: `packages/jabs-core/src/jabs/core/abstract/pose_est.py` (constants-only deprecated shim), `packages/jabs-core/src/jabs/core/abstract/__init__.py`, `packages/jabs-core/src/jabs/core/enums/pose_version.py` (keep; still used by `convert_parquet` if so, else delete)
- Test: `packages/jabs-core/tests/test_pose_est_deprecated.py`, UI tests touching overlays

**Interfaces:**
- Overlays check `pose.has_bounding_boxes` / `pose.has_segmentation`; segmentation overlay draws from `segmentation_window(identity, frame, frame+1)`.
- `jabs.core.abstract.PoseEstimation.KeypointIndex` still resolves and warns `DeprecationWarning`.
- Viewer file filter: `"Pose Files (*_pose.h5 *_pose_est_v*.h5);;All Files (*)"`.

- [ ] **Step 1:** Tests: deprecation warning test; segmentation overlay on a unified file draws the window contour; `video_info_dialog` reads static objects from `Pose`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement and delete. `grep -rn "PoseEstimationV\|identity_indexed_pose\|format_major_version" src packages --include='*.py'` returns nothing.
- [ ] **Step 4:** Run the full suite (root + all packages) → PASS.
- [ ] **Step 5:** Commit `refactor(pose): delete per-version pose readers`.

### Task 9: FEATURE_VERSION 20, classifier feature version, end to end

**Files:**
- Modify: `src/jabs/feature_extraction/features.py:39`, `src/jabs/classifier/{base,classifier,multi_class_classifier}.py`, `src/jabs/resources/docs/user_guide/{features,file-formats,project-setup,viewer}.md`, `CLAUDE.md` (pose_estimation line), `docs/development/development.md` pose section
- Test: `tests/classifier/test_feature_version.py`, `tests/end_to_end/test_pose_file_projects.py`

**Interfaces:**
- `FEATURE_VERSION = 20` with a changelog paragraph in `features.md` citing the spec §2 table.
- `train` sets `self._feature_version = FEATURE_VERSION`; `from_pickle` and `load` raise `ValueError("... trained on feature version X; this JABS computes feature version 20. Retrain this classifier.")` when it is missing or different. Import `FEATURE_VERSION` lazily inside those methods (import cycle).

- [ ] **Step 1:** Tests: old pickle (no attr) refused with `retrain`; mismatched version refused; fresh train/save/load round-trips. End to end (tmp project, copying repo-root samples): legacy-only project (v4 sample), unified-only (the v4 sample converted with `convert_legacy` and renamed `_pose.h5`), and mixed; each: open project → compute features → train on synthetic labels → predict; the legacy and unified projects produce identical features for the same video (same file content, two routes).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement and update docs.
- [ ] **Step 4:** Full suite + ruff → PASS.
- [ ] **Step 5:** Commit `feat: feature version 20 on the unified pose file; classifiers record it`.
