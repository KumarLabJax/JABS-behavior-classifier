# Design Proposal / RFC: `#Hashtag` Video Comments, In-Track Manual Markers & Controlled Vocabulary

- **Status:** Draft / Request for Comments (`@gbeane`, `@bergsalex`)
- **Authors:** Vivek Kumar (with Antigravity)
- **Repository:** `jabs-behavior-classifier`
- **Scope:** Phase-1 design proposal & UI mockups only (no code or test changes in this PR). Aligns with [`jabs-hub-backed-projects-plan.md`](jabs-hub-backed-projects-plan.md), [`timeline_annotations.py`](../../../src/jabs/project/timeline_annotations.py), [`BehaviorTimelineWidget`](../../../src/jabs/ui/behavior_timeline/behavior_timeline_widget.py), and [`jabs.classifier.mlflow_logging`](../../../src/jabs/classifier/mlflow_logging.py).

---

## 1. Motivation & Concrete Gaps in Existing `TimelineAnnotations`

During behavioral labeling, active-learning review (BALE), and large-cohort screening, researchers routinely spot phenotypes and QC issues that do **not** belong to the currently open classifier:

- *Cross-behavior & health observations:* spotting `#shake`, `#sick`, `#lorr`, or `#scratch` while labeling `Grooming` or `Jumping`.
- *Pose & video QC issues:* `#id_swap`, `#missing_keypoints`, `#cage_glare`.
- *Exemplar & borderline curation:* flagging crisp `#exemplar` clips with free-form rationale.

JABS already stores interval annotations under `"annotations": [...]` in `jabs/annotations/<video>.json` ([`timeline_annotations.py`](../../../src/jabs/project/timeline_annotations.py), [`annotation_edit_dialog.py`](../../../src/jabs/ui/dialogs/annotation_edit_dialog.py), [`annotation_overlay.py`](../../../src/jabs/ui/player_widget/overlays/annotation_overlay.py), [`behavior_search_util.py`](../../../src/jabs/behavior_search/behavior_search_util.py)). Codebase inspection identifies six concrete blockers to address before building `#hashtag` commenting on top:

1. **Silent metadata stripping on round-trip (`TimelineAnnotations` & `VideoLabels.merge`):** `TimelineAnnotations.Annotation` (`timeline_annotations.py:79-90`) and `load()`/`serialize()` discard any keys beyond the 7 built-in fields, and `VideoLabels.merge()` (`video_labels.py:305-322`) ignores `other._annotations`. Unpatched saves, `archive_behavior()`, `jabs-cli update-pose`, or `jabs-cli merge` drop extra fields or whole comment lists.
2. **No UUID keying & tagless rejection:** Edit/delete/dedup currently key on `(start, end, tag.lower(), identity_index)` (`_interval_matches`), causing multiple notes on the same span to collide, and `TimelineAnnotations.load()` drops entries whose `tag` is empty rather than falling back cleanly.
3. **Rigid single tag + broken multi-tag search (`C4`):** `TimelineAnnotationSearchQuery` (`behavior_search_util.py:229-258`) checks only `annotation.get("tag") == timeline_query.tag` (case-sensitive), `_GatherTimelineAnnotationTagsWorker` (`behavior_search_dialog.py:43-44`) only gathers top-level `tag`, and `_sorted_search_results()` (`behavior_search_util.py:287`) fails `int(None)` sorting when whole-video (`identity=None`) hits appear. Clicking `#sick` in `"#shake ... mouse is #sick"` returns `0` results today.
4. **Selection-mode gate, single-frame `TypeError`, & dropped `#` key (`C6`):** `_timeline_annotation_button` (`main_control_widget.py:365`) is disabled unless Select Mode is active; `_on_timeline_annotation_button_clicked()` (`central_widget.py:2578`) raises `TypeError` on `min(self._selection_start, self._curr_frame_index)` if `_selection_start is None`, ignores `_curr_selection_end`, crashes when `_pose_est is None`, and double-calls `save_annotations()`. `MainWindow.keyPressEvent()` (`main_window.py:163-177`) drops `Qt.Key.Key_NumberSign`, and `Shift+#` is US-layout-specific.
5. **Top-edge timeline paint collision & `identity=None` gap (`C5`):** `LabelDetailBar._draw_second_ticks()` (`label_detail_bar.py:51, 294-299`) already paints `4 px` ticks at `y = 0..4` along the top edge, colliding directly with top-edge comment marks. `BehaviorTimelineWidget` (`behavior_timeline_widget.py:651-659`) only draws per-identity `LabelTrackWidget`s and hides them in `ViewMode.PREDICTIONS`.
6. **No controlled vocabulary (`#sha` → `#shake`):** Without prefix completion and alias canonicalization (`#wds` → `#shake`), free-typed tags fracture downstream queries.

---

## 2. Core UX Design (Zero Extra Tracks, Bottom `1s` Ticks, `#` Key & Live Links)

![Candidate GUI Mockups for In-Track #Hashtag Comments and Controlled Vocabulary](imgs/hashtag_gui_options_mockup.png)

### Key Interaction & Timeline Geometry Decisions

1. **Zero Extra Timeline Tracks — Collision-Free `Manual` Markers:**
   - **`LabelDetailBar` (`30 px` / `18 px` compact):** Relocate `_draw_second_ticks()` (`4 px`) from the top edge to the **bottom edge** (`y = bar_height - 4`), freeing the top `2–3 px` inside `_BORDER_COLOR` cleanly for comment stripes + start-edge `▾` ticks. Clamp `1`-frame point notes (`start == end`) to **`>= 3 px`** rendered width so point comments remain clickable at 1-hour zoom, and show a subtle hatch/count badge (`+N`) when overlapping notes share an interval.
   - **`LabelOverviewBar` (`6 px` bar + `2 px` padding):** Render subtle `2 px` marks inside top padding outside `_highlight_span` collision zones.
   - **Whole-Video (`identity=None`) & `ViewMode.PREDICTIONS` Coverage:** Add one compact **`6 px` Video Pins strip** on the timeline ruler exclusively for `identity=None` comments (and zero extra height per animal), keeping `#comment` indicators visible even when `ViewMode.PREDICTIONS` collapses manual label bars.
   - **Mouse Plumbing:** Pass `timeline_annotations` into `LabelDetailBar` / `LabelOverviewBar` and handle hover/click to open a **custom frameless `QFrame` popup card** (`QLabel` with `linkActivated` — since standard Qt `QToolTip` does not support clickable links) featuring clickable `#tag` links plus direct **Edit / Extend Span / Delete** buttons.
2. **Always-Enabled Entry (`Frame or Span` + `#` Key or `Add #Comment`):**
   - Rename `New Timeline Annotation` (`MainControlWidget`) to **`Add #Comment [#]`** and keep it enabled whenever a video is loaded (`central_widget.py` defaults `[curr_frame, curr_frame]` when Select Mode is off, handles `_curr_selection_end`, guards `_pose_est is None`, and removes the duplicate `save_annotations()` call).
   - Bind **`Qt.Key.Key_NumberSign`** (`#` without `ShiftModifier`, matching `Key_Question`) in `MainWindow.keyPressEvent()` so `#` works across US, UK, ISO, and AZERTY layouts.
   - **Inline Focus & Enter-Completes-First Contract:** Opening the inline `QLineEdit` temporarily acquires keyboard focus; typing `#sha` or alias `#wds` displays the controlled-vocab popup (`Tab` or `Enter` completes the highlighted `#tag` **first**, preserving cursor text after the token; `Enter` with no active popup saves the note; `Enter`/`Esc` explicitly returns keyboard focus to `MainWindow` so playback/labeling shortcuts `Z/X/C/Space/Arrows` work immediately).
3. **Controlled Vocabulary + Mixed Prose (`description`):**
   - Free-form prose and multiple inline `#hashtags` live naturally in `description`, e.g. `"#shake head shakes slightly & looks #sick"`.
   - Tagless observations are still allowed cleanly via fallback `tag = "note"` (`hashtags = []`).
   - Video viewport pills (`AnnotationOverlay`) cap display at **`<= 2` `#tag` pills (`+N`)** so long tag lists never obscure tracked mice.
4. **Live Clickable `#Tag` Links into Multi-Tag Search:**
   - Clicking `#shake` or `#sick` in any popup card, video overlay, or optional drawer executes normalized multi-tag matching in [`SearchBarWidget`](../../../src/jabs/ui/search_bar_widget.py) (`TimelineAnnotationSearchQuery` matching lowercase `hashtags[]` + `tag`), letting researchers step cleanly across every hit with `Previous` / `Next`.

### GUI Options Compared

- **Option A — In-Track `Manual` Markers + `#` Quick-Comment Bar *(Phase 1 Core)*:** Zero extra tracks and zero permanent drawers; inline `#` bar handles vocab completion, single-frame or span capture, clickable `QFrame` hover/edit/delete cards, and multi-tag Search.
- **Option B — Option A + Collapsible Live-Link Drawer *(Optional Phase 1b)*:** Adds a toggleable side drawer listing project/video comments with `#tag` filter chips and quick edit/delete controls.
- **Option C — Project `#Hashtag` Explorer *(Deferred to Phase 2 inside BALE)*:** In keeping with YAGNI, exploratory multi-video summaries and pose-feature embeddings belong in **BALE** (`jabs.label_explorer`) once real `#tag` volume exists — not in the JABS Main labeling GUI.

---

## 3. Minimal Lossless Data Schema & Architecture

![Phase-1 Lossless Data Architecture across Annotations JSON, project.json Vocab, JABS Hub D13/D21, and MLflow](imgs/hashtag_data_and_ecosystem_architecture.png)

### Tier 1 — Minimal Backward-Compatible `jabs/annotations/<video>.json`

To eliminate redundant drifting fields (`text` duplicating `description`, `kind` duplicating `start == end`, and per-comment `anchor_xy_px`/`pose_sha256_prefix`/`origin_project` that duplicate existing bbox IoU remapping in `update_pose.py:28-30` and Hub `poseHash`), Phase 1 keeps `"annotations"` entries minimal:

```json
{
  "id": "019360a4-8c2b-7f10-9a12-4d3e8b1f2041",
  "start": 740,
  "end": 768,
  "identity": 1,
  "external_identity": 1234,
  "description": "#shake head shakes slightly and looks like mouse is #sick",
  "hashtags": ["shake", "sick"],
  "tag": "shake",
  "color": "#1E40AF",
  "author": "vkumar",
  "created_at": "2026-11-25T00:52:10Z"
}
```

**Mandatory Phase-1 Persistence Hardening:**
- **`extras` passthrough (`TimelineAnnotations.Annotation` & `Interval.data`):** Load, hold, and re-serialize unknown keys (`id`, `hashtags`, `author`, `created_at`, future Hub metadata) untouched so unpatched clients or CLI commands never strip fields.
- **UUID-aware `VideoLabels.merge()`:** Union `other._annotations` alongside `_identity_labels` deduplicated by `id` (with `(start, end, tag.lower(), identity_index)` fallback only for legacy ID-less entries).
- **Primary `tag` validation & system QC filtering:** Derive `tag` cleanly on save (`hashtags[0]` sanitized to `[A-Za-z0-9_-]{1,32}`, fallback `"note"` when `hashtags == []`), resolve display `color` from vocabulary at render time, and filter system QC tags generated by `update-pose` (`remap-metric-disagreement`, `remap-split`) out of researcher comment feeds.

### Controlled Vocabulary inside `project.json["hashtag_vocabulary"]`

Rather than adding a loose file in `jabs/annotations/` (which breaks `update_labels.py:89-99` globs) or an un-merged `jabs/hashtag_vocabulary.json`, the controlled vocabulary lives under **`project.json["hashtag_vocabulary"]`**:

- Automatically travels with `project.json` exports and future **Hub `projects.settings`** sync.
- Supports optional read-only lab defaults merged with per-project additions/overrides, canonical lowercase normalization + alias mapping (`"wds": "shake"`), and identical `1..32` char `[A-Za-z0-9_-]` rules across autocomplete, annotation validation, and search.

### No Unowned Phase-1 Derived Caches (In-Memory Query `< 40 ms`)

Phase 1 **drops** both `<project>/jabs/hashtags.parquet` and `~/.jabs/hashtags_ledger.jsonl`: unowned sidecar caches go stale on `git pull`, `jabs-cli merge`, `prune`, or `update-pose`, and append-only home-directory ledgers resurrect deleted comments without tombstones while violating project isolation (`D13`). Querying `Project.annotation_store` directly in memory across ~500 videos takes `~15–40 ms`. If multi-project local scanning is needed later in Phase 2/3, an `mtime + size`-validated SQLite index under `.cache/` can be added safely.

---

## 4. JABS Hub (`D13` / `D21` / `§2.5`), MLflow & Docs Boundaries

- **JABS Hub *(Strict Per-Project Isolation First)*:** Per confirmed Hub decisions in [`jabs-hub-backed-projects-plan.md`](jabs-hub-backed-projects-plan.md), annotations are scoped strictly to **`project_video_id` (`D13`)** and Hub projects operate on `0`-based clips `[clip_start, clip_end]` of underlying library videos (`D21`). Cross-project comment surfacing is explicitly **deferred to post-Phase-3**, gated by **`project_members` authorization (`§2.5`)**, explicit **`[clip_start, clip_end]` frame offset projection**, and read-only display without leaking blinded study arms or `origin_project` across unauthorized projects.
- **MLflow *(Frozen Provenance Snapshot Only)*:** [`src/jabs/classifier/mlflow_logging.py`](../../../src/jabs/classifier/mlflow_logging.py) already archives **`annotations.zip`** (`ANNOTATIONS_ARCHIVE_NAME`), capturing exact per-video comments automatically on `--mlflow` runs. Phase 1 keeps MLflow untouched (no extra `hashtags.csv` duplicating free-form notes outside ACLs).
- **Documentation Sync:** Implementation PRs touching GUI buttons, shortcuts, or search must update `docs/user-guide/` + `src/jabs/resources/docs/user_guide/` (`labeling.md`, `keyboard-shortcuts.md`, `behavior-search.md`), `GUIDE_TREE_STRUCTURE`, and `zensical.toml`, verified via `tests/ui/test_user_guide_sync.py`.

---

## 5. Open Design Questions for Review (`@gbeane`, `@bergsalex`)

1. **`TimelineAnnotations` `extras` passthrough & `VideoLabels.merge()`:** Are you comfortable making `extras: dict[str, Any]` passthrough + `id`-keyed unioning in `VideoLabels.merge()` the foundational prerequisite commit before any `#hashtag` UI work?
2. **Relocating `LabelDetailBar._draw_second_ticks()` (`y = 0..4` → bottom edge):** Any objection to moving second ticks to the bottom edge of `LabelDetailBar`, reserving a slim `6 px` ruler strip for `identity=None` pins, and keeping `#comment` indicators visible in `ViewMode.PREDICTIONS`?
3. **Custom `QFrame` Hover/Edit Card & `#` Focus Contract:** Does a frameless `QFrame` popup card (`QLabel` `linkActivated` + Edit/Delete buttons) triggered from `Manual` marks + `Qt.Key.Key_NumberSign` (returning focus to `MainWindow` on close) fit cleanly with `CentralWidget`?
4. **Controlled Vocabulary in `project.json["hashtag_vocabulary"]`:** Does storing canonical tags, aliases (`#wds -> #shake`), and colors in `project.json` align cleanly with Hub `projects.settings` sync?
5. **Deferring Option C (BALE Summary) & Cross-Project Hub (`D13`/`D21`):** Confirmed good scope cut to ship Phase 1 (`Option A` + lossless JSON + multi-tag `SearchBarWidget`) first and defer BALE explorer (`Option C`) + ACL-gated Hub clip projection (`D21`) to Phase 2/3?
