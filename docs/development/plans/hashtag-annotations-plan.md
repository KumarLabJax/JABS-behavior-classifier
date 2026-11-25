# Design Proposal / RFC: `#Hashtag` Video Comments, In-Track Manual Markers & Controlled Vocabulary

- **Status:** Draft / Request for Comments (`@gbeane`, `@bergsalex`)
- **Authors:** Vivek Kumar (with Antigravity)
- **Repository:** `jabs-behavior-classifier`
- **Scope:** Design proposal & UI mockups only (no code changes in this PR). Aligns with [JABS Hub-backed projects plan](jabs-hub-backed-projects-plan.md), [`BehaviorTimelineWidget`](../../../src/jabs/ui/behavior_timeline/behavior_timeline_widget.py), and [`jabs.classifier.mlflow_logging`](../../../src/jabs/classifier/mlflow_logging.py).

---

## 1. Motivation & Gaps in Existing `TimelineAnnotations`

During labeling, active-learning review (BALE), and large-cohort screening, researchers routinely spot important phenotypes and QC issues that are **not** labels for the currently open classifier:

- *Cross-behavior & health observations:* spotting `#shake`, `#sick`, `#lorr`, or `#scratch` while labeling `Grooming` or `Jumping`.
- *Pose & video QC issues:* `#id_swap`, `#missing_keypoints`, `#cage_glare`.
- *Exemplar & ambiguity curation:* flagging crisp `#exemplar` clips or borderline bouts with nuance in free-form prose.

JABS already has an early interval annotation system ([`timeline_annotations.py`](../../../src/jabs/project/timeline_annotations.py), [`annotation_edit_dialog.py`](../../../src/jabs/ui/dialogs/annotation_edit_dialog.py), [`annotation_overlay.py`](../../../src/jabs/ui/player_widget/overlays/annotation_overlay.py), and [`TimelineAnnotationSearchQuery`](../../../src/jabs/behavior_search/behavior_search_util.py)) saved in `jabs/annotations/<video>.json` under `"annotations": [...]`. Four design limitations prevent fast `#hashtag` commentary today:

1. **Modal multi-step entry & no shortcut:** Users click `New Timeline Annotation` (`MainControlWidget`, [`main_control_widget.py`](../../../src/jabs/ui/main_control_widget/main_control_widget.py)) to open a modal form; there is no quick **`Shift+#`** (`#`) inline entry.
2. **Rigid single tag (`MAX_TAG_LEN = 32`, no free-form inline `#tags`):** Each entry carries a single alphanumeric tag and a separate unindexed description string — you cannot naturally type mixed prose with multiple live `#hashtags` such as:
   > `"#shake the head shakes slightly and looks like the mouse is #sick"`
3. **No controlled vocabulary / prefix completion:** Uncontrolled tags fracture downstream analysis (`#shake`, `#shakes`, `#head_shake`, `#wds`) instead of completing `#sha` → `#shake` cleanly from a curated dictionary.
4. **Trapped per project & unlinked:** Notes in Project A (`Grooming_Round3`) are invisible in Project B (`Shaking`) even for the same video, and `#tags` inside comments are plain static text rather than clickable live links into JABS Search.

---

## 2. Core UX Design (Zero Extra Timeline Tracks + Controlled Vocab + Live Links)

![Candidate GUI Mockups for In-Track #Hashtag Comments and Controlled Vocabulary](imgs/hashtag_gui_options_mockup.png)

### Key Interaction Decisions

1. **Zero Extra Timeline Tracks — Comment Marks Directly on the `Manual` Track:**
   - Vertical timeline height in `BehaviorTimelineWidget` is tight (especially in Multi-Class and All-Animals view). Adding another horizontal track row is undesirable.
   - Instead, comments render as **compact in-track markers along the top edge of JABS's existing `Manual` label bars**:
     - **`LabelDetailBar` (`30 px` / `18 px`):** `4 px` colored top-edge accent cap + start-edge `▾` notch along the top edge of the Manual bar (leaving underlying `Behavior` / `Not-Behavior` / `Unlabeled` colors unobstructed underneath). Hovering shows the rich comment tooltip with clickable `#tag` links; clicking opens/seeks the comment.
     - **`LabelOverviewBar` (`6 px` bar + `2 px` padding):** `2 px` top tick inside the existing padding — **`0 px` added vertical height**.
2. **Simple Entry (`Select Frames` + `Shift+#` or `Add #Comment` Button):**
   - Select a frame span (or pause on a single frame) and press **`Shift+#`** (`#`) or click **`Add #Comment`** (upgrading the existing `New Timeline Annotation` button in `MainControlWidget` — zero extra buttons).
   - Video, `[start, end]` frame span, and selected animal `identity` (with a 1-click `Whole Video` toggle) are attached automatically.
3. **Controlled Vocabulary + Inline Prefix Autocomplete (`#sha` → `#shake`):**
   - Typing `#` *anywhere* inside the comment text opens an inline suggestion popup driven by a **Controlled Vocabulary** (`hashtag_vocabulary.json` + Hub dictionary, with canonical lowercase tags, short definitions, aliases like `wds -> shake`, and display colors).
   - Typing `#sha` completes `#shake` (`Tab` / `Enter` / click), while casual prose around it remains free-form: `"#shake the head shakes slightly and looks like the mouse is #sick"`.
   - Unknown tags trigger a guard badge (`Not in Controlled Vocabulary`) with an explicit `+ Define "#tag" in Vocabulary…` action so typos never silently pollute lab analytics.
4. **Live Clickable `#Tag` Links:**
   - Every `#tag` inside comments (`#shake`, `#sick`, etc.) renders as a **live clickable link** in hover tooltips, overlays, and comment cards. Clicking `#shake` immediately runs `TimelineAnnotationSearchQuery(tag="shake")` in JABS's top [`SearchBarWidget`](../../../src/jabs/ui/search_bar_widget.py) so you can step across every `#shake` bout with `Previous` / `Next`.

### GUI Options Compared

- **Option A — In-Track `Manual` Markers + `Shift+#` Quick-Comment Bar *(Lowest Clutter)*:** Zero permanent drawers; floating input bar pops below the video when pressing `Shift+#` or clicking `Add #Comment`, shows inline controlled-vocab completion, marks the `Manual` track, and closes cleanly on `Enter`.
- **Option B — Option A + Collapsible Right-Side Live-Link Drawer *(Team Review)*:** Optional side drawer beside the player listing comment cards with `#tag` filter chips and clickable `#shake` / `#sick` live links wired straight to `SearchBarWidget`.
- **Option C — Controlled Vocabulary & Cross-Project Explorer *(Analytics / Seeding)*:** Manage canonical `#tags`, query across all projects (`hashtags.parquet` / Hub), verify kinematic clustering in pose feature space, and seed new JABS classifier projects. *(Can live inside BALE `Tools → BALE…` or as a dedicated dialog.)*

---

## 3. Data Structure for Cross-Project Reuse

![Cross-Project Storage Architecture across Manual-Track JSON, Controlled Vocab, Ledger, JABS Hub, and MLflow](imgs/hashtag_data_and_ecosystem_architecture.png)

### Tier 1 — Backward-Compatible `jabs/annotations/<video>.json` + `hashtag_vocabulary.json`

Keeping comments in `"annotations"` preserves [`AnnotationStore`](../../../packages/jabs-io/src/jabs/io/annotations/base.py), `AnnotationOverlay`, `TimelineAnnotationSearchQuery`, and [`jabs-hub-backed-projects-plan.md`](jabs-hub-backed-projects-plan.md) Appendix A.1 compatibility:

```json
{
  "id": "019360a4-8c2b-7f10-9a12-4d3e8b1f2041",
  "start": 740,
  "end": 768,
  "kind": "span",
  "identity": 1,
  "external_identity": 1234,
  "anchor_xy_px": [1842.5, 914.0],
  "text": "#shake the head shakes slightly and looks like the mouse is #sick",
  "hashtags": ["shake", "sick"],
  "tag": "shake",
  "description": "#shake the head shakes slightly and looks like the mouse is #sick",
  "color": "#0F766E",
  "author": "vkumar",
  "created_at": "2026-11-25T00:52:10Z",
  "origin_project": "Grooming_Round3_JABS_Project",
  "pose_sha256_prefix": "a84f91c2d01e"
}
```

And a tiny controlled vocabulary file (`<project>/jabs/hashtag_vocabulary.json` and optional shared lab/Hub dictionary):

```json
{
  "version": 1,
  "tags": {
    "shake": { "description": "Rapid head or body shake burst", "color": "#0F766E", "aliases": ["wds", "head_shake"] },
    "sick":  { "description": "Hunched/unwell posture or health concern", "color": "#B91C1C", "aliases": ["ill", "lethargic"] }
  }
}
```

Why these fields make cross-project reuse reliable:
- **Multi-tag + backward compatibility (`hashtags[]` + `tag` + `text`):** Supports multiple inline controlled `#tags` inside free-form prose while legacy single-tag readers still see `tag` (`"shake"`).
- **Pose re-tracking resilience (`anchor_xy_px`):** Physical `(x, y)` coordinates at `start` (plus `external_identity` when present) re-link the note to the right physical mouse even if track IDs (`0, 1, 2`) renumber in another pose version.
- **Deduplication (`id` UUID):** Safe merging across projects and ledgers.

### Tier 2 — Cross-Project Analytical Ledger (`hashtags.parquet`)

Updating a compact flat index (`<project>/jabs/hashtags.parquet`, plus optional shared lab ledger `~/.jabs/hashtags_ledger.jsonl`) avoids parsing hundreds of JSONs and enables `< 20 ms` pandas/R queries such as *"pull all `#shake` clips that do not also carry `#sick` across 20 projects"*.

---

## 4. JABS Hub & MLflow Integration

- **JABS Hub *(Primary Multi-User Sync)*:** Hub already plans shared library videos (`videos` table, many-to-many with `projects`) and per-video `annotations` sync. Hub can host both the lab-wide **Controlled Vocabulary** and per-video comments (with the open question below on project-scoped vs. shared library-video visibility).
- **MLflow *(Frozen Provenance Snapshot, NOT Live Query Store)*:** [`src/jabs/classifier/mlflow_logging.py`](../../../src/jabs/classifier/mlflow_logging.py) **already** uploads `annotations.zip` (`ANNOTATIONS_ARCHIVE_NAME`) on every `jabs-cli cross-validation --mlflow` run, so per-video comments travel automatically with the model snapshot. We can also attach a flat `hashtags.csv` artifact + `n_hashtag_notes` tag per CV run while keeping live cross-project querying in `hashtags.parquet` / JABS Hub.

---

## 5. Open Design Questions for Review (`@gbeane`, `@bergsalex`)

1. **In-track `Manual` markers (`LabelDetailBar` / `LabelOverviewBar`):** Does drawing a `4 px` colored top-edge accent cap + `▾` notch on existing Manual label bars cleanly avoid adding a new track while keeping `Behavior` / `Not-Behavior` colors legible? Any multi-class / compact-mode paint caveats?
2. **Schema in `"annotations"` vs. sibling key:** Is extending entries in `"annotations"` with optional `id`, `kind`, `hashtags`, `text`, `anchor_xy_px`, `author`, `created_at`, `origin_project` clean for `jabs.io`, NWB export, and JABS Hub Appendix A.1?
3. **Controlled vocabulary storage & Hub sync:** Should `hashtag_vocabulary.json` live at the project level with optional lab/Hub inheritance so `#shake` / `#sick` definitions stay consistent across collaborators?
4. **Hub visibility scope:** Should comments on a shared Hub library video stay project-scoped by default or allow opt-in cross-project surfacing by `video_id`?
5. **Upgrading `"New Timeline Annotation"` + `Shift+#`:** Any shortcut conflicts with `Shift+#` in `CentralWidget` / `MainControlWidget`, and do you prefer clicking a `#tag` link to trigger `SearchBarWidget` directly (`TimelineAnnotationSearchQuery`)?
6. **MLflow:** Alongside existing `annotations.zip`, is attaching a small `hashtags.csv` summary artifact on `jabs-cli cross-validation --mlflow` worthwhile?
