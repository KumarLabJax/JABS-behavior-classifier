# BALE (Bout & Label Explorer)

**BALE (Bout & Label Explorer)** is an interactive companion window built into JABS that helps you audit manual labels, review classifier predictions, discover kinematic sub-types, and screen whole videos across an entire project without leaving the GUI.

BALE opens as a non-modal side-by-side window alongside the main JABS player. Clicking any bout or video in BALE immediately jumps the main JABS video player, active animal identity, and timeline selection to that exact frame span. Manual navigation in the main JABS window never overwrites your current triage queue in BALE.

## Quick Start

1. Open any Binary or Multi-Class JABS project.
2. Open BALE from the **Tools → BALE (Bout & Label Explorer)…** menu, or press **`Ctrl+Shift+B`** (**`Cmd+Shift+B`** on macOS).
3. Select the behavior you want to inspect in the top bar (or **All Classes (Multi-Class)** in Multi-Class projects).
4. Pick one of the four workflow modes in the top segmented bar:
   - **1. Audit Labels** — find likely mislabeled bouts and sloppy start/end boundaries in your manual annotations.
   - **2. Review Predictions** — inspect uncertain, surprising, or outlier predicted bouts and promote them directly into training labels.
   - **3. Sub-Types** — cluster bouts by posture and movement kinematics to inspect behavioral variants and explore the interactive 2D map.
   - **4. Videos** — screen whole videos across the project by total behavior time, bout count, zero-behavior status, or cohort-level outlier buckets (`E1–E5`).
5. Click any row (or press **`Space`** to play/stop the clip in the main JABS window) and use the right-hand **Single-Bout Inspector** or single-key shortcuts (**`Z`**, **`C`**, **`X`**, **`V`**, **`1`**, **`2`**, **`T`**) to confirm, flip, trim, split, promote, or delete labels directly on disk.

Every edit made in BALE immediately updates the project's annotation JSON files, refreshes JABS's timeline and label counts, preserves single-step **`Ctrl+Z` / `Cmd+Z` Undo**, and guards against dropping below JABS's dual-class cross-validation track threshold.

---

## Mode 1: Audit Labels

Use **1. Audit Labels** whenever you want to clean up an annotation set before training or after importing labels.

- **Mislabel & Confusion Ranking:** BALE computes leave-one-animal-out cross-validated predictions and k-nearest-neighbor agreement across all labeled bouts in the project. Bouts whose movement profile strongly resembles the opposite class (or a different behavior in Multi-Class mode) rise to the top of the table.
- **Boundary Slop Detection:** BALE flags chimeric bouts whose opening or closing frames diverge sharply from the bout core and suggests tighter start/end trim bounds.
- **Fast Single-Key Triage:**
  - **`1` (Keep):** Mark the bout as verified/examined (persisted in `jabs/session/bale_reviewed.json`), or apply the selected target class in Multi-Class mode.
  - **`2` (Flip):** Switch Behavior ↔ Not-Behavior (or switch to `None` / target class in Multi-Class mode).
  - **`Z` / `C`:** Explicitly label the bout as **Behavior** (`Z`, matching JABS's main window) or **Not-Behavior / None** (`C`).
  - **`T` (Trim):** Apply the suggested boundary trim (or use the **Split** button in the Inspector to split a chimeric bout at the divergence point).
  - **`X` / `Delete` / `Backspace`:** Remove the labeled bout cleanly.
  - **`V` (Examined):** Toggle the reviewed/examined checkmark without modifying annotations.
- **Dual-Class CV Guard:** The header badge displays how many animal tracks currently have both positive and negative labels (`>= 10` frames each). If a relabel or delete would break a dual-class track, BALE warns you before applying the edit.

---

## Mode 2: Review Predictions

Use **2. Review Predictions** after running **Classify** in JABS to rapidly turn informative predictions into round-2 training labels.

- **Active-Learning Queue:** BALE extracts predicted bouts from `jabs/predictions/*.h5` (using postprocessed predictions when available) and scores each bout by model uncertainty (confidence near the decision boundary), novelty (feature distance from existing labeled bouts), and k-NN disagreement.
- **Direct Label Promotion:**
  - **`1` or `Z` (Confirm Positive / Add as Behavior):** Promote the predicted bout directly into a verified positive training label.
  - **`2` or `C` (Mark Not-Behavior / None):** Promote a false positive directly into a hard-negative training label.
  - **`V` (Mark Examined):** Record the bout as reviewed without modifying annotations.
- **No Human-Label Leakage:** Mode 2 isolates predicted bouts so already-labeled human bouts do not clutter your active-learning review queue.

---

## Mode 3: Sub-Types

Use **3. Sub-Types** to explore behavioral diversity, check whether a behavior contains distinct biological variants, and inspect how bouts cluster in feature space.

- **Interactive 2D Map:** Zoom with the mouse wheel (or **`+` / `-` / `0`**), pan by dragging, press **`M`** to expand/collapse the map, click any bout point to jump JABS directly to that video and animal, press **`N`** to select the 15 nearest similar bouts, or box/lasso-select a cohort of bouts to filter the table or batch-triage them together.
- **Scopes & Feature Lenses:** Choose **Positive Only** or **All Labeled (Pos + Neg)** and switch cleanly between six lenses:
  - **Intrinsic Kinematics**,
  - **Locomotion & Turn**,
  - **Core Body Posture**,
  - **Social & Huddle**,
  - **Classifier-Weighted** (weighted by saved classifier feature importance), and
  - **RF Leaf Proximity** (Random Forest co-leaf similarity when a Random Forest model is saved).
- **Automatic Sub-Type Clustering:** BALE selects the optimal cluster count `k` via silhouette score (**Auto (2–6)**, or override `k = 2..6` directly) and summarizes the top distinguishing kinematic features and cage skew for each cluster.

---

## Mode 4: Videos

Use **4. Videos** to jump quickly across large projects by whole-video behavior prevalence and cohort outlier criteria instead of stepping through one bout at a time.

- **Whole-Video Summary Table:** Scans every video in the project from **Predictions** or **Human Labels** (with zero Parquet I/O) and displays total behavior frames, `%` of valid video frames, bout count, longest bout duration, confidence/near-miss counts, and outlier tags.
- **Quick Filter Presets (`E1–E5`):** Filter immediately with:
  - **All Videos**,
  - **50k Outliers (E1–E4)** (any statistical extreme or disagreement bucket),
  - **E1 • Most Frames (%)** and **E1 • Most Bouts**,
  - **E2 • Longest Bout**,
  - **E3a • Low-Conf Beh** (weak/uncertain positive predictions),
  - **E3b • Near-Miss None** (low-confidence Not-Behavior spans),
  - **E4 • Fragile / Disagree** (high raw-vs-postprocessed fragility or label disagreement),
  - **E5 • Zero Behavior** and **E5 • Zero (High-Conf)** (videos with zero detected/labeled bouts),
  - **Unlabeled w/ Behavior** and **Labeled Only**, plus an **Unlabeled Only** toggle and search filter.
- **Direct Video Jump & Queue Sync:** Clicking a video row jumps JABS directly to that video (and to its focal bout when present, or frame `0` for zero-behavior videos). You can also push the filtered video or bout queue into JABS's top **Behavior Search Bar** (`Send to JABS Search`) for sequential stepping.

---

## Arena Spatial Map (`🗺 Arena [A]`)

Click **`🗺 Arena [A]`** in the 2D map toolbar (or press **`A`**, or select **Options → 2D Map Projection → Arena (Spatial Position)**) in Modes 1–3 to switch the 2D scatter map directly into physical **top-down arena coordinates**:

- **Spatial Localization:** Plots each bout at the animal's median body centroid `(x, y)` over a representative grayscale arena backdrop (when videos are present) alongside cage perimeter corners, interior zone border, food hopper, and lixit landmarks from the pose files.
- **Spatial Box / Lasso Filtering & Triage:** Drag a rectangle or lasso around a specific arena region (for example, the food hopper, water shroud, wheel nub, or cage interior) to isolate and audit every bout that occurred there. Press **`A`** or click **`🗺 Arena [A]`** again to switch back to your feature projection (`PCA`, `MDS`, or `t-SNE`).

---

## Right-Hand Single-Bout Inspector

Selecting any bout opens its detailed diagnostic readout in the right-hand pane:

- **Frame-by-Frame Probability & Feature Plot:** Click anywhere along the trace to scrub the main JABS video player to that exact frame while keeping the bout's timeline selection intact.
- **Top Feature Deviations:** Shows the specific posture and motion features that make the selected bout stand out relative to typical positive and negative examples.
- **Local Trajectory / Spatial Context:** Visualizes the animal's path across the bout span (including static arena landmarks when available in the pose file).
- **Clip Playback (`Space`):** Plays the bout in the main JABS player with a short pre/post context buffer and toggles off cleanly on a second press.

---

## Exporting Results

- **Bout Modes (Modes 1–3):** Open the **Export / More** menu and choose **Export CSV + Summary Figure (PNG & SVG)…** to save both the full per-bout metrics CSV (video, identity, frame span, duration, label/prediction status, confusion/doubt scores, cluster assignment, and 2D coordinates) and a high-resolution summary figure in `.png` and editable vector `.svg`.
- **Videos Mode (Mode 4):** Click **Export CSV** in the Mode 4 toolbar to export the visible whole-video screening table (preserving your current filter and sort order) and `E1–E5` outlier buckets.

---

## Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| **`Ctrl+Shift+B`** / **`Cmd+Shift+B`** | Open or focus BALE from the main JABS window |
| **`Space`** | Play / stop the selected bout clip (with context frames) in JABS |
| **`Up`** / **`K`** and **`Down`** / **`J`** | Move to the previous / next row in the current table |
| **`Enter`** | Jump and sync the selected row to the main JABS player |
| **`Z`** | Label selected bout as **Behavior** (Modes 1–3) |
| **`C`** | Label selected bout as **Not-Behavior / None** (Modes 1–3) |
| **`X`** / **`Delete`** / **`Backspace`** | Delete / clear selected labeled bout (Modes 1–3) |
| **`V`** | Toggle **Examined / Reviewed** status (Modes 1–3) |
| **`1`** | Primary action: Keep verified label (Mode 1) or Confirm Positive (Mode 2) |
| **`2`** | Secondary action: Flip class (Mode 1) or Mark Not-Behavior / None (Mode 2) |
| **`T`** | Apply suggested boundary trim (when flagged) |
| **`A`** | Toggle **Arena Spatial Map** vs. feature projection (Modes 1–3) |
| **`N`** | Select the 15 nearest similar bouts in feature space |
| **`M`** | Expand or collapse the 2D bout map |
| **`+`** / **`-`** / **`0`** | Zoom in, zoom out, or reset the 2D map view |
| **`Esc`** | Clear box/lasso selection, collapse expanded map, or reset map zoom |
| **`R`** | Refresh analysis (or whole-video summary in Mode 4) |
| **`Ctrl+Z`** / **`Cmd+Z`** | Undo the most recent BALE label edit (Modes 1–3) |
