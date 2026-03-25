# JABS NWB Format

This document describes the NWB files produced by JABS. It covers the two output modes
(combined and per-identity), the full file layout, how animal pose, static objects, and
dynamic objects are stored, the `jabs_metadata` scratch field, and why the
`ndx-multisubjects` extension is not currently used.

JABS NWB files use the [ndx-pose 0.2](https://github.com/rly/ndx-pose) extension for
all pose and object data.

---

## Output modes

JABS can write NWB in two modes, selectable at export time.

### Combined file (default)

All identities from a single recording session are written into one NWB file. This is
the simplest output and potentially the most compatible with third-party NWB tooling.

```
session.nwb
  └── all identities, all objects
```

**When to use:** sharing data with collaborators, archiving, downstream analysis that
needs all animals in one place.

### Per-identity files

One NWB file is written per animal. The output path is used as a naming template; the
combined file is never created. Static and dynamic objects are written to every
per-identity file identically (they are session-level, not animal-level data).

```
session_subject_1.nwb   ← identity 0 + all objects
session_subject_2.nwb   ← identity 1 + all objects
session_subject_3.nwb   ← identity 2 + all objects
```

Identity names in the filenames come from `external_ids` in the pose file (sanitized
for HDF5 compatibility) or fall back to `subject_1`, `subject_2`, … when no external
IDs are present.

**When to use:** downstream workflows that require one NWB file per animal (e.g. tools
that expect a single `Subject` in each file).

#### Reading per-identity files

The JABS reader re-assembles per-identity files transparently. Point it at **any one**
sibling file; it detects the `per_identity_files` flag in `jabs_metadata`, globs for
siblings matching `{base_stem}_*.nwb` in the same directory, filters to those that
share the same `split_subject_count` count, sorts by `source_identity_index`, and
concatenates them into a single `PoseData` with all identities in their original order.

```
# read any sibling — result is identical
pose_data = load("session_subject_1.nwb", PoseData)
pose_data = load("session_subject_3.nwb", PoseData)
```

Validation ensures the expected number of sibling files are present before merging; a
`ValueError` is raised if any file is missing.

---

## Full NWB layout

The layout below shows a combined file containing two animal identities, two static
objects (`corners`, `lixit`), and one dynamic object (`fecal_boli`).

```
NWBFile
├── processing/
│   └── behavior/                          [ProcessingModule]
│       ├── Skeletons/                     [Skeletons container]
│       │   ├── subject/                   Skeleton — animal keypoints + edges
│       │   ├── corners/                   Skeleton — static object (4 nodes)
│       │   ├── lixit/                     Skeleton — static object (1 or 3 nodes)
│       │   └── fecal_boli/                Skeleton — dynamic object (max_count nodes)
│       │
│       ├── subject_1/                     [PoseEstimation] animal identity 0
│       │   ├── nose/                      [PoseEstimationSeries] num_frames timestamps
│       │   ├── left_ear/
│       │   └── ...
│       │
│       ├── subject_2/                     [PoseEstimation] animal identity 1
│       │   ├── nose/
│       │   └── ...
│       │
│       ├── corners/                       [PoseEstimation] static object
│       │   ├── corners_0/                 [PoseEstimationSeries] 1 timestamp
│       │   ├── corners_1/
│       │   ├── corners_2/
│       │   └── corners_3/
│       │
│       ├── lixit/                         [PoseEstimation] static object
│       │   └── lixit_0/                   [PoseEstimationSeries] 1 timestamp
│       │
│       ├── fecal_boli/                    [PoseEstimation] dynamic object
│       │   ├── fecal_boli_0/              [PoseEstimationSeries] n_predictions timestamps
│       │   ├── fecal_boli_1/
│       │   └── ...
│       │
│       ├── jabs_identity_mask             [TimeSeries] uint8 identity presence mask
│       ├── jabs_bounding_boxes_subject_1  [TimeSeries] optional, one per identity
│       └── jabs_bounding_boxes_subject_2  [TimeSeries] optional, one per identity
│
└── scratch/
    └── jabs_metadata/                     [ScratchData] JSON string (see below)
```

In a per-identity file the layout is identical, except only one animal identity
container is present and `jabs_identity_mask` / `jabs_bounding_boxes_<identity>` cover
that identity only.

---

## Animal pose

Each animal identity is a `PoseEstimation` container in `processing/behavior`. The
container name is the sanitized external ID from the pose file, or `subject_1`,
`subject_2`, … (1-based) when no external IDs are available.

A single `Skeleton` named `subject` (or overridden via `skeleton_name`) is shared by
all animal identities and stored in the `Skeletons` container.

### PoseEstimationSeries fields (per keypoint)

| Field                   | Value                                                                             |
|-------------------------|-----------------------------------------------------------------------------------|
| `name`                  | Keypoint name (e.g. `"nose"`, `"left_ear"`)                                       |
| `data`                  | shape `(num_frames, 2)` — `(x, y)` coordinates in pixels                         |
| `rate`                  | Frames per second (float)                                                         |
| `unit`                  | `"pixels"`                                                                        |
| `reference_frame`       | `"Top-left corner of video frame, x increases rightward, y increases downward"`  |
| `confidence`            | shape `(num_frames,)` — `0.0` = missing keypoint, `> 0.0` = valid               |
| `confidence_definition` | `"0.0=invalid/missing keypoint, >0.0=valid keypoint"`                            |

### Identity mask

`jabs_identity_mask` is a `TimeSeries` that records whether each identity is present in
each frame.

| Mode             | Shape stored in file           | Shape returned by reader         |
|------------------|-------------------------------|----------------------------------|
| Combined         | `(num_frames, num_identities)` | `(num_identities, num_frames)`   |
| Per-identity     | `(num_frames,)`                | `(1, num_frames)`                |

### Bounding boxes (optional)

When the pose file contains bounding box data, one `TimeSeries` per identity is written
with the name `jabs_bounding_boxes_{identity_name}`. This naming makes the containers
self-describing — no external index mapping is required.

| Property             | Value                                                     |
|----------------------|-----------------------------------------------------------|
| Name                 | `jabs_bounding_boxes_{identity_name}` (one per identity)  |
| Shape stored in file | `(num_frames, 2, 2)`                                      |
| Shape returned by reader | `(num_identities, num_frames, 2, 2)` (all stacked)    |
| Both modes           | Same per-identity shape in combined and per-identity files |

Format: `[[upper_left_x, upper_left_y], [lower_right_x, lower_right_y]]` in pixels.

The reader looks for keys `jabs_bounding_boxes_{name}` for each name in `identity_names`
(from `jabs_metadata`). If all are present, they are stacked in identity order to form
the returned array. If any are missing, `bounding_boxes` is `None`.

---

## Static objects

Static objects are fixed-position spatial landmarks that do not move during a session.
They are read from `static_objects/` in JABS pose HDF5 files (v5+).

Common static objects:

| Object        | Shape                | Description                                       |
|---------------|----------------------|---------------------------------------------------|
| `corners`     | `(4, 2)`             | Four corners of the arena                         |
| `lixit`       | `(1, 2)` or `(3, 2)` | Water spout — single tip, or tip + left + right  |
| `food_hopper` | `(4, 2)`             | Four corners of the food hopper opening           |

### NWB representation

Each static object is a `PoseEstimation` container with a **single timestamp
(`t = 0.0 s`)**, one `PoseEstimationSeries` per keypoint, and a dedicated `Skeleton`
in the `Skeletons` container. Nodes are named `{object_name}_{i}` (zero-indexed).

**PoseEstimationSeries fields:**

| Field                   | Value                                                                             |
|-------------------------|-----------------------------------------------------------------------------------|
| `name`                  | `{object_name}_{i}`                                                               |
| `data`                  | shape `(1, 2)` — the `(x, y)` coordinate                                         |
| `timestamps`            | `[0.0]`                                                                           |
| `unit`                  | `"pixels"`                                                                        |
| `reference_frame`       | `"Top-left corner of video frame, x increases rightward, y increases downward"`  |
| `confidence`            | `[1.0]`                                                                           |
| `confidence_definition` | `"Static landmark; confidence is always 1.0"`                                    |

*JABS pose files carry no confidence values for static objects; `1.0` is a placeholder.
Consumers should ignore the confidence field for static objects.*

**PoseEstimation fields:**

| Field             | Value                                                   |
|-------------------|---------------------------------------------------------|
| `name`            | `{object_name}` (e.g. `"corners"`)                      |
| `description`     | `"Static object: {object_name}"`                        |
| `source_software` | `"JABS"`                                                |
| `skeleton`        | The matching `Skeleton` from the `Skeletons` container  |

### Example — `corners` (4 keypoints)

```
Skeletons/
  corners/
    nodes: ["corners_0", "corners_1", "corners_2", "corners_3"]

processing/behavior/
  corners/                         PoseEstimation
    corners_0/                     PoseEstimationSeries
      data:       [[10.0, 20.0]]   shape (1, 2)
      timestamps: [0.0]
      confidence: [1.0]
    corners_1/
      data:       [[300.0, 20.0]]
      timestamps: [0.0]
      confidence: [1.0]
    corners_2/
      data:       [[10.0, 300.0]]
      timestamps: [0.0]
      confidence: [1.0]
    corners_3/
      data:       [[300.0, 300.0]]
      timestamps: [0.0]
      confidence: [1.0]
```

### Example — `lixit` (3-keypoint variant)

```
Skeletons/
  lixit/
    nodes: ["lixit_0", "lixit_1", "lixit_2"]

processing/behavior/
  lixit/                           PoseEstimation
    lixit_0/                       tip
      data:       [[62.0, 166.0]]
      timestamps: [0.0]
      confidence: [1.0]
    lixit_1/                       left side
      data:       [[65.0, 160.0]]
      timestamps: [0.0]
      confidence: [1.0]
    lixit_2/                       right side
      data:       [[60.0, 172.0]]
      timestamps: [0.0]
      confidence: [1.0]
```

---

## Dynamic objects

Dynamic objects are objects whose position or count may change over time. Unlike animal
pose, predictions are not made every frame — only a sparse subset of frames is sampled.
Dynamic objects are introduced in JABS pose format v7.

The HDF5 pose v7 format stores dynamic objects under `dynamic_objects/[name]/`:

| Dataset          | Shape                                                                                              | Description                                         |
|------------------|----------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| `points`         | `(n_predictions, max_count, 2)` for single-keypoint; `(n_predictions, max_count, n_keypoints, 2)` for multi-keypoint | Keypoint coordinates; axis order set by `axis_order` attribute |
| `counts`         | `(n_predictions,)`                                                                                 | Number of valid object instances at each prediction |
| `sample_indices` | `(n_predictions,)`                                                                                 | Frame indices at which predictions were made        |

The `points` dataset carries an optional HDF5 attribute:

| Attribute    | Values            | Default | Meaning                                                                  |
|--------------|-------------------|---------|--------------------------------------------------------------------------|
| `axis_order` | `"xy"` or `"yx"` | `"yx"`  | Coordinate ordering in the file. JABS always normalizes to `(x, y)` on read. |

> **Note:** The `"yx"` default matches the fecal boli network, which was trained with
> HRNet and stores coordinates in row-major (y, x) order.

JABS normalizes all dynamic object `points` arrays to 4-D
`(n_predictions, max_count, n_keypoints, 2)` internally. Single-keypoint objects stored
as 3-D in HDF5 are expanded to `n_keypoints=1` on read.

### NWB representation

Each dynamic object is a `PoseEstimation` container with `n_predictions` **irregular
timestamps**. One `PoseEstimationSeries` is written per **instance slot × keypoint**
combination, with a dedicated `Skeleton` in the `Skeletons` container.

#### Timestamps

Frame indices are converted to seconds for the NWB time axis:

```
timestamps[p] = sample_indices[p] / fps

# Recover on read:
sample_indices[p] = round(timestamps[p] * fps)
```

#### Instance slot validity via confidence

Not all `max_count` slots are occupied at every prediction. Occupancy is encoded in
`confidence`, using the same field used for keypoint validity in animal pose:

```
confidence[p] = 1.0   if counts[p] > slot_index
              = 0.0   otherwise
```

The same confidence value applies to every keypoint within a slot. `counts` can be
recovered on read by summing slots where `confidence > 0` at each prediction timestamp.
Coordinate values in empty slots are meaningless padding and must not be used by
consumers.

#### Node naming convention

| Condition          | Node name pattern    | Example                              |
|--------------------|----------------------|--------------------------------------|
| `n_keypoints == 1` | `{name}_{slot}`      | `fecal_boli_0`                       |
| `n_keypoints > 1`  | `{name}_{slot}_{kp}` | `door_0_0`, `door_0_1`, `door_1_0`  |

**PoseEstimationSeries fields:**

| Field                   | Value                                                                              |
|-------------------------|------------------------------------------------------------------------------------|
| `name`                  | `{name}_{slot}` or `{name}_{slot}_{kp}`                                            |
| `data`                  | shape `(n_predictions, 2)` — `(x, y)` coordinates                                 |
| `timestamps`            | `sample_indices / fps`                                                             |
| `unit`                  | `"pixels"`                                                                         |
| `reference_frame`       | `"Top-left corner of video frame, x increases rightward, y increases downward"`   |
| `confidence`            | `1.0` if slot occupied, `0.0` otherwise                                            |
| `confidence_definition` | `"1.0=valid object instance in this slot, 0.0=slot unoccupied at this prediction"` |

**PoseEstimation fields:**

| Field             | Value                                                   |
|-------------------|---------------------------------------------------------|
| `name`            | `{object_name}` (e.g. `"fecal_boli"`)                  |
| `description`     | `"Dynamic object: {object_name}"`                       |
| `source_software` | `"JABS"`                                                |
| `skeleton`        | The matching `Skeleton` from the `Skeletons` container  |

### Example — `fecal_boli` (single keypoint per instance, up to 3 instances)

50 predictions were made; up to 3 fecal boli are visible at once.

```
Skeletons/
  fecal_boli/
    nodes: ["fecal_boli_0", "fecal_boli_1", "fecal_boli_2"]

processing/behavior/
  fecal_boli/                      PoseEstimation
    fecal_boli_0/                  PoseEstimationSeries — slot 0
      data:       shape (50, 2)
      timestamps: [t_0, t_1, ..., t_49]    # sample_indices / fps
      confidence: [1.0, 1.0, 0.0, ...]    # 1.0 where counts > 0
    fecal_boli_1/                  slot 1
      data:       shape (50, 2)
      timestamps: [t_0, t_1, ..., t_49]
      confidence: [1.0, 0.0, 0.0, ...]    # 1.0 where counts > 1
    fecal_boli_2/                  slot 2
      data:       shape (50, 2)
      timestamps: [t_0, t_1, ..., t_49]
      confidence: [0.0, 0.0, 0.0, ...]    # 1.0 where counts > 2
```

At prediction `p=0`, `counts[0]=2`: slots 0 and 1 are valid, slot 2 is padding.
At prediction `p=1`, `counts[1]=1`: only slot 0 is valid.

### Example — multi-keypoint dynamic object (2 keypoints per instance, up to 2 instances)

A hypothetical `door` object with 2 keypoints per instance (left edge, right edge),
maximum 2 doors in the arena, 30 predictions.

```
Skeletons/
  door/
    nodes: ["door_0_0", "door_0_1", "door_1_0", "door_1_1"]
    #          slot 0     slot 0     slot 1     slot 1
    #          kp 0       kp 1       kp 0       kp 1

processing/behavior/
  door/                            PoseEstimation
    door_0_0/                      slot 0, keypoint 0 — left edge of door 0
      data:       shape (30, 2)
      timestamps: [t_0, ..., t_29]
      confidence: [1.0, 1.0, ...]  # 1.0 where counts > 0
    door_0_1/                      slot 0, keypoint 1 — right edge of door 0
      data:       shape (30, 2)
      timestamps: [t_0, ..., t_29]
      confidence: [1.0, 1.0, ...]  # same as door_0_0 — slot-level validity
    door_1_0/                      slot 1, keypoint 0 — left edge of door 1
      data:       shape (30, 2)
      timestamps: [t_0, ..., t_29]
      confidence: [0.0, 1.0, ...]  # 1.0 where counts > 1
    door_1_1/                      slot 1, keypoint 1 — right edge of door 1
      data:       shape (30, 2)
      timestamps: [t_0, ..., t_29]
      confidence: [0.0, 1.0, ...]  # same as door_1_0
```

---

## `jabs_metadata` scratch field

Every JABS NWB file contains a `ScratchData` object named `jabs_metadata` in the NWB
`scratch` space. Its `data` field is a JSON string carrying all JABS-specific metadata
needed for a lossless round-trip. Standard NWB fields alone are insufficient because
pynwb returns `PoseEstimationSeries` in alphabetical order from HDF5, which would
otherwise scramble the keypoint ordering.

### Keys

| Key                     | Type                    | Present                      | Description |
|-------------------------|-------------------------|------------------------------|-------------|
| `format_version`        | `int`                   | Always                       | JABS NWB format version. Currently `1`. |
| `identity_names`        | `list[str]`             | Always                       | Ordered list of `PoseEstimation` container names that are animal identities. Defines identity order on read. |
| `num_identities`        | `int`                   | Always                       | Total number of animal identities in the recording session. In per-identity mode this equals `split_subject_count`; the file itself contains only one identity. |
| `body_parts`            | `list[str]`             | Always                       | Ordered list of keypoint names for animal skeletons. Preserves original write order, since HDF5 returns groups alphabetically. |
| `cm_per_pixel`          | `float \| null`         | Always                       | Pixel-to-centimetre scale factor. `null` if not available in the source pose file. |
| `external_ids`          | `list[str] \| null`     | Always                       | Original external identity names from the pose file (e.g. mouse cage IDs). `null` if the pose file had no external IDs. |
| `subjects`              | `dict[str, dict] \| null` | Always                     | Per-identity subject metadata keyed by identity name. `null` if no subject metadata is available. Inner dict may contain `subject_id`, `sex`, `species`, `age` (ISO 8601 duration), `date_of_birth` (ISO 8601 datetime), `genotype`, `strain`, `weight`, and `description`. DANDI requires `species`, `sex`, and either `age` or `date_of_birth`. Values are `null` when not available. |
| `metadata`              | `dict`                  | Always                       | Provenance metadata from the source pose file. Includes `source_file`, `pose_format_version`, and optionally `source_file_hash`. |
| `static_object_names`   | `list[str]`             | When static objects present  | Names of all `PoseEstimation` containers that are static objects. |
| `dynamic_object_names`  | `list[str]`             | When dynamic objects present | Names of all `PoseEstimation` containers that are dynamic objects. |
| `dynamic_object_shapes` | `dict[str, [int, int]]` | When dynamic objects present | Maps each dynamic object name to `[max_count, n_keypoints]`. Required to reconstruct the 4-D `points` array `(n_predictions, max_count, n_keypoints, 2)` from the flat series list on read. |
| `per_identity_files`    | `bool`                  | Per-identity mode only       | `true` if this file is one of a set of per-identity NWB files. |
| `source_identity_index` | `int`                   | Per-identity mode only       | Zero-based index of the identity in this file within the original multi-identity dataset. Used to restore original identity order when merging siblings. |
| `split_subject_count`      | `int`                   | Per-identity mode only       | Total number of subjects in the session across all split files. Used to validate that all sibling files are present before merging. |

### Example — combined file with two identities, static objects, and dynamic objects

```json
{
  "format_version": 1,
  "identity_names": ["subject_1", "subject_2"],
  "num_identities": 2,
  "body_parts": ["nose", "left_ear", "right_ear", "base_neck", "left_front_paw",
                 "right_front_paw", "center_spine", "left_rear_paw", "right_rear_paw",
                 "base_tail", "mid_tail", "tip_tail"],
  "cm_per_pixel": 0.043,
  "external_ids": null,
  "subjects": {
    "subject_1": {
      "subject_id": "M123",
      "sex": "M",
      "species": "Mus musculus",
      "age": "P70D",
      "genotype": "WT",
      "strain": "C57BL/6J",
      "weight": null,
      "description": null
    },
    "subject_2": {
      "subject_id": "M124",
      "sex": "F",
      "species": "Mus musculus",
      "age": "P72D",
      "genotype": "Shank3+/-",
      "strain": "C57BL/6J",
      "weight": null,
      "description": null
    }
  },
  "metadata": {
    "source_file": "/data/session_pose_est_v7.h5",
    "pose_format_version": 7,
    "source_file_hash": "a3f1c8..."
  },
  "static_object_names": ["corners", "lixit"],
  "dynamic_object_names": ["fecal_boli"],
  "dynamic_object_shapes": {
    "fecal_boli": [3, 1]
  }
}
```

### Example — per-identity file (identity 1 of 3)

```json
{
  "format_version": 1,
  "identity_names": ["subject_2"],
  "num_identities": 3,
  "body_parts": ["nose", "left_ear", "..."],
  "cm_per_pixel": 0.043,
  "external_ids": null,
  "subjects": {
    "subject_1": { "subject_id": "M123", "sex": "M", "species": "Mus musculus", "age": "P70D", "genotype": "WT" },
    "subject_2": { "subject_id": "M124", "sex": "F", "species": "Mus musculus", "age": "P72D", "genotype": "Shank3+/-" },
    "subject_3": { "subject_id": "M125", "sex": "M", "species": "Mus musculus", "age": "P68D", "genotype": "WT" }
  },
  "metadata": { "source_file": "...", "pose_format_version": 7 },
  "static_object_names": ["corners", "lixit"],
  "dynamic_object_names": ["fecal_boli"],
  "dynamic_object_shapes": { "fecal_boli": [3, 1] },
  "per_identity_files": true,
  "source_identity_index": 1,
  "split_subject_count": 3
}
```

> **Note:** Per-identity files store the full `subjects` dict for all identities, not
> just the one identity in that file. This makes each file self-contained and means the
> reader can recover complete subject metadata from any sibling.

---

## Read-path disambiguation

All `PoseEstimation` containers in `processing/behavior` are classified using three
explicit lists from `jabs_metadata`. Each container name appears in exactly one list:

```
all PoseEstimation containers in behavior
    │
    ├── name in identity_names        →  animal identity
    ├── name in static_object_names   →  static object
    └── name in dynamic_object_names  →  dynamic object  (use dynamic_object_shapes to reconstruct)
```

Using explicit lists rather than inference rules (e.g. "everything else is a static
object") ensures the classification remains correct if new container types are added in
future format versions.

---

## Coordinate system

All coordinates in JABS NWB files use the following convention:

| Property | Value                                    |
|----------|------------------------------------------|
| Origin   | Top-left corner of the video frame       |
| x axis   | Increases rightward (column direction)   |
| y axis   | Increases downward (row direction)       |
| Units    | Pixels                                   |

This applies to animal keypoints, static object points, and dynamic object points.
Dynamic object coordinates are stored as `(y, x)` in the HDF5 pose file by default (the
`axis_order` attribute controls this); JABS flips them to `(x, y)` on read before
writing to NWB. NWB files always contain `(x, y)` order.

---

## Why ndx-multisubjects is not used

[ndx-multisubjects](https://github.com/nehatk17/ndx-multisubjects) is a NWB extension
that adds multi-subject support to NWB files through three new types:

- **`SubjectsTable`** — a `DynamicTable` with one row per animal, storing standard
  subject fields (`subject_id`, `sex`, `genotype`, `strain`, `age`, `weight`, etc.)
- **`NdxMultiSubjectsNWBFile`** — a subclass of `NWBFile` that embeds the
  `SubjectsTable` in general metadata
- **`SelectSubjectsContainer`** — an `NWBDataInterface` that links data to a subject
  subset via a `DynamicTableRegion`

Standard NWB only supports a single `Subject` on `NWBFile.subject`, so this extension
addresses a real gap for multi-animal recordings. JABS evaluated it and chose not to
adopt it for the following reasons:

**1. `NdxMultiSubjectsNWBFile` is a non-standard `NWBFile` subclass.**
Any tool that opens a JABS NWB file without the extension installed will either fail
or silently lose the subjects table. The core value of NWB is that files are readable
by the broader ecosystem using only pynwb. This extension undermines that guarantee.

**2. `SelectSubjectsContainer` does not compose cleanly with ndx-pose.**
The extension's model for associating data with subjects requires wrapping data
containers inside `SelectSubjectsContainer`. JABS `PoseEstimation` containers live
directly in `processing/behavior` and are already named by identity — adding a wrapper
layer would significantly restructure the layout without a commensurate benefit.

**3. The extension is still Beta.**
ndx-multisubjects is published on PyPI (v0.1.1, November 2025) and sleap-io merged
support for it in December 2025, so the ecosystem is beginning to form. However, at
Beta (0.1.x) the API may still change, and broad adoption across NWB tooling has not
yet occurred. Since JABS NWB support is itself under active development, taking a
dependency on an immature extension adds unnecessary coupling at this stage.

**What JABS does instead.**
Per-animal biological metadata (`subject_id`, `sex`, `species`, `age`, `date_of_birth`,
`genotype`, `strain`, `weight`, `description`) can be stored in the `subjects` key of
`jabs_metadata`. DANDI requires `species`, `sex`, and either `age` (ISO 8601 duration,
e.g. `"P70D"`) or `date_of_birth` (ISO 8601 datetime) on every subject.
This keeps the file readable by any standard NWB tool while preserving the metadata in
a structured, machine-readable form. If ndx-multisubjects stabilises and achieves
broader adoption, migrating to it would be straightforward since all the underlying
data is already present.