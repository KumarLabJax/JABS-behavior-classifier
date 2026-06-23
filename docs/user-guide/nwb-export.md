# NWB Export

The `jabs-cli convert-to-nwb` command converts a JABS pose estimation HDF5 file to
[NWB (Neurodata Without Borders)](https://www.nwb.org/) format using the
[ndx-pose](https://github.com/rly/ndx-pose) extension.

!!! note "Optional dependency"
    NWB support is not installed by default. Install the `nwb` extra before use:

```bash
pip install "jabs-behavior-classifier[nwb]"
```

The extra adds `pynwb`, `ndx-pose`, and `ndx-multisubjects` as dependencies.

Two output modes are available. **Choose the mode based on how the files will be used:**

| Mode                            | When to use                                                                       |
|---------------------------------|-----------------------------------------------------------------------------------|
| Per-identity (default)          | DANDI archive upload, tools that expect one subject per file                      |
| Multisubject (`--multisubject`) | A single shareable file holding every subject (via the ndx-multisubjects extension) |

---

## Command usage

```
jabs-cli convert-to-nwb INPUT_PATH OUTPUT [OPTIONS]
```

| Argument / Option            | Description                                                                                                                  |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `INPUT_PATH`                 | JABS pose HDF5 file, any version v2–v8. Format version is inferred automatically from the filename (e.g. `_pose_est_v6.h5`). |
| `OUTPUT`                     | Destination `.nwb` file. By default (per-identity), used as a naming template; the file itself is not created directly. With `--multisubject`, the single combined file is written directly to this path. |
| `--multisubject`             | Write a single multi-subject NWB file (using the ndx-multisubjects extension) instead of the default one file per identity. |
| `--session-description TEXT` | NWB session description string. Defaults to `'JABS PoseEstimation Data'`.                                                    |
| `--subjects PATH`            | Path to a JSON file with per-animal biological metadata.                                                                     |
| `--session-metadata PATH`    | Path to a JSON file with NWB session-level metadata (start time, experimenter, etc.).                                        |

### Examples

```bash
# One NWB file per identity (default; recommended for DANDI upload)
jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb

# A single multi-subject file holding every identity
jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb --multisubject

# Include per-animal metadata
jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb --subjects subjects.json

# Specify session start time and experimenter
jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb --session-metadata session.json
```

---

## Subjects JSON format

Pass a JSON file to `--subjects` to attach per-animal biological metadata to the NWB
output. Keys are identity names: use external IDs from the pose file when present (e.g.
`"mouse_a"`), or `subject_1`, `subject_2`, … when the pose file has no external IDs.

**DANDI requires `species`, `sex`, and either `age` or `date_of_birth` on every
subject.** All other fields are optional.

```json
{
  "subject_1": {
    "subject_id": "M123",
    "sex": "M",
    "species": "Mus musculus",
    "age": "P70D",
    "genotype": "WT",
    "strain": "C57BL/6J"
  },
  "subject_2": {
    "subject_id": "M124",
    "sex": "F",
    "species": "Mus musculus",
    "age": "P72D",
    "genotype": "Shank3+/-",
    "strain": "C57BL/6J"
  }
}
```

| Field            | Type   | Notes                                                                                  |
|------------------|--------|----------------------------------------------------------------------------------------|
| `subject_id`     | string | Lab identifier for the animal                                                          |
| `sex`            | string | **Required by DANDI.** `"M"`, `"F"`, `"U"`, or `"O"`                                   |
| `species`        | string | **Required by DANDI.** Latin binomial, e.g. `"Mus musculus"`                           |
| `age`            | string | **Required by DANDI** (or `date_of_birth`). ISO 8601 duration, e.g. `"P70D"` (70 days) |
| `date_of_birth`  | string | Alternative to `age`. ISO 8601 datetime, e.g. `"2024-01-15T00:00:00+00:00"`            |
| `genotype`       | string | Genetic background, e.g. `"Shank3B+/-"`                                                |
| `strain`         | string | Inbred strain, e.g. `"C57BL/6J"`                                                       |
| `weight`         | string | Body weight, e.g. `"25g"`                                                              |
| `description`    | string | Free-text notes                                                                        |

In per-identity mode (the default), subject metadata is written to both the standard
`NWBFile.subject` field and the `jabs_metadata` scratch field. If no `--subjects` file is
provided, a minimal subject with `subject_id` set to the identity name is written
automatically. In multisubject mode, subject metadata is written to a `SubjectsTable`
(one row per subject) and to `jabs_metadata` (see
[below](#subject-metadata-by-mode)).

---

## Session metadata JSON format

Pass a JSON file to `--session-metadata` to set NWB session-level fields. This is the
primary way to specify `session_start_time`, which is not currently stored in JABS pose
files and otherwise defaults to the time the export was run.

```json
{
  "session_start_time": "2024-03-15T10:30:00-05:00",
  "experimenter": ["Jane Smith", "John Doe"],
  "lab": "Kumar Lab",
  "institution": "The Jackson Laboratory",
  "experiment_description": "Open field test",
  "session_id": "session_001"
}
```

| Key                      | Type                      | Description                                                                                                                                                                               |
|--------------------------|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `session_start_time`     | ISO 8601 string           | Recording start time. Should include a UTC offset (e.g. `-05:00`, `+00:00`, or `Z`). If no offset is provided, the time is assumed UTC and a warning is emitted. Defaults to export time. |
| `experimenter`           | string or list of strings | Name(s) of the experimenter(s).                                                                                                                                                           |
| `lab`                    | string                    | Lab name.                                                                                                                                                                                 |
| `institution`            | string                    | Institution name.                                                                                                                                                                         |
| `experiment_description` | string                    | Free-text description of the experiment.                                                                                                                                                  |
| `session_id`             | string                    | Lab-specific session identifier.                                                                                                                                                          |

All fields are optional. Unknown keys are ignored with a warning.

---

## Output modes

### Per-identity files (default)

One NWB file is written per animal. The `OUTPUT` path is used as a naming template;
files are written as `{output_stem}_{identity_name}.nwb` in the same directory. The
`OUTPUT` path itself is **not** created.

```
session_subject_1.nwb   ← identity 0 + all objects
session_subject_2.nwb   ← identity 1 + all objects
session_subject_3.nwb   ← identity 2 + all objects
```

**This is the most standard output.** Each file contains exactly one animal, so
`NWBFile.subject` is populated with that animal's biological metadata (when provided
via `--subjects`). Any standard NWB tool — including the DANDI archive — can read
the subject field directly without knowing anything about JABS.

Identity names in the filenames come from `external_ids` in the pose file (sanitized
for filesystem compatibility), or fall back to `subject_1`, `subject_2`, … when no
external IDs are present. Static and dynamic objects are written to every per-identity
file identically, since they are session-level data.

#### Reading per-identity files

The JABS reader re-assembles per-identity files transparently. Point it at **any one**
sibling file; it detects the `per_identity_files` flag in `jabs_metadata`, finds all
siblings, and merges them into a single result with all identities in their original
order.

### Multisubject single file (`--multisubject`)

All identities from the recording session are written into a single, self-contained NWB
file at `OUTPUT`, using the
[ndx-multisubjects](https://github.com/nehatk17/ndx-multisubjects) extension.

```
session.nwb
  └── all identities + all objects + a SubjectsTable listing every subject
```

The file is an `NdxMultiSubjectsNWBFile` (a drop-in `NWBFile` subclass). Because standard
NWB's `NWBFile.subject` only holds one subject, multiple subjects are instead described by
a **`SubjectsTable`** (one row per animal) stored in `acquisition`. The pose data itself
is laid out in `processing/behavior` exactly as in a per-identity file, and the full,
lossless JABS round-trip still rides on the `jabs_metadata` scratch field.

This mode is intended for sharing a whole session as one artifact. Reading it back with
the JABS reader returns all identities directly (no sibling files are involved).

### Subject metadata by mode

| Mode         | NWBFile.subject              | SubjectsTable          | jabs_metadata.subjects |
|--------------|------------------------------|------------------------|------------------------|
| Per-identity | Set for this file's identity | —                      | Set (all identities)   |
| Multisubject | Not set                      | One row per subject    | Set (all identities)   |

`jabs_metadata.subjects` always carries the full dict for all identities, in both modes.
This makes each per-identity file self-contained: the JABS reader can recover complete
subject metadata from any sibling without loading the others. In multisubject mode the
`SubjectsTable` is provided for standard NWB / DANDI consumers; JABS itself recovers
subject metadata from `jabs_metadata`.

---

## NWB file structure

For the full format specification — including all field definitions, `jabs_metadata`
keys, and worked examples for static and dynamic objects — see
[File Formats — NWB Pose File](file-formats.md#nwb-pose-file).

The layout below shows a multisubject file with two animal identities, two static objects
(`corners`, `lixit`), and one dynamic object (`fecal_boli`).

```
NdxMultiSubjectsNWBFile
├── acquisition/
│   └── SubjectsTable                      [DynamicTable] multisubject mode only — one row per subject
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

A per-identity file (the default) uses a plain `NWBFile` and the same
`processing/behavior` layout, except:
- The root is a standard `NWBFile`; there is no `SubjectsTable`
- `NWBFile.subject` is populated (when subject metadata is provided)
- Only one animal identity container is present
- `jabs_identity_mask` / `jabs_bounding_boxes_<identity>` cover that identity only

---

### Animal pose

Each animal identity is a `PoseEstimation` container in `processing/behavior`. The
container name is the sanitized external ID from the pose file, or `subject_1`,
`subject_2`, … (1-based) when no external IDs are available.

A single `Skeleton` named `subject` is shared by all animal identities and stored in
the `Skeletons` container.

#### PoseEstimationSeries fields (per keypoint)

| Field                    | Value                                                                           |
|--------------------------|---------------------------------------------------------------------------------|
| `name`                   | Keypoint name (e.g. `"nose"`, `"left_ear"`)                                     |
| `data`                   | shape `(num_frames, 2)` — `(x, y)` coordinates in pixels                        |
| `rate`                   | Frames per second (float)                                                       |
| `unit`                   | `"pixels"`                                                                      |
| `reference_frame`        | `"Top-left corner of video frame, x increases rightward, y increases downward"` |
| `confidence`             | shape `(num_frames,)` — `0.0` = missing keypoint, `> 0.0` = valid               |
| `confidence_definition`  | `"0.0=invalid/missing keypoint, >0.0=valid keypoint"`                           |

---

### Identity mask

`jabs_identity_mask` is a `TimeSeries` that records whether each identity is present in
each frame.

| Mode         | Shape stored in file           | Shape returned by reader       |
|--------------|--------------------------------|--------------------------------|
| Multisubject | `(num_frames, num_identities)` | `(num_identities, num_frames)` |
| Per-identity | `(num_frames,)`                | `(1, num_frames)`              |

---

### Bounding boxes (optional)

When the pose file contains bounding box data, one `TimeSeries` per identity is written
with the name `jabs_bounding_boxes_{identity_name}`.

| Property                   | Value                                              |
|----------------------------|----------------------------------------------------|
| Name                       | `jabs_bounding_boxes_{identity_name}`              |
| Shape stored in file       | `(num_frames, 2, 2)`                               |
| Shape returned by reader   | `(num_identities, num_frames, 2, 2)`               |

Format: `[[upper_left_x, upper_left_y], [lower_right_x, lower_right_y]]` in pixels.

---

### Static objects

Static objects are fixed-position spatial landmarks that do not move during a session.
They are read from `static_objects/` in JABS pose HDF5 files (pose format v5+).

Common static objects:

| Object        | Shape                | Description                                     |
|---------------|----------------------|-------------------------------------------------|
| `corners`     | `(4, 2)`             | Four corners of the arena                       |
| `lixit`       | `(1, 2)` or `(3, 2)` | Water spout — single tip, or tip + left + right |
| `food_hopper` | `(4, 2)`             | Four corners of the food hopper opening         |

Each static object is a `PoseEstimation` container with a **single timestamp
(`t = 0.0 s`)**, one `PoseEstimationSeries` per keypoint, and a dedicated `Skeleton`.
Nodes are named `{object_name}_{i}` (zero-indexed).

Each `PoseEstimationSeries` for a static object has data shape `(1, 2)` — one row for
the single timestamp and two columns for `(x, y)`. Because the time dimension (1) is
shorter than the spatial dimension (2), the DANDI validator will emit a
`NWBI.check_data_orientation` warning for each static keypoint:

```
[NWBI.check_data_orientation] — Data may be in the wrong orientation. Time should be
in the first dimension, and is usually the longest dimension. Here, another dimension
is longer.
```

**These warnings are expected and can be ignored.** The check is a heuristic designed
to catch transposed animal pose arrays; it fires a false positive for static objects,
which legitimately have only one timestamp by definition.

---

### Dynamic objects

Dynamic objects are objects whose position or count may change over time. Unlike animal
pose, predictions are made only for a sparse subset of frames. Dynamic objects are
available from JABS pose format v7+.

Each dynamic object is a `PoseEstimation` container with `n_predictions` irregular
timestamps. One `PoseEstimationSeries` is written per instance slot × keypoint
combination.

Instance slot occupancy is encoded in the `confidence` field:

- `confidence = 1.0` — slot is occupied at this prediction
- `confidence = 0.0` — slot is unoccupied; coordinate values are padding and must be
  ignored

Node naming:

| Condition                       | Node name pattern    | Example                       |
|---------------------------------|----------------------|-------------------------------|
| Single keypoint per instance    | `{name}_{slot}`      | `fecal_boli_0`                |
| Multiple keypoints per instance | `{name}_{slot}_{kp}` | `door_0_0`, `door_0_1`        |

---

### `jabs_metadata` scratch field

Every JABS NWB file contains a `ScratchData` object named `jabs_metadata` in the NWB
`scratch` space. Its `data` field is a JSON string carrying JABS-specific metadata
needed for a lossless round-trip. Standard NWB fields alone are insufficient because
pynwb returns `PoseEstimationSeries` in alphabetical order from HDF5, which would
otherwise scramble the keypoint ordering.

Tools that do not use the JABS reader can parse this JSON directly to recover identity
ordering, subject metadata, and object classification. (Keypoint ordering is not stored
here; the JABS reader restores it from the canonical keypoint index.)

#### Keys

| Key                     | Type                      | Present                      | Description                                                                                                                                                                                                         |
|-------------------------|---------------------------|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `format_version`        | `int`                     | Always                       | JABS NWB format version. Currently `1`.                                                                                                                                                                             |
| `identity_names`        | `list[str]`               | Always                       | Ordered list of animal identity container names. Defines identity order on read.                                                                                                                                    |
| `num_identities`        | `int`                     | Always                       | Total number of animal identities in the recording session.                                                                                                                                                         |
| `cm_per_pixel`          | `float \| null`           | Always                       | Pixel-to-centimetre scale factor. `null` if not available.                                                                                                                                                          |
| `external_ids`          | `list[str] \| null`       | Always                       | Original external identity names from the pose file. `null` if the pose file had no external IDs.                                                                                                                   |
| `subjects`              | `dict[str, dict] \| null` | Always                       | Per-identity subject metadata keyed by identity name, for all identities. `null` if no subject metadata is available. Fields: `subject_id`, `sex`, `species`, `age`, `date_of_birth`, `genotype`, `strain`, `weight`, `description`. DANDI requires `species`, `sex`, and either `age` or `date_of_birth`. |
| `metadata`              | `dict`                    | Always                       | Provenance from the source pose file: `source_file`, `pose_format_version`, and optionally `source_file_hash`.                                                                                                      |
| `static_object_names`   | `list[str]`               | When static objects present  | Names of all static object `PoseEstimation` containers.                                                                                                                                                             |
| `dynamic_object_names`  | `list[str]`               | When dynamic objects present | Names of all dynamic object `PoseEstimation` containers.                                                                                                                                                            |
| `dynamic_object_shapes` | `dict[str, [int, int]]`   | When dynamic objects present | Maps each dynamic object name to `[max_count, n_keypoints]`.                                                                                                                                                        |
| `multisubject`          | `bool`                    | Multisubject mode only       | `true` if this is a single multi-subject file written with the ndx-multisubjects extension.                                                                                                                         |
| `per_identity_files`    | `bool`                    | Per-identity mode only       | `true` if this file is one of a set of per-identity NWB files.                                                                                                                                                      |
| `source_identity_index` | `int`                     | Per-identity mode only       | Zero-based index of the identity in this file.                                                                                                                                                                      |
| `split_subject_count`      | `int`                     | Per-identity mode only       | Total number of subjects in the session across all split files.                                                                                                                                                     |

#### Example — multisubject file

```json
{
  "format_version": 1,
  "multisubject": true,
  "identity_names": ["subject_1", "subject_2"],
  "num_identities": 2,
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

---

## Coordinate system

All coordinates in JABS NWB files use the following convention:

| Property | Value                                  |
|----------|----------------------------------------|
| Origin   | Top-left corner of the video frame     |
| x axis   | Increases rightward (column direction) |
| y axis   | Increases downward (row direction)     |
| Units    | Pixels                                 |

This applies to animal keypoints, static object points, and dynamic object points.
NWB files always store coordinates in `(x, y)` order.

---

## Data not exported

Pose files v6 and later may contain instance segmentation data. This data is **not**
included in the NWB output. See
[File Formats — Data not exported to NWB](file-formats.md#data-not-exported-to-nwb)
for the full list of omitted fields. If you need segmentation data, read it directly
from the source JABS pose HDF5 file.