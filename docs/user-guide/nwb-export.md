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
| `--subjects PATH`            | **Required.** Path to a JSON file with per-animal biological metadata. The conversion fails without the fields DANDI requires - see [Subjects JSON format](#subjects-json-format). |
| `--session-metadata PATH`    | Path to a JSON file with NWB session-level metadata (start time, experimenter, etc.).                                        |

### Examples

```bash
# One NWB file per identity (default; recommended for DANDI upload)
jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb --subjects subjects.json

# A single multi-subject file holding every identity
jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb --subjects subjects.json --multisubject

# Also set session start time and experimenter
jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb \
    --subjects subjects.json --session-metadata session.json
```

`--subjects` appears in every example because the conversion fails without it; see
[Subjects JSON format](#subjects-json-format).

---

## Subjects JSON format

Pass a JSON file to `--subjects` to attach per-animal biological metadata to the NWB
output. Keys are identity names: use external IDs from the pose file when present (e.g.
`"mouse_a"`), or `subject_1`, `subject_2`, … (1-based) when the pose file has no external
IDs. `subject_1` is identity index 0.

An optional `name` field renames the identity — see
[Naming identities](#naming-identities).

!!! warning "`--subjects` is required"
    `species`, `sex`, and either `age` or `date_of_birth` are mandatory on every
    identity. The converter validates them before writing anything and **fails** if any
    are missing or malformed, because the DANDI archive rejects files without them.

    A key no identity reads is ignored with a warning - either because it matches
    nothing, or because a higher-precedence key for the same identity shadows it.
    That usually leaves an identity without metadata and fails the check, so if you
    see the warning, compare your keys against the identity names it lists.

    A blank value (`""`) counts as not supplied, exactly as the writer treats it.

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

| Field           | Type   | Notes                                                                                                                                                          |
|-----------------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`          | string | Renames the identity itself: the NWB container, the per-identity output filename, and the bounding box series. Defaults to the key. See [Naming identities](#naming-identities). |
| `subject_id`    | string | Lab identifier for the animal. Defaults to the identity name when omitted or blank. Must not contain `/`, which breaks DANDI paths.                           |
| `sex`           | string | **Required.** Exactly `"M"`, `"F"`, `"O"`, or `"U"` - case-sensitive, so `"male"` and `"m"` are both rejected. For *C. elegans*, `"XO"` or `"XX"` instead.      |
| `species`       | string | **Required.** Latin binomial, e.g. `"Mus musculus"` - capitalized genus, lowercase epithet, so `"Mus Musculus"` and `"mouse"` are rejected. An NCBI taxonomy IRI such as `"http://purl.obolibrary.org/obo/NCBITaxon_10090"` is also accepted. |
| `age`           | string | **Required** (or `date_of_birth`). ISO 8601 duration, e.g. `"P70D"` (70 days) or `"P2Y"`. A range is allowed: `"P1D/P3D"`, and either end may be left open (`"P90Y/"`, `"/P3D"`). A range must be strictly increasing. |
| `date_of_birth` | string | Alternative to `age`. ISO 8601 datetime, e.g. `"2024-01-15T00:00:00+00:00"`.                                                                                   |
| `genotype`      | string | Genetic background, e.g. `"Shank3B+/-"`                                                                                                                        |
| `strain`        | string | Inbred strain, e.g. `"C57BL/6J"`                                                                                                                               |
| `weight`        | string | Body weight as `[numeric] [unit]` **with a space**, e.g. `"25 g"` or `"0.025 kg"`. `"25g"` is rejected. Units: kg, g, mg, ug, μg, ng, pg. A bare number is accepted only as a JSON float, interpreted as kilograms. |
| `description`   | string | Free-text notes                                                                                                                                                |

In per-identity mode (the default), subject metadata is written to both the standard
`NWBFile.subject` field and the `jabs_metadata` scratch field. In multisubject mode it is
written to a `SubjectsTable` (one row per subject) and to `jabs_metadata` (see
[below](#subject-metadata-by-mode)).

The validation above is a pre-flight check that mirrors `nwbinspector`'s subject checks,
which DANDI treats as blocking. It is not a substitute for the real thing - run
`nwbinspector` against the converted files before uploading:

```bash
nwbinspector session_subject_1.nwb --config dandi
```

### Naming identities

A pose file with no external identities leaves its animals called `subject_1`,
`subject_2`, … . That name is not cosmetic — it names the `PoseEstimation` container, the
per-identity output file (`{stem}_{identity_name}.nwb`) and the bounding box series.
`subject_id` does **not** change any of them; it only labels the `Subject`.

The `name` field supplies the external identity such a pose file is missing. A pose file
that already carries `external_identities` normally keeps them, and needs none of this.

To name an identity, give its entry a `name`:

```json
{
  "subject_1": {
    "name": "NV1-B2A",
    "subject_id": "M123",
    "species": "Mus musculus",
    "sex": "M",
    "age": "P70D"
  }
}
```

That writes `session_NV1-B2A.nwb` holding an `NV1-B2A` container, whose `Subject` is
`M123`. Use the same `name` and `subject_id` if you want them to agree.

- The key still identifies *which* identity you mean, so it stays `subject_1` (or the
  pose file's external ID) even though the entry renames it.
- Renaming some identities and not others is fine; the rest keep the names they had.
- Names must be unique, and are sanitized for use as container names — any character
  that is not alphanumeric, `_` or `-` becomes `_`, so `NV1/B2A` is stored as `NV1_B2A`
  and a warning says so.
- Whatever it resolves to is recorded as that identity's `external_ids` entry in
  `jabs_metadata`, which is the field the reader restores identity names from. Identities
  you leave unnamed keep their `subject_N` placeholder, and that placeholder is what
  lands in `external_ids` for them.
- A pose file that already has external identities *can* be renamed the same way, keyed
  by the existing external ID, but normally you would leave those names alone.


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
  "session_id": "session_001",
  "keywords": ["open field", "mouse", "behavior"]
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
| `keywords`               | list of strings           | Keywords/tags describing the session.                                                                                                                                                     |

All fields are optional. Unknown keys are ignored with a warning.

---

## Publishing to the DANDI archive

JABS NWB output is intended for the
[EMBER archive](https://emberarchive.org/) — the data archive for the NIH BRAIN
Initiative's Brain Behavior Quantification and Synchronization (BBQS) program. EMBER
is operationally distinct from [DANDI](https://dandiarchive.org/) but runs on DANDI
infrastructure at [dandi.emberarchive.org](https://dandi.emberarchive.org/), so
DANDI's validation rules apply.

Uploads are validated with [`nwbinspector`](https://nwbinspector.readthedocs.io/),
and **any `CRITICAL` finding blocks the upload.** The guidance below is about
clearing that gate.

### Requirements at a glance

| Requirement | Why |
|---|---|
| Use per-identity mode (the default) | One `Subject` per file is what DANDI's subject checks expect. `--multisubject` is **not archive-eligible** — see [Limitations](#limitations-for-archive-submission). |
| Pass `--subjects` with complete metadata | `species`, `sex`, and `age` or `date_of_birth` are mandatory. Without them every file fails three `CRITICAL` checks. |
| Pass `--session-metadata` | Not mandatory, but it clears the remaining best-practice warnings and records provenance the archive displays. |

### 1. Prepare the metadata files

Write a `subjects.json` covering **every** identity in the pose file (see
[Subjects JSON format](#subjects-json-format)), and a `session.json` with at least
`session_start_time` (see
[Session metadata JSON format](#session-metadata-json-format)). Pose files do not
record a session start time, so without it the export falls back to the time the
conversion ran, which is not the recording time.

### 2. Convert

```bash
jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb \
    --subjects subjects.json \
    --session-metadata session.json
```

The converter validates subject metadata **before writing anything**. If a required
field is missing or malformed it reports every problem across every identity at once
and writes no files:

```
Error: Subject metadata required by the DANDI archive is missing or malformed:
  subject_1: species is missing; sex is missing; age or date_of_birth is missing
  subject_2: sex 'male' must be one of 'M', 'F', 'O', 'U'
```

A `--subjects` key that matches no identity is reported as a warning naming the valid
identity names, which is the usual explanation for an identity that looks like it was
given metadata but reports it as missing.

### 3. Validate locally before uploading

JABS's pre-flight check mirrors `nwbinspector`'s subject rules, but it is not a
substitute for running the real validator — it checks subject metadata, not the rest
of the file. Run `nwbinspector` on each output before you upload:

```bash
pip install nwbinspector
nwbinspector session_subject_1.nwb --config dandi
```

A clean result reports no `CRITICAL` findings. See
[Expected warnings](#expected-warnings) for the ones that are safe to ignore.

### 4. Upload

Follow the [DANDI upload documentation](https://docs.dandiarchive.org/user-guide-sharing/uploading-data/),
pointing the DANDI CLI at the EMBER instance rather than the main archive. Upload all
per-identity files from a session together — the JABS reader needs the full set of
siblings to reassemble the session (see [Reading per-identity files](#reading-per-identity-files)).

### What JABS checks, and what it does not

| Checked before writing | Left to `nwbinspector` / the archive |
|---|---|
| `species` present and in Latin binomial or NCBI IRI form | Everything outside `Subject` metadata |
| `sex` present and a valid code for the species | File structure, timestamps, data orientation |
| `age` or `date_of_birth` present, and well-formed | Dataset-level best practices |
| `age` ranges strictly increasing | |
| `subject_id` free of `/` | |
| `weight` in `[numeric] [unit]` form | |

Passing the JABS check is necessary but **not sufficient** — always run
`nwbinspector` before uploading.

### Expected warnings

These appear in a normal, acceptable export and do not block upload:

- **Missing session-level metadata**, when `--session-metadata` is omitted:
  `check_experimenter_exists`, `check_institution`, `check_keywords`,
  `check_experiment_description`, and a missing subject `description`. All are
  best-practice suggestions rather than blocking findings.

### Limitations for archive submission

- **`--multisubject` output cannot be validated.** The file depends on the
  `ndx-multisubjects` extension, and `nwbinspector` fails to read it, so the archive
  cannot validate it. It also does not populate `NWBFile.subject`, which DANDI's
  required subject checks read. Use the default per-identity mode for anything
  destined for the archive.
- **Segmentation data is not exported.** See [Data not exported](#data-not-exported).

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
`NWBFile.subject` is populated with that animal's biological metadata from
`--subjects`. Any standard NWB tool — including the DANDI archive — can read
the subject field directly without knowing anything about JABS.

Identity names in the filenames come from `external_ids` in the pose file (sanitized
for filesystem compatibility), or from the `name` field in `--subjects` when the pose
file has none — see [Naming identities](#naming-identities) — falling back to
`subject_1`, `subject_2`, … when neither supplies one. Static and dynamic objects are written to every per-identity
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

Each static object is a `PoseEstimation` container with one
`PoseEstimationSeries` per keypoint and a dedicated `Skeleton`. Nodes are named
`{object_name}_{i}` (zero-indexed).

The constant `(x, y)` value is written at **two timestamps spanning the session** —
the first and last frame — giving each series data shape `(2, 2)`. A single-timestamp
series would have shape `(1, 2)`, whose non-time axis is longer than its time axis,
which `nwbinspector`'s `check_data_orientation` flags regardless of the data being
genuinely static. Repeating the value at both ends leaves it unchanged while keeping
the export clean.

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
| `external_ids`          | `list[str] \| null`       | Always                       | External identity names: from the pose file when it has them, otherwise the ones supplied with the `name` field in `--subjects`. `null` if neither. Identities left unnamed take their `subject_N` placeholder. |
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