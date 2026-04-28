# Command Line Tools

## jabs-classify

JABS includes a script called `jabs-classify`, which can be used to classify a single video from the command line.

```text
usage: jabs-classify COMMAND COMMAND_ARGS

commands:
 classify   classify a pose file
 train      train a classifier that can be used to classify multiple pose files

See `jabs-classify COMMAND --help` for information on a specific command.
```

### Classify Command

```text
usage: jabs-classify classify [-h] [--random-forest | --xgboost]
                            (--training TRAINING | --classifier CLASSIFIER) --input-pose
                            INPUT_POSE --out-dir OUT_DIR [--fps FPS]
                            [--feature-dir FEATURE_DIR] [--skip-window-cache]
                            [--use-pose-hash]

optional arguments:
  -h, --help            show this help message and exit
  --fps FPS             frames per second, default=30
  --feature-dir FEATURE_DIR
                        Feature cache dir. If present, look here for features before computing.
                        If features need to be computed, they will be saved here.
  --skip-window-cache   Only cache per-frame features when --feature-dir is provided, reducing
                        cache size at the cost of needing to re-calculate window features.
  --use-pose-hash       Include the pose file hash as a subdirectory level in the feature cache
                        path (e.g. <feature-dir>/<video>/<pose-hash>/<identity>). Prevents
                        collisions when multiple pipelines share a feature cache directory and
                        different pose files happen to share the same video name.

required arguments:
  --input-pose INPUT_POSE
                        input HDF5 pose file (v2, v3, or v4).
  --out-dir OUT_DIR     directory to store classification output

optionally override the classifier specified in the training file:
 Ignored if trained classifier passed with --classifier option.
 (the following options are mutually exclusive):
  --random-forest       Use Random Forest
  --xgboost             Use XGBoost

Classifier Input (one of the following is required):
  --training TRAINING   Training data h5 file exported from JABS
  --classifier CLASSIFIER
                        Classifier file produced from the `jabs-classify train` command
```

When `--feature-dir` is provided, `jabs-classify` automatically detects the existing cache format (HDF5 or Parquet) and reads from it. New cache files are always written in Parquet format. If the directory already contains an HDF5 cache, it is read as-is and any new window features are added to the existing HDF5 file — no conversion takes place.

### Train Command

```text
usage: jabs-classify train [-h] [--random-forest | --xgboost]
                         training_file out_file

positional arguments:
  training_file        Training h5 file exported by JABS
  out_file             output filename

optional arguments:
  -h, --help           show this help message and exit

optionally override the classifier specified in the training file:
 (the following options are mutually exclusive):
  --random-forest      Use Random Forest
  --xgboost            Use XGBoost
```

> Note: XGBoost may be unavailable on macOS if `libomp` isn't installed. See `jabs-classify classify --help` output for list of classifiers supported in the current execution environment.

> Note: fps parameter is used to specify the frames per second (used for scaling time unit for speed and velocity features from "per frame" to "per second").

## jabs-features

JABS includes a script called `jabs-features`, which can be used to generate a feature file for a single video from the command line.

```text
usage: jabs-features [-h] --pose-file POSE_FILE --pose-version POSE_VERSION
                            --feature-dir FEATURE_DIR [--use-cm-distances]
                            [--window-size WINDOW_SIZE] [--fps FPS]
                            [--use-pose-hash]

options:
  -h, --help            show this help message and exit
  --pose-file POSE_FILE
                        pose file to compute features for
  --pose-version POSE_VERSION
                        pose version to calculate features
  --feature-dir FEATURE_DIR
                        directory to write output features
  --use-cm-distances    use cm distance units instead of pixel
  --window-size WINDOW_SIZE
                        window size for features (default none)
  --fps FPS             frames per second to use for feature calculation
  --use-pose-hash       Include the pose file hash as a subdirectory level in the feature cache
                        path (e.g. <feature-dir>/<video>/<pose-hash>/<identity>). Prevents
                        collisions when a shared cache directory is used across multiple pipelines.
```

Features are always written in Parquet format. Use `--use-pose-hash` when building a shared feature cache for multiple pipelines where video filenames may collide.

## jabs-cli

`jabs-cli` is a command line interface that provides access to multiple JABS command line tools. To get a listing of current commands, run:

```bash
jabs-cli --help
```

```bash
Usage: jabs-cli [OPTIONS] COMMAND [ARGS]...

  JABS CLI.

Options:
  --verbose  Enable verbose output.
  --help     Show this message and exit.

Commands:
  convert-parquet       Convert a parquet pose file to JABS HDF5 pose format.
  postprocess           Apply a postprocessing pipeline to a JABS prediction HDF5 file.
  convert-to-nwb        Convert a JABS pose HDF5 file to NWB format.
  cross-validation      Run leave-one-group-out cross-validation for a JABS project.
  export-training       Export training data for a specified behavior and JABS project directory.
  prune                 Prune unused videos from a JABS project directory.
  rename-behavior       Rename a behavior in a JABS project.
  sample-frames         Sample PNG frames from a JABS project filtered by a behavior label.
  sample-pose-intervals Sample contiguous intervals from a batch of JABS pose and video files.
  update-pose           Update a JABS project to use updated pose files while remapping labels.
```

For full documentation of the `convert-to-nwb` command, including output modes, subjects
and session metadata JSON formats, and NWB file structure, see [NWB Export](nwb-export.md).

To get help for a specific command, run:

```bash
jabs-cli <command> --help
```

## jabs-init

The `jabs-init` command initializes a JABS project directory and computes features for all videos in parallel. This is much faster than computing features through the GUI during training. It also validates the project directory and can accept a metadata file describing the project and videos.

**Usage:**

```bash
jabs-init <project_dir> [--metadata <metadata.json>] [--force] [--processes <N>]
          [-w WINDOW_SIZE] [--cache-format {hdf5,parquet}] [--skip-feature-generation]
```

- `<project_dir>`: Path to the JABS project directory containing video and pose files.
- `--metadata <metadata.json>`: Optional path to a JSON metadata file describing the project and videos.
- `--force`: Recompute features even if cache files already exist.
- `--processes <N>`: Number of parallel workers to use for feature computation. If omitted, this defaults to the logical CPU count.
- `-w WINDOW_SIZE`: Window size(s) to pre-compute. Can be repeated (e.g. `-w 2 -w 5`). Defaults to 5 if omitted.
- `--cache-format {hdf5,parquet}`: Feature cache storage format. If omitted, the existing project setting is preserved. New projects default to `parquet`; projects created before this option existed default to `hdf5`. Pass `--cache-format` explicitly only when you want to change or set the format — use `--force` alongside it to rewrite existing cache files in the new format.
- `--skip-feature-generation`: Validate and initialize the project without computing features.

**Examples:**

```bash
# Initialize a project with default settings (Parquet cache, window size 5)
jabs-init /path/to/project

# Initialize with metadata, 8 workers, and explicit window sizes
jabs-init /path/to/project --metadata project_metadata.json --processes 8 -w 2 -w 5

# Migrate an existing HDF5 cache to Parquet (recomputes all features)
jabs-init /path/to/project --cache-format parquet --force
```

See the [Project Setup Guide](project-setup.md#initialization--jabs-init) for a brief overview and [Feature Cache Format](project-setup.md#feature-cache-format) for migration guidance.

## jabs-cli update-pose

The `jabs-cli update-pose` command updates an existing JABS project to use updated pose files for the same videos while remapping existing labels onto the updated poses. This is intended for keeping labels when pose files have been regenerated or otherwise updated.

**Usage:**

```bash
jabs-cli update-pose <project_dir> <updated_pose_dir> [--min-iou-thresh <FLOAT>] [--verbose] [--annotate-failures] [--drop-timeline-annotations] [--skip-feature-gen]
```

- `<project_dir>`: Path to the JABS project to update in place.
- `<updated_pose_dir>`: Directory containing updated pose files for the project videos. For each video, the latest pose version in this directory is used. All videos must resolve to the same latest version, and that version must include bounding boxes.
- `--min-iou-thresh <FLOAT>`: Minimum acceptable median IoU for a label remap match. Blocks below this threshold are skipped. Default: `0.5`.
- `--verbose`: Print successful label remap assignments in addition to warnings.
- `--annotate-failures`: Add timeline annotations to the project for blocks whose label remap fails.
- `--drop-timeline-annotations`: Discard existing timeline annotations from the source project instead of copying or remapping them.
- `--skip-feature-gen`: Skip automatic feature regeneration after a successful pose update.

Before modifying the project, the command validates the updated pose files, runs the pose update and label remap in disposable staging projects, and creates a timestamped backup zip under `<project_dir>/.backup`. By default, existing timeline annotations are also carried forward: video-level annotations are copied as-is, and identity-scoped annotations are remapped by the same interval-matching logic used for label blocks. Use `--drop-timeline-annotations` if you want to discard existing timeline annotations instead. Only after the staged pose update succeeds are annotations, project metadata, and pose files copied back into the project.

After a successful live pose update, features are regenerated automatically only when `--skip-feature-gen` is not passed and the existing `jabs/project.json` already contains explicit `window_sizes`. If the project file has no `window_sizes` entry, there is nothing to regenerate and feature generation is skipped. If you want more control over feature regeneration, use `--skip-feature-gen` and run `jabs-init` manually, or generate features from the GUI.

**Example:**

```bash
jabs-cli update-pose /path/to/project /path/to/updated_pose_dir --min-iou-thresh 0.5
```

## jabs-cli sample-frames

The `jabs-cli sample-frames` command extracts individual video frames as PNG images
from a JABS project, filtered to frames annotated with a specific behavior label.
The primary use case is building targeted training datasets for the keypoint/pose
model — particularly for behaviors where pose estimation quality is known to be
degraded.

**Usage:**

```bash
jabs-cli sample-frames --behavior <label> \
    (--num-frames N | --frames-per-bout N) \
    [--out-dir <path>] \
    <project_dir>
```

- `<project_dir>`: Path to the JABS project directory.
- `--behavior`: Behavior label name to sample frames for (required). Must match a behavior defined in the project.
- `--num-frames N`: Sample a total of N frames distributed uniformly at random across all labeled bouts in the project. Mutually exclusive with `--frames-per-bout`.
- `--frames-per-bout N`: Sample up to N frames from each individual labeled bout. If a bout contains fewer than N frames, all frames in that bout are used. Mutually exclusive with `--num-frames`.
- `--out-dir`: Directory to write PNG files. Defaults to the current working directory. Created (including intermediate directories) if it does not exist.

Exactly one of `--num-frames` or `--frames-per-bout` must be provided.

**Output filenames** follow the JABS GUI "Export Frame" convention:

```
{video_stem}_frame{frame_number:06d}.png
```

For example, frame 1234 from `session_2024.mp4` is written as `session_2024_frame001234.png`.

Frames are deduplicated across identities: if multiple identities overlap in the
same bout, each unique video frame is written only once.

**Examples:**

```bash
# Sample 200 frames total, distributed uniformly across all walking bouts
jabs-cli sample-frames --behavior walking --num-frames 200 /path/to/project

# Sample up to 10 frames from each individual walking bout
jabs-cli sample-frames --behavior walking --frames-per-bout 10 /path/to/project

# Write frames to a specific output directory
jabs-cli sample-frames --behavior walking --num-frames 100 \
    --out-dir /data/frames /path/to/project
```

## jabs-cli sample-pose-intervals

The `jabs-cli sample-pose-intervals` command clips a contiguous interval from each
video and pose file in a batch, writing the results to an output directory. It is
useful for creating smaller representative subsets of large datasets for annotation
or testing. The pose file version is inferred automatically from the filename —
the highest version available for each video is used.

**Usage:**

```bash
jabs-cli sample-pose-intervals \
    --batch-file <batch.txt> \
    --root-dir <root_dir> \
    --out-dir <out_dir> \
    --out-frame-count <N> \
    [--start-frame <F>] \
    [--only-pose]
```

- `--batch-file`: Path to a plain-text file listing video filenames to process, one per line (see [Batch file format](#batch-file-format) below).
- `--root-dir`: Root directory. All filenames in the batch file are interpreted as relative to this directory.
- `--out-dir`: Output directory for clipped pose and video files. Created automatically if it does not exist.
- `--out-frame-count`: Number of frames in each clipped output. At 30 fps, 1800 ≈ one minute.
- `--start-frame`: 1-based start frame index. If omitted, a random start frame is chosen independently for each video.
- `--only-pose`: Write only the clipped pose HDF5 file; skip video output.

**Example:**

```bash
jabs-cli sample-pose-intervals \
    --batch-file batch.txt \
    --root-dir /data/videos \
    --out-dir /data/sampled \
    --out-frame-count 9000 \
    --start-frame 54000
```

### Batch file format

The batch file is a plain-text file with one video filename per line, relative to
`--root-dir`. Blank lines are ignored. Subdirectory separators (`/` or `\`) are
replaced with `+` in output filenames to keep the output directory flat.

```
experiment1/mouse_a.avi
experiment1/mouse_b.avi
experiment2/mouse_c.avi
```

For each video, the corresponding pose file is located automatically by searching
beside the video for `<stem>_pose_est_v*.h5` files and selecting the highest
version. If no pose file is found the video is skipped with a warning.

### Output files

For each successfully processed video, two files are written to `--out-dir`
(or one file with `--only-pose`):

- `<flat_video_name>_<start_frame>_pose_est_v*.h5` — clipped pose HDF5 file
- `<flat_video_name>_<start_frame>.avi` — clipped video (MJPEG, 30 fps)

where `<flat_video_name>` is the video filename with path separators replaced
by `+` and the extension removed, and `<start_frame>` is the 1-based index of
the first frame included in the clip.

### Dynamic object handling

Pose v7+ files may contain dynamic objects (e.g. fecal boli) stored as sparse
predictions indexed by `sample_indices` rather than by frame. When clipping:

- The last prediction strictly before the clip start is included and clamped to
  clip-relative frame 0, so the most recent pre-clip state is carried into the clip.
- All predictions whose `sample_indices` fall within `[start, stop)` are included
  and rebased by subtracting `start`.
- Predictions at or after `stop` are excluded.

If no predictions exist within or before the clip window, empty arrays are written
for that dynamic object.

## jabs-cli postprocess

The `jabs-cli postprocess` command applies a postprocessing pipeline to an
existing JABS prediction HDF5 file. It reads the raw predicted classes, runs the
configured pipeline stages, and writes the results back as
`predicted_class_postprocessed` datasets alongside the original predictions. The raw
predictions are never modified.

**Usage:**

```bash
jabs-cli postprocess PREDICTION_FILE \
    --config CONFIG_FILE \
    [--behavior BEHAVIOR] \
    [--output OUTPUT_FILE]
```

- `PREDICTION_FILE`: Path to the JABS prediction HDF5 file to process.
- `--config CONFIG_FILE`: Path to a JSON or YAML pipeline config file (see [Config file format](#config-file-format) below). Required unless `--list-behaviors` is used.
- `--behavior BEHAVIOR`: Restrict processing to a single named behavior. Required when the config file is a list of stages. When the config is a dict, filters to only the specified behavior.
- `--output OUTPUT_FILE`: Path for the output file. If omitted, the input file is updated in place.
- `--list-behaviors`: Print the behavior names present in the prediction file and exit. Useful for finding the correct behavior names to use in a config dict.

To inspect which behaviors are stored in a prediction file before writing a config:

```bash
jabs-cli postprocess /path/to/predictions.h5 --list-behaviors
```

**Examples:**

Apply a single-behavior pipeline (config is a list of stages):

```bash
jabs-cli postprocess /path/to/predictions.h5 \
    --config grooming_pipeline.json \
    --behavior grooming
```

Apply a multi-behavior pipeline and write to a new file:

```bash
jabs-cli postprocess /path/to/predictions.h5 \
    --config all_behaviors_pipeline.yaml \
    --output /path/to/predictions_postprocessed.h5
```

### Config file format

The config file is a JSON or YAML file in one of two formats.

> **Note:** YAML support requires the `yaml` extra:
> `pip install 'jabs-behavior-classifier[yaml]'`

#### List format (single behavior)

A top-level list of stage configurations applied in order to a single behavior.
`--behavior` is required when this format is used.

```json
[
  {
    "stage_name": "GapInterpolationStage",
    "parameters": {"max_interpolation_gap": 5}
  },
  {
    "stage_name": "BoutDurationFilterStage",
    "parameters": {"min_duration": 10},
    "enabled": true
  }
]
```

#### Dict format (multiple behaviors)

A top-level object mapping behavior names to lists of stage configurations.
Behavior names must match the safe names stored in the HDF5 file (special characters
are replaced with underscores). Use `--list-behaviors` to check the names in a file.

```json
{
  "grooming": [
    {"stage_name": "GapInterpolationStage", "parameters": {"max_interpolation_gap": 5}},
    {"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 10}}
  ],
  "rearing": [
    {"stage_name": "BoutStitchingStage", "parameters": {"max_stitch_gap": 3}}
  ]
}
```

#### Stage schema

Each stage entry has the following fields:

| Field | Required | Description |
|---|---|---|
| `stage_name` | Yes | Name of the postprocessing stage class (see [Available stages](#available-stages) below). |
| `parameters` | Yes | Object of stage-specific parameters, or `null` if the stage takes no parameters. |
| `enabled` | No | Boolean; defaults to `true`. Set to `false` to skip the stage without removing it from the config. |

#### Available stages

| Stage name | Parameter | Description |
|---|---|---|
| `GapInterpolationStage` | `max_interpolation_gap` (int > 0) | Fill short gaps of no-prediction frames between behavior bouts. Gaps up to this length (inclusive) are filled in. |
| `BoutStitchingStage` | `max_stitch_gap` (int > 0) | Merge behavior bouts separated by short not-behavior gaps. Gaps up to this length (inclusive) are stitched over. |
| `BoutDurationFilterStage` | `min_duration` (int > 0) | Remove behavior bouts shorter than the specified duration (in frames). |

Stages are applied in the order they appear in the list. A typical pipeline runs
`GapInterpolationStage` or `BoutStitchingStage` before `BoutDurationFilterStage` so
that short gaps are merged prior to duration filtering.

## jabs-cli convert-parquet

The `jabs-cli convert-parquet` command converts a parquet pose file to the JABS HDF5 pose format (v5).
The output file is named after the input with `.parquet` replaced by `_pose_est_v8.h5`.

### Usage

```bash
jabs-cli convert-parquet PARQUET_PATH [OPTIONS]
```

### Options

| Option | Default | Description |
|---|---|---|
| `--lixit-parquet PATH` | - | Summary parquet with lixit keypoints. Each row of 3 keypoints (tip, left_side, right_side) defines one lixit; 6 keypoints = 2 lixits. |
| `--num-frames INT` | 1800 | Total number of frames in the video. |
| `--out-dir PATH` | same as input | Output directory for the converted HDF5 file. Created if it does not exist. |

### Input format

The input parquet file must have the following columns:

- `frame` - frame number
- `eartag_code` - external identity string (used for `external_identity_mapping`); rows with empty, `"00"`, or `"01"` values are treated as no-reads and excluded
- `kpt_1_x` / `kpt_1_y` - nose
- `kpt_2_x` / `kpt_2_y` - left ear
- `kpt_3_x` / `kpt_3_y` - right ear
- `kpt_4_x` / `kpt_4_y` - base of tail
- `kpt_5_x` / `kpt_5_y` - tip of tail

### Examples

```bash
# Basic conversion
jabs-cli convert-parquet session_poses.parquet

# With lixit landmarks, custom frame count, and custom output directory
jabs-cli convert-parquet session_poses.parquet \
    --lixit-parquet lixit_summary.parquet \
    --num-frames 3600 \
    --out-dir /path/to/output
```
