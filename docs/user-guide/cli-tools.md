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
  update-labels         Replace a JABS project's labels with labels imported from another project.
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

If instead you want to import labels from another JABS project while keeping the target's pose untouched, see [`jabs-cli update-labels`](#jabs-cli-update-labels).

## jabs-cli update-labels

The `jabs-cli update-labels` command is the inverse of [`update-pose`](#jabs-cli-update-pose): instead of keeping the target's labels and replacing its pose, it keeps the target's pose and replaces its labels with labels imported from another JABS project. The source project provides both the labels and the pose used for IoU-based identity matching.

**Usage:**

```bash
jabs-cli update-labels <project_dir> <source_project_dir> [--min-iou-thresh <FLOAT>] [--verbose] [--annotate-failures] [--drop-timeline-annotations]
```

- `<project_dir>`: Path to the target JABS project whose labels will be replaced in place. The target's pose is unchanged. If `<project_dir>` is a directory of videos + pose files with no `jabs/` subdirectory, a minimal JABS project is scaffolded automatically — features are not generated, so you may want to run `jabs-init` separately afterwards.
- `<source_project_dir>`: Path to a JABS project providing the replacement labels and the pose used for IoU matching. Must already be a valid JABS project; every source-labeled video must also exist in the target.
- `--min-iou-thresh <FLOAT>`: Minimum acceptable median IoU for a label remap match. Blocks below this threshold are skipped. Default: `0.5`.
- `--verbose`: Print successful label remap assignments in addition to warnings.
- `--annotate-failures`: Add timeline annotations to the target for blocks whose label remap fails. Annotations use the same `behavior-remap-failed` / `not-behavior-remap-failed` tags as `update-pose`; the description text distinguishes the originating operation.
- `--drop-timeline-annotations`: Discard source timeline annotations instead of copying or remapping them.

Before modifying the project, the command validates both inputs, runs the label remap in disposable staging projects, and creates a timestamped backup zip under `<project_dir>/.backup` covering `jabs/project.json`, annotations, and predictions (pose files are not touched). Labels are processed block by block, matched by median bbox IoU between the source pose and the target's existing pose, and written to the staged destination label track. By default, source timeline annotations are also carried forward and remapped the same way as label blocks; use `--drop-timeline-annotations` to discard them.

Existing target labels for videos that the source does not cover are left untouched (per-video replace). Behaviors named in the source's `project.json` but not present in the target are merged into the target's `project.json` so the imported labels are usable in the GUI; behaviors already configured in the target keep their existing settings.

The target's pose is unchanged, so the feature cache stays valid and is **not** regenerated. Predictions are cleared because they are stale relative to the new labels; classifiers, the performance cache, and feature files are all left in place. If you need to retrain after a label import, run training from the GUI or via `jabs-classify`.

If a failure occurs after the live apply begins, the command prints the backup path plus cleanup and manual restore instructions instead of restoring automatically.

**Example:**

```bash
jabs-cli update-labels /path/to/target_project /path/to/source_project --min-iou-thresh 0.5
```

If instead you want to replace the target's pose while keeping its labels, see [`jabs-cli update-pose`](#jabs-cli-update-pose).

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
| `--lixit-parquet PATH` | - | Single-row summary parquet with lixit keypoints. The `keypoints` value contains `(x, y)` pairs; each consecutive group of 3 keypoints (`tip`, `left_side`, `right_side`) defines one lixit. 6 keypoints = 2 lixits. |
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

## jabs-cli cross-validation

The `jabs-cli cross-validation` command runs leave-one-group-out cross-validation for a single behavior in a JABS project, then trains a final model on all labeled data to report feature importance. It prints per-iteration metrics to the console and writes a training report file (the same report produced by the GUI). Use it to estimate how well a classifier generalizes before committing to a trained model.

Features must already be computed for the project (for example via [`jabs-init`](#jabs-init)); if they are missing this command will compute them, which can be slow.

**Usage:**

```bash
jabs-cli cross-validation DIRECTORY --behavior BEHAVIOR \
    [-k SPLITS] \
    [--grouping-strategy {video|individual|filename}] \
    [--grouping-pattern REGEX] \
    [--classifier {catboost|random_forest|xgboost}] \
    [--report-file FILE] \
    [--mlflow [ENV_FILE]] [--mlflow-tag KEY=VALUE] [--mlflow-no-report]
```

- `DIRECTORY`: Path to the JABS project directory.
- `--behavior BEHAVIOR` (required): Behavior to evaluate. Quote it if it contains spaces; must match an existing behavior in the project.
- `-k SPLITS`: Number of cross-validation iterations. `0` (the default) uses the maximum number of splits supported by the data and grouping strategy.
- `--grouping-strategy {video|individual|filename}`: How labeled frames are grouped into cross-validation folds (see [Grouping strategies](#grouping-strategies)). If omitted, the project's saved setting is used.
- `--grouping-pattern REGEX`: Regular expression applied to each video filename to derive a grouping key. Only used with `--grouping-strategy filename`. If omitted, the pattern saved in the project is used.
- `--classifier {catboost|random_forest|xgboost}`: Classifier to evaluate. Defaults to `xgboost`. The available choices depend on which classifier libraries are installed; see [Classifier Types](classifier-types.md).
- `--report-file FILE`: Where to write the training report. The format is chosen by extension: `.md` (Markdown) or `.json` (JSON). If omitted, a timestamped Markdown file is written to the current directory (`<behavior>_<timestamp>_training_report.md`).
- `--mlflow`, `--mlflow-tag`, `--mlflow-no-report`: Optional MLflow logging (see [MLflow logging](#mlflow-logging)).

### Grouping strategies

Cross-validation holds out one *group* of labeled data per iteration and trains on the rest, so groups define what "generalization" means for the score. JABS supports three strategies:

| Strategy | Each group is... | Use when |
|---|---|---|
| `individual` | one (video, identity) pair | you want to measure generalization across individual animals |
| `video` | one whole video (all identities in it) | you want to measure generalization across videos/sessions |
| `filename` | all videos whose filename yields the same key under `--grouping-pattern` | videos from the same cage/cohort/day share a filename component and should not be split across train and test |

For the `filename` strategy, the pattern is applied with `re.search`, so it matches anywhere in the filename. If the pattern has a capturing group, the first captured group is the key; otherwise the whole match is the key. Videos that do not match the pattern are placed in their own single-video group. For example, `--grouping-pattern '^(\w+?)_'` groups `cage12_2024-01-01.mp4` and `cage12_2024-01-02.mp4` together under the key `cage12`.

### Training report

The report (and the console output) include:

- Per-iteration accuracy, precision and recall for both classes, and F1 for the behavior class, plus the held-out test group label for each iteration.
- The top features (by importance) from a final model trained on all labeled data.
- Labeled frame and bout counts, the window size, distance unit, classifier type, and the grouping strategy/pattern used.

**Examples:**

```bash
# Cross-validate "grooming" with default settings (project grouping, all splits, xgboost)
jabs-cli cross-validation /path/to/project --behavior grooming

# 5-fold, grouped by individual animal, with a CatBoost classifier
jabs-cli cross-validation /path/to/project --behavior grooming \
    -k 5 --grouping-strategy individual --classifier catboost

# Group videos by a shared filename prefix and write a JSON report
jabs-cli cross-validation /path/to/project --behavior grooming \
    --grouping-strategy filename --grouping-pattern '^(\w+?)_' \
    --report-file grooming_cv.json
```

### MLflow logging

The cross-validation command can optionally log each run to an [MLflow](https://mlflow.org/) tracking server, recording aggregate metrics, run parameters, descriptive tags, and the training report as an artifact. This is opt-in and off by default.

#### Installing the MLflow extra

MLflow is an optional dependency. Install it with the `mlflow` extra:

```bash
pip install 'jabs-behavior-classifier[mlflow]'
```

If you request MLflow logging without the extra installed, the command prints a warning, ignores the MLflow options, and still runs the cross-validation and writes the report (it exits `0`).

#### Enabling logging

Add the `--mlflow` flag:

```bash
# Use connection settings from the ambient environment
jabs-cli cross-validation /path/to/project --behavior grooming --mlflow

# Use connection settings from a .env file
jabs-cli cross-validation /path/to/project --behavior grooming --mlflow settings.env
```

`--mlflow` optionally takes the path to a `.env` file. With no path, connection settings are read from the current environment.

#### Connection configuration

Connection details — tracking server URI, experiment, authentication, TLS — are **not** passed as command-line options. They come from standard `MLFLOW_*` environment variables, either exported in your shell or written to the `.env` file you pass to `--mlflow`. Only keys beginning with `MLFLOW_` are read from the `.env` file; everything else is ignored.

Common variables:

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | URL (or local path) of the tracking server, e.g. `https://mlflow.example.org` |
| `MLFLOW_EXPERIMENT_NAME` | Name of the experiment the run is logged under |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | HTTP basic-auth credentials |
| `MLFLOW_TRACKING_TOKEN` | Bearer-token auth (alternative to username/password) |

Example `.env` file:

```
MLFLOW_TRACKING_URI=https://mlflow.example.org
MLFLOW_EXPERIMENT_NAME=mouse-grooming
MLFLOW_TRACKING_USERNAME=jabs
MLFLOW_TRACKING_PASSWORD=hunter2
```

#### Selecting the experiment

The run is logged under the experiment named by `MLFLOW_EXPERIMENT_NAME` (or `MLFLOW_EXPERIMENT_ID`). If neither is set, MLflow's built-in **Default** experiment is used. There is no dedicated command-line option for the experiment; set the environment variable in the `.env` file or your shell:

```bash
export MLFLOW_EXPERIMENT_NAME=mouse-grooming
jabs-cli cross-validation /path/to/project --behavior grooming --mlflow
```

#### What gets logged

Each invocation creates one MLflow run named `<behavior>-cv-<timestamp>`.

**Metrics** (aggregated across cross-validation iterations):

- `cv_accuracy_mean`, `cv_accuracy_std`
- `cv_precision_behavior_mean` / `_std`, `cv_recall_behavior_mean` / `_std`, `cv_f1_behavior_mean` / `_std`
- `cv_iterations` — number of folds run
- `frames_behavior`, `frames_not_behavior`, `bouts_behavior`, `bouts_not_behavior` — dataset composition
- `training_time_ms`

**Parameters:** `behavior`, `classifier`, `window_size`, `balance_labels`, `symmetric_behavior`, `distance_unit`, `cv_grouping_strategy`, and `cv_grouping_regex` (only for the `filename` strategy).

**Tags:** auto-derived `behavior`, `classifier`, `cv_grouping_strategy`, and `jabs_git` (the short git SHA of the JABS checkout, when available). Any `--mlflow-tag` entries are merged on top, so a user tag wins over an auto tag with the same key.

**Artifact:** the generated training report file, unless `--mlflow-no-report` is passed.

#### Free-form tags

Add searchable tags to the run with `--mlflow-tag`, which is repeatable:

```bash
jabs-cli cross-validation /path/to/project --behavior grooming --mlflow settings.env \
    --mlflow-tag purpose=baseline --mlflow-tag cohort=2024Q1
```

Each entry is `KEY=VALUE`; only the first `=` splits the entry, so values may contain `=`.

#### Skipping the report artifact

To log metrics and parameters only (no report upload):

```bash
jabs-cli cross-validation /path/to/project --behavior grooming --mlflow --mlflow-no-report
```

#### Exit codes and failure handling

MLflow logging happens **after** the cross-validation results are printed and the report is saved, so a logging failure never costs you the results:

- **Extra not installed:** a warning is printed, the MLflow options are ignored, and the command exits `0`.
- **Logging fails** (for example the tracking server is unreachable or authentication fails): the results and report are preserved, a warning is printed, and the command exits with code **`3`** — distinct from the generic error code `1`, so automation can tell a push failure apart from a cross-validation failure.

#### Full example

```bash
# settings.env contains:
#   MLFLOW_TRACKING_URI=https://mlflow.example.org
#   MLFLOW_EXPERIMENT_NAME=mouse-grooming
jabs-cli cross-validation /path/to/project \
    --behavior grooming \
    --grouping-strategy individual \
    --classifier xgboost \
    --mlflow settings.env \
    --mlflow-tag purpose=baseline
```
