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
                            [--feature-dir FEATURE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --fps FPS             frames per second, default=30
  --feature-dir FEATURE_DIR
                        Feature cache dir. If present, look here for features before computing.
                        If features need to be computed, they will be saved here.

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
```

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
  convert-to-nwb        Convert a JABS pose estimation file to NWB format.
  cross-validation      Run leave-one-group-out cross-validation for a JABS project.
  export-training       Export training data for a specified behavior and JABS project directory.
  prune                 Prune unused videos from a JABS project directory.
  rename-behavior       Rename a behavior in a JABS project.
  sample-pose-intervals Sample contiguous intervals from a batch of JABS pose and video files.
  update-pose           Update a JABS project to use updated pose files while remapping labels.
```

See [NWB Export](nwb-export.md) for full documentation of the `convert-to-nwb` command
and NWB file structure.

To get help for a specific command, run:

```bash
jabs-cli <command> --help
```

## jabs-init

The `jabs-init` command initializes a JABS project directory and computes features for all videos in parallel. This is much faster than computing features through the GUI during training. It also validates the project directory and can accept a metadata file describing the project and videos.

**Usage:**

```bash
jabs-init <project_dir> [--metadata <metadata.json>] [--force] [--processes <N>]
```

- `<project_dir>`: Path to the JABS project directory containing video and pose files.
- `--metadata <metadata.json>`: Optional path to a JSON metadata file describing the project and videos.
- `--force`: Overwrite existing features and settings if present.
- `--processes <N>`: Number of parallel workers to use for feature computation. If omitted, this defaults to the logical CPU count.

**Example:**

```bash
jabs-init /path/to/project --metadata project_metadata.json --processes 8
```

See the [Project Setup Guide](project-setup.md#initialization--jabs-init) for a brief overview.

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

- `<flat_video_name>_<start_frame>.h5` — clipped pose HDF5 file
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
