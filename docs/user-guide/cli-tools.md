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
  cross-validation  Run leave-one-group-out cross-validation for a JABS project.
  export-training   Export training data for a specified behavior and JABS project directory.
  prune             Prune unused videos from a JABS project directory.
  rename-behavior   Rename a behavior in a JABS project.
```
To get help for a specific command, run:

```bash
jabs-cli <command> --help
```

## jabs-init

The `jabs-init` command initializes a JABS project directory and computes features for all videos in parallel. This is much faster than computing features through the GUI during training. It also validates the project directory and can accept a metadata file describing the project and videos.

**Usage:**

```bash
jabs-init <project_dir> [--metadata <metadata.json>] [--force] [--parallel <N>]
```

- `<project_dir>`: Path to the JABS project directory containing video and pose files.
- `--metadata <metadata.json>`: Optional path to a JSON metadata file describing the project and videos.
- `--force`: Overwrite existing features and settings if present.
- `--parallel <N>`: Number of parallel workers to use for feature computation (default: number of CPUs).

**Example:**

```bash
jabs-init /path/to/project --metadata project_metadata.json --parallel 8
```

See the [Project Setup Guide](project-setup.md#initialization--jabs-init) for a brief overview.
