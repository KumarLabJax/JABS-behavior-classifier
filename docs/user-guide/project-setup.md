# Project Setup

## Project Directory

A JABS project is a directory of video files and their corresponding pose estimation files. The first time a project directory is opened in JABS, it will create a subdirectory called "jabs", which contains various files created by JABS to save project state, including labels and current predictions.

### Example JABS Project Directory listing:

```text
VIDEO_1.avi
VIDEO_1_pose.h5
VIDEO_2.avi
VIDEO_2_pose.h5
VIDEO_3.avi
VIDEO_3_pose_est_v6.h5
VIDEO_4.avi
VIDEO_4_pose_est_v6.h5
jabs/
```

### Pose Files

Each video needs a pose file beside it, named after the video in one of two ways:

| Name | Format |
|---|---|
| `<video name>_pose.h5` | The unified JABS pose file. |
| `<video name>_pose_est_v<N>.h5`, N = 2 to 8 | A legacy pose file, one format per version. |

A project can mix the two. When a video has both, the unified file is used; among legacy
files, the highest version is used. Files whose names start with `._` (macOS metadata
files) are ignored.

JABS reads legacy files by converting them to the unified format. Nothing beside the
video is written or renamed: the converted copy is stored in the project's `jabs/cache`
directory, made once and reused until the legacy file (or the converter, after a JABS
upgrade) changes. Opening a large legacy project for the first time can take a while as
its files are converted, and the converted copies take extra disk space (typically two to
four times the legacy file). Deleting `jabs/cache` is safe; files are converted again
when needed.

#### Unusable videos

If a pose file cannot be read or converted, for example because it is damaged or because
the converter refuses it (two animals given the same identity in one frame, or one track
shared by two animals), the project still opens. That video is marked unusable, with the
reason, and is left out of training and classification. Fix or replace the pose file and
reopen the project to use the video again. Items the older readers ignored, such as an
unrecognized dataset or a malformed dynamic object, do not make a video unusable: they are
left out of the converted copy with a warning, unless the `JABS_STRICT_POSE_CONVERSION`
environment variable is set.

#### What a project can use

Which features a project offers depends on what every pose file in it provides:

- **Social features** need more than one identity in a pose file. A project offers them
  only when every usable video has more than one identity.
- **Segmentation features** need segmentation data in every pose file.
- **Landmark features** need the landmark (for example arena corners, food hopper or
  lixit) in every pose file.

## Initialization & jabs-init

The first time you open a project directory with JABS it will create the `jabs` subdirectory. Features will be computed the first time the "Train" button is clicked. This can be time-consuming depending on the number and length of videos in the project directory.

The `jabs-init` script can also be used to initialize a project directory before it is opened in the JABS GUI. This command validates the project directory and can compute features for all videos in parallel, which is significantly faster than doing so through the GUI during training.

For full usage details and options, see the [JABS CLI Tools Guide](cli-tools.md#jabs-init).

### Project Metadata

The --metadata argument can be used to pass a JSON file containing project metadata to `jabs-init`. Metadata can be applied to the JABS project or individual videos. Metadata with the field name "nwb" will be validated against a schema. Additional arbitrary metadata fields are allowed to allow for more flexibility than nwb format. This file has the following schema:

```json
{
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:jax.org:schemas:jabs:project-metadata:v1",
    "title": "JABS Project Metadata Schema",
    "type": "object",
    "properties": {
        "project": {"$ref": "#/$defs/project"},
        "videos": {
            "type": "object",
            "additionalProperties": {"$ref": "#/$defs/video"},
            "propertyNames": {"pattern": "^[^/]{1,251}\\.(avi|mp4)$"}
        }
    },
    "required": [],
    "additionalProperties": false,
    "$defs": {
        "primitive": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]},
        "project": {
            "type": "object",
            "properties": {"nwb": {"$ref": "#/$defs/nwb"}},
            "additionalProperties": {"$ref": "#/$defs/primitive"}
        },
        "video": {
            "type": "object",
            "properties": {"nwb": {"$ref": "#/$defs/nwb"}},
            "required": [],
            "additionalProperties": {"$ref": "#/$defs/primitive"}
        },
        "nwb": {
            "type": "object",
            "properties": {
                "nwb_version": {"type": "string"},
                "session_description": {"type": "string", "minLength": 1},
                "identifier": {"type": "string", "minLength": 1},
                "session_start_time": {"type": "string", "format": "date-time"},
                "file_create_date": {
                    "type": "array",
                    "items": {"type": "string", "format": "date-time"},
                    "minItems": 1
                },
                "general": {"$ref": "#/$defs/general"},
                "analysis": {
                    "description": "Lab-specific and custom analysis results. Free-form by design.",
                    "type": "object",
                    "additionalProperties": true
                }
            },
            "required": [],
            "additionalProperties": true
        },
        "general": {
            "type": "object",
            "properties": {
                "institution": {"type": "string"},
                "lab": {"type": "string"},
                "experimenter": {"type": "array", "items": {"type": "string"}},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "experiment_description": {"type": "string"},
                "data_collection": {"type": "string"},
                "notes": {"type": "string"},
                "pharmacology": {"type": "string"},
                "protocol": {"type": "string"},
                "slices": {"type": "string"},
                "related_publications": {"type": "array", "items": {"type": "string"}},
                "session_id": {"type": "string"},
                "subject": {"$ref": "#/$defs/Subject"},
                "devices": {"type": "array", "items": {"$ref": "#/$defs/Device"}}
            },
            "additionalProperties": false
        },
        "Subject": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "subject_id": {"type": "string"},
                "description": {"type": "string"},
                "species": {"type": "string"},
                "strain": {"type": "string"},
                "genotype": {"type": "string"},
                "sex": {"type": "string"},
                "age": {"type": "string"},
                "age_reference": {"type": "string", "enum": ["birth", "gestational"]},
                "date_of_birth": {"type": "string", "format": "date-time"},
                "weight": {"type": "string"}
            }
        },
        "Device": {
            "type": "object",
            "additionalProperties": true,
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "manufacturer": {"type": "string"},
                "model_number": {"type": "string"},
                "model_name": {"type": "string"},
                "serial_number": {"type": "string"}
            },
            "required": ["name"]
        }
    }
}
```

## JABS Directory Structure

JABS creates a subdirectory called "jabs" inside the project directory. This directory contains app-specific data such as project settings, generated features, user labels, cache files, and the latest predictions.

### jabs/project.json

The `project.json` file contains project settings and metadata (behaviors, feature toggles, etc.).

### jabs/annotations

This directory stores the user's labels, stored in one JSON file per labeled video.

### jabs/archive

This directory contains archived labels. These are compressed files (gzip) containing labels for behaviors that the user has removed from the project. JABS only archives labels. Trained classifiers and predictions are deleted if a user removes a behavior from a project.

### jabs/cache

Files cached by JABS to speed up performance. Some of these files may not be portable, so this directory should be deleted if a JABS project is copied to a different platform.

The `jabs/cache/converted` subdirectory holds the unified copies of the project's legacy pose files (see [Pose Files](#pose-files)). They are named `<video name>.<legacy file hash>-<converter version>.h5`; a copy made by an older converter is replaced the next time the video is used.

### jabs/classifiers

This directory contains trained classifiers. Currently, these are stored in Python Pickle files and should be considered non-portable. While non-portable, these files can be used alongside `jabs-classify classify --classifier` for CLI-based prediction on the same machine as the gui running the training.

Each classifier records the feature version it was trained with. When an upgrade of JABS changes the features (the feature version increases), classifiers trained with the earlier version cannot be loaded: JABS tells you to retrain them, and `jabs-classify` refuses them with the same message. Training data exported with an earlier feature version must likewise be exported again from the project before it can be used to train. Your labels are not affected, so retraining needs no relabeling.

### jabs/features

This directory contains the computed features. There is one directory per project video, and within each video directory there will be one feature subdirectory per identity:

```
jabs/features/
└── video_name/
    ├── 0/          ← identity 0
    ├── 1/          ← identity 1
    └── ...
```

Feature files are portable between machines, but JABS may need to recompute them if the version of JABS that computed them is different from the one reading them. Each cache records a feature version number that is incremented whenever features are added, changed, or the cache format is updated.

JABS supports two feature cache formats: **HDF5** (the original format) and **Parquet** (the current default). See [Feature Cache Format](#feature-cache-format) below for details and migration guidance.

### jabs/predictions

This directory contains one HDF5 prediction file per video (e.g., `VIDEO_1.h5`). Each file has a `/predictions` group with one subgroup per behavior. Prediction files are automatically opened and displayed by JABS if they exist and are portable. This is the same format produced by `jabs-classify`.

### jabs/training_logs

This directory contains training reports generated each time a classifier is trained. Reports are saved as Markdown files with filenames in the format `<BehaviorName>_<timestamp>_training_report.md`. Each report includes training performance metrics, cross-validation results, and feature importance rankings. These reports provide a permanent record of training sessions for documentation and comparison purposes.

## Feature Cache Format

JABS supports two storage formats for the computed feature cache:

| Format      | Description |
|-------------|-------------|
| **Parquet** | The current default. A columnar format that offers faster reads and smaller file sizes. Each identity directory contains `metadata.json`, `per_frame.parquet`, and one `window_<N>.parquet` file per cached window size. |
| **HDF5**    | The original format. All features for an identity are stored in a single `features.h5` file. Supported by all versions of JABS. |

New projects created with `jabs-init` or the GUI default to Parquet. JABS automatically detects the format of existing cache files on read, so HDF5 and Parquet projects both open correctly without any manual configuration.

### Changing the cache format for a project (GUI)

The cache format for a project is stored in `jabs/project.json` and can be changed from the GUI at any time via **Edit > Project Settings > Feature Cache Format**.

Changing the setting alone does not convert existing cache files — JABS reads whatever format is already on disk. To force the cache to be regenerated in the new format:

1. Open the project in JABS.
2. Go to **Edit > Project Settings > Feature Cache Format** and select the desired format.
3. Click **OK** to save.
4. Use **File > Clear Feature Cache…** to delete all existing cache files.
5. Trigger a training or classification run. JABS will recompute and write the cache in the new format on the next cache miss.

### Migrating an existing project to Parquet (CLI)

Use `jabs-init` with `--cache-format parquet` and `--force` to recompute all features and write them in Parquet format:

```bash
jabs-init /path/to/project --cache-format parquet --force
```

`--force` is required to overwrite existing HDF5 cache files. Without it, `jabs-init` skips identities that already have a valid cache. The `--cache-format parquet` flag is also written to `project.json` so the GUI and subsequent CLI runs use Parquet going forward.

If the project has multiple window sizes, pass them explicitly so they are all pre-computed:

```bash
jabs-init /path/to/project --cache-format parquet --force -w 2 -w 5
```

To migrate back to HDF5, use `--cache-format hdf5 --force` in the same way.
