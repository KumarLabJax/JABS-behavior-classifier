# Project Setup

## Project Directory

A JABS project is a directory of video files and their corresponding pose estimation files. The first time a project directory is opened in JABS, it will create a subdirectory called "jabs", which contains various files created by JABS to save project state, including labels and current predictions.

### Example JABS Project Directory listing:

```text
VIDEO_1.avi
VIDEO_1_pose_est_v3.h5
VIDEO_2.avi
VIDEO_2_pose_est_v3.h5
VIDEO_3.avi
VIDEO_3_pose_est_v3.h5
VIDEO_4.avi
VIDEO_4_pose_est_v3.h5
VIDEO_5.avi
VIDEO_5_pose_est_v3.h5
VIDEO_6.avi
VIDEO_6_pose_est_v3.h5
VIDEO_7.avi
VIDEO_7_pose_est_v3.h5
jabs/
```

## Initialization & jabs-init

The first time you open a project directory with JABS it will create the `jabs` subdirectory. Features will be computed the first time the "Train" button is clicked. This can be time-consuming depending on the number and length of videos in the project directory.

The `jabs-init` script can also be used to initialize a project directory before it is opened in the JABS GUI. This script checks to make sure that a pose file exists for each video in the directory, and that the pose file and video have the same number of frames. Then, after these basic checks, the script will compute features for all the videos in the project. Since `jabs-init` can compute features for multiple videos in parallel, it is significantly faster than doing so through the GUI during the training process.

### jabs-init Usage:

```text
usage: jabs-init [-h] [-f] [-p PROCESSES] [-w WINDOW_SIZE] [--force-pixel-distances] [--metadata METADATA]
                 [--skip-feature-generation]
                 project_dir

positional arguments:
  project_dir

options:
  -h, --help            show this help message and exit
  -f, --force           recompute features even if file already exists, replace existing project metadata
  -p PROCESSES, --processes PROCESSES
                        number of multiprocessing workers
  -w WINDOW_SIZE        Specify window sizes to use for computing window features. Argument can be repeated to specify
                        multiple sizes (e.g. -w 2 -w 5). Size is number of frames before and after the current frame to
                        include in the window. For example, '-w 2' results in a window size of 5 (2 frames before, 2
                        frames after, plus the current frame). If no window size is specified, a default of 5 will be
                        used.
  --force-pixel-distances
                        use pixel distances when computing features even if project supports cm
  --metadata METADATA   path to a JSON file containing project metadata to be validated and injected into the project
  --skip-feature-generation
                        Skip feature calculation and only initialize/validate the project
```

### Example jabs-init Command

The following command runs the `jabs-init` script to compute features using window sizes of 2, 5, and 10. The script will use up to 8 processes for computing features (-p8). If no -p argument is passed, `jabs-init` will use up to 4 processes.

`jabs-init -p8 -w2 -w5 -w10 <path/to/project/dir>`

### Project Metadata

The --metadata argument can be used to pass a JSON file containing project metadata. This file has the following schema:

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

### jabs/classifiers

This directory contains trained classifiers. Currently, these are stored in Python Pickle files and should be considered non-portable. While non-portable, these files can be used alongside `jabs-classify classify --classifier` for CLI-based prediction on the same machine as the gui running the training.

### jabs/features

This directory contains the computed features. There is one directory per project video, and within each video directory there will be one feature subdirectory per identity. Feature files are portable between machines, but JABS may need to recompute the features if they were created with a different version of JABS. Feature files contain a version attribute that is incremented when features are added or changed, or the format of the features file is changed.

### jabs/predictions

This directory contains one HDF5 prediction file per video (e.g., `VIDEO_1.h5`). Each file has a `/predictions` group with one subgroup per behavior. Prediction files are automatically opened and displayed by JABS if they exist and are portable. This is the same format produced by `jabs-classify`.

### jabs/training_logs

This directory contains training reports generated each time a classifier is trained. Reports are saved as Markdown files with filenames in the format `<BehaviorName>_<timestamp>_training_report.md`. Each report includes training performance metrics, cross-validation results, and feature importance rankings. These reports provide a permanent record of training sessions for documentation and comparison purposes.

