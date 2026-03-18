# JABS Singularity/Apptainer Containers

This directory contains container definition files and SLURM batch scripts for running
JABS on Linux compute clusters. The definitions are compatible with both
[Singularity](https://docs.sylabs.io/guides/latest/user-guide/) and
[Apptainer](https://apptainer.org/docs/user/latest/) (the Linux Foundation fork of
Singularity). The two tools are largely command-compatible; substitute `apptainer`
for `singularity` in any command below if your cluster uses Apptainer.

## Container Definitions

| File                           | Use Case                                      |
|--------------------------------|-----------------------------------------------|
| [`headless.def`](headless.def) | Command-line use on compute clusters (no GUI) |
| [`gui.def`](gui.def)           | Interactive GUI in a portable environment     |

### Building a Container

```bash
# Headless (batch classification)
singularity build --fakeroot behavior-classifier.sif headless.def  # or: apptainer build ...

# GUI
singularity build --fakeroot JABS-GUI.sif gui.def  # or: apptainer build ...
```

> **Note:** `gui.def` installs additional system packages for Qt6/OpenGL support
> (`qt6-base-dev`, `libglu1-mesa-dev`, `libxcb*`, etc.).

## SLURM Batch Scripts

These scripts are written for JAX's Sumner cluster but can be adapted for other
SLURM environments. Both scripts are self-submitting: run them from the command line
and they will re-submit themselves as SLURM array jobs.

### `behavior-classify-batch.sh`

Classifies a batch of pose files using a trained JABS classifier.

```bash
behavior-classify-batch.sh CLASSIFIER.h5 BATCH_FILE.txt
```

- `CLASSIFIER.h5` — exported JABS classifier file
- `BATCH_FILE.txt` — newline-separated list of video or pose file paths

Each line in the batch file can be a video path (`.avi`/`.mp4`) or a pose file
(`.h5`). When a video path is provided, the script searches for a matching
`*_pose_est_v#.h5` file automatically.

**Expected resource usage per job:**

| Resource | Typical    | Maximum |
|----------|------------|---------|
| Time     | 22–28 min  | 35 min  |
| RAM      | 1.3–2.2 GB | 3 GB    |

### `generate-features.sh`

Generates features for a batch of pose files without running classification.

```bash
generate-features.sh BATCH_FILE.txt FEATURE_FOLDER
```

- `BATCH_FILE.txt` — newline-separated list of video or pose file paths
- `FEATURE_FOLDER` — directory where extracted features will be written

## Container Image Path

Both scripts default to looking for the container image at:

```
/projects/kumar-lab/JABS/JABS-Classify-current.sif
```

Update the `CLASSIFICATION_IMG` variable at the top of each script if your image
is in a different location.