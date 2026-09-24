# JABS Slurm tools

This directory contains scripts that might be useful for running JABS on a Slurm cluster.
These scripts have been developed for our specific Slurm environment at JAX and may require
modifications to work in other Slurm environments. Please use these scripts as a starting
point and adapt them to your specific needs.

Each job is submitted with a `submit_*.sh` script that takes all of its settings on the
command line, so a new run never requires editing a file. The submit script:

1. freezes the list of input files into a timestamped manifest under `manifests/`, so
   every array task agrees on the file ordering even if files appear in the input
   directory while the array is running,
2. writes every setting of the run to a matching `.env` file, which doubles as a record
   of what was submitted,
3. sizes the array from the file count and the batch size, checking it against the
   cluster's `MaxArraySize`, and
4. submits the matching `run_*.sh` worker with the Slurm resource options applied.

The `run_*.sh` workers read the manifest and the `.env` file. They contain no
configuration and refuse to run under a bare `sbatch`.

## Script Descriptions

| Script | Purpose |
|---|---|
| `submit_classify.sh` | Submit an array job running `jabs-classify classify` over a directory of pose files. |
| `run_classify.sh` | Array worker for `submit_classify.sh`. Not run directly. |
| `submit_postprocess.sh` | Submit an array job running `jabs-cli postprocess` over a directory of prediction files. |
| `run_postprocess.sh` | Array worker for `submit_postprocess.sh`. Not run directly. |
| `top_prediction_bouts.py` | Scan a directory of prediction files and write the longest positive bouts to a CSV. Useful for picking clips to review after a classification run. |
| `lib/submit_common.sh` | Option parsing, manifest creation, array sizing, and `sbatch` invocation shared by the submit scripts. |
| `lib/worker_common.sh` | Manifest slicing, virtualenv activation, and the per-file loop shared by the workers. |

Run either submit script with `--help` for the full option list.

## Classification

```bash
./submit_classify.sh \
    --classifier /projects/kumar-lab/USERS/me/StraubTail_20260806.pickle \
    --input-dir  /projects/kumar-lab/USERS/me/study_514/pose \
    --out-dir    /projects/kumar-lab/USERS/me/study_514/predictions
```

Each array task processes `--files-per-task` pose files sequentially (16 by default).
Common adjustments:

```bash
# Smaller batches on a busy partition, with a shared feature cache
./submit_classify.sh -c model.pickle -i pose/ -o predictions/ \
    --files-per-task 8 --throttle 20 --time 02:00:00 \
    --feature-dir /fastscratch/me/features --use-pose-hash

# Train from an exported training file instead of using a saved classifier
./submit_classify.sh --training StraubTail_training.h5 -i pose/ -o predictions/

# Pass an option this wrapper does not expose
./submit_classify.sh -c model.pickle -i pose/ -o predictions/ \
    --extra-arg --skip-window-cache
```

## Postprocessing

```bash
./submit_postprocess.sh \
    --config   /projects/kumar-lab/USERS/me/postprocess.json \
    --behavior StraubTail \
    --input-dir /projects/kumar-lab/USERS/me/study_514/predictions
```

Prediction files are updated in place unless `--out-dir` is given, in which case each
result is written there under the same file name. `--behavior` is required when the
config file is a bare list of stages; it is optional (and acts as a filter) when the
config maps behavior names to stages.

## Output name collisions

Output directories are flat, and each output file is named after its input, so the submit
scripts refuse to start a run in which two inputs would produce the same output file.
For classification that comparison is made on the pose stem, because `jabs-classify`
strips the `_pose_est_vN` suffix: `video_pose_est_v4.h5` and `video_pose_est_v6.h5` both
write `video_behavior.h5`, so keep one pose version per video in an input directory.

## Sizing a run

`--files-per-task` times the worst case time for one file must fit inside `--time`. The
defaults assume 15 min/file for classification (16 files, 4 hours) and 10 min/file for
postprocessing (24 files, 4 hours). If files take longer than expected, tasks are killed
at the walltime and the remainder of their batch is left unprocessed; lower
`--files-per-task` or raise `--time`.

`--throttle` caps how many tasks run at once. Lower it when sharing a partition.

Use `--dry-run` to see the manifest, the resolved settings, and the exact `sbatch`
command without submitting anything.

## Re-running failures

A task exits non-zero if any file in its batch failed, and lists the failures in its
`.err` log. Collect them into a file list and resubmit just those:

```bash
grep -h '^FAILED: ' logs/jabs-classify_12345_*.err | sed 's/^FAILED: //' > retry.txt
./submit_classify.sh -c model.pickle --file-list retry.txt -o predictions/ \
    --files-per-task 1 --job-name jabs-classify-retry
```

`--file-list` accepts one input path per line and replaces `--input-dir`.

## Environment

The workers activate `~/jabs.venv` on the compute node. Point them somewhere else with
`--venv /path/to/venv`, or use `--no-venv` when JABS is already on `PATH` (for example
when it comes from a module load in `--sbatch-arg`).
