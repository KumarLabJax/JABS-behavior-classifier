#!/bin/bash
# Submit a Slurm array job that runs `jabs-classify classify` over a directory
# of pose files.
#
# Snapshots the pose file list into a manifest, sizes the array from the file
# count, records every setting in a job environment file, and submits
# run_classify.sh. Nothing in this directory needs to be edited to change
# where the input comes from, which classifier is used, or how the work is
# divided across tasks.
set -euo pipefail
export LC_ALL=C

JABS_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JABS_LIB_DIR="${JABS_SCRIPT_DIR}/lib"
# shellcheck source=lib/submit_common.sh
source "${JABS_LIB_DIR}/submit_common.sh"

usage() {
    cat <<'USAGE'
Usage: submit_classify.sh --classifier PATH --input-dir DIR --out-dir DIR [options]

Runs `jabs-classify classify` over every pose file in a directory as a Slurm
array job, processing --files-per-task files per array task.

Classifier (exactly one is required):
  -c, --classifier PATH    Trained classifier pickle (from `jabs-classify train`
                           or saved by the JABS GUI)
      --training PATH      Training HDF5 exported from JABS; each task trains
                           from it before classifying

Required:
  -o, --out-dir DIR        Directory to write prediction files to

  An input source is required as well: --input-dir or --file-list, both
  described under Input selection below.

Classification options:
      --feature-dir DIR    Feature cache directory
      --use-pose-hash      Include the pose hash in the feature cache path
      --skip-window-cache  Cache only per-frame features
      --fps N              Frames per second of the source video

Sizing note:
  --files-per-task times the worst case time per file must fit inside --time.
  The defaults (16 files, 4 hours) assume 15 minutes per file.

USAGE
    jabs_common_usage
}

# Job-specific defaults.
JABS_JOB_NAME="jabs-classify"
JABS_FILES_PER_TASK=16
JABS_WALLTIME="04:00:00"
JABS_MEM="4G"

CLASSIFIER=""
TRAINING=""
FEATURE_DIR=""
FPS=""
USE_POSE_HASH=""
SKIP_WINDOW_CACHE=""

jabs_normalize_args "$@"
set -- ${JABS_ARGV[@]+"${JABS_ARGV[@]}"}

if (( $# == 0 )); then
    usage >&2
    exit 2
fi

while (( $# > 0 )); do
    case $1 in
        -c|--classifier)     jabs_opt_value "$@"; CLASSIFIER=$JABS_OPT_VALUE; shift 2 ;;
        --training)          jabs_opt_value "$@"; TRAINING=$JABS_OPT_VALUE; shift 2 ;;
        -o|--out-dir)        jabs_opt_value "$@"; JABS_OUT_DIR=$JABS_OPT_VALUE; shift 2 ;;
        --feature-dir)       jabs_opt_value "$@"; FEATURE_DIR=$JABS_OPT_VALUE; shift 2 ;;
        --fps)               jabs_opt_value "$@"; FPS=$JABS_OPT_VALUE; shift 2 ;;
        --use-pose-hash)     USE_POSE_HASH=1; shift ;;
        --skip-window-cache) SKIP_WINDOW_CACHE=1; shift ;;
        *)
            if jabs_common_opt "$@"; then
                shift "$JABS_OPT_SHIFT"
            else
                jabs_usage_error "unknown option: $1"
            fi
            ;;
    esac
done

if [[ -n $CLASSIFIER && -n $TRAINING ]]; then
    jabs_usage_error "--classifier and --training are mutually exclusive"
fi
if [[ -z $CLASSIFIER && -z $TRAINING ]]; then
    jabs_usage_error "one of --classifier or --training is required"
fi
if [[ -n $CLASSIFIER && ! -r $CLASSIFIER ]]; then
    jabs_fail "cannot read classifier: ${CLASSIFIER}"
fi
if [[ -n $TRAINING && ! -r $TRAINING ]]; then
    jabs_fail "cannot read training file: ${TRAINING}"
fi
if [[ -z $JABS_OUT_DIR ]]; then
    jabs_usage_error "--out-dir is required"
fi
if [[ -n $FPS && ! $FPS =~ ^[0-9]+$ ]]; then
    jabs_usage_error "--fps must be a positive integer"
fi

jabs_validate_common

mkdir -p -- "$JABS_OUT_DIR" || jabs_fail "cannot create ${JABS_OUT_DIR}"
JABS_OUT_DIR=$(jabs_abspath "$JABS_OUT_DIR")
if [[ -n $CLASSIFIER ]]; then CLASSIFIER=$(jabs_abspath "$CLASSIFIER"); fi
if [[ -n $TRAINING ]]; then TRAINING=$(jabs_abspath "$TRAINING"); fi
if [[ -n $FEATURE_DIR ]]; then
    mkdir -p -- "$FEATURE_DIR" || jabs_fail "cannot create ${FEATURE_DIR}"
    FEATURE_DIR=$(jabs_abspath "$FEATURE_DIR")
fi

jabs_build_manifest classify
jabs_check_basename_collisions
jabs_write_common_env
jabs_env_var JABS_CLASSIFIER "$CLASSIFIER"
jabs_env_var JABS_TRAINING "$TRAINING"
jabs_env_var JABS_FEATURE_DIR "$FEATURE_DIR"
jabs_env_var JABS_FPS "$FPS"
jabs_env_var JABS_USE_POSE_HASH "$USE_POSE_HASH"
jabs_env_var JABS_SKIP_WINDOW_CACHE "$SKIP_WINDOW_CACHE"

jabs_submit "${JABS_SCRIPT_DIR}/run_classify.sh"
