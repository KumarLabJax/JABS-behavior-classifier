#!/bin/bash
# Submit a Slurm array job that runs `jabs-cli postprocess` over a directory of
# JABS prediction files.
#
# Snapshots the prediction file list into a manifest, sizes the array from the
# file count, records every setting in a job environment file, and submits
# run_postprocess.sh. Nothing in this directory needs to be edited to change
# the input directory, the pipeline config, the behavior, or the batch size.
set -euo pipefail
export LC_ALL=C

JABS_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JABS_LIB_DIR="${JABS_SCRIPT_DIR}/lib"
# shellcheck source=lib/submit_common.sh
source "${JABS_LIB_DIR}/submit_common.sh"

usage() {
    cat <<'USAGE'
Usage: submit_postprocess.sh --config PATH --input-dir DIR [options]

Runs `jabs-cli postprocess` over every prediction file in a directory as a
Slurm array job, processing --files-per-task files per array task.

Required:
      --config PATH        JSON or YAML postprocessing pipeline config

  An input source is required as well: --input-dir or --file-list, both
  described under Input selection below.

Postprocessing options:
  -b, --behavior NAME      Restrict processing to a single behavior. Required
                           when the config is a bare list of stages.
  -o, --out-dir DIR        Write results here under the same file name. Omit to
                           update each prediction file in place.

Sizing note:
  --files-per-task times the worst case time per file must fit inside --time.
  The defaults (24 files, 4 hours) assume 10 minutes per file.

USAGE
    jabs_common_usage
}

# Job-specific defaults.
JABS_JOB_NAME="jabs-postprocess"
JABS_FILES_PER_TASK=24
JABS_WALLTIME="04:00:00"
JABS_MEM="4G"

CONFIG=""
BEHAVIOR=""

jabs_normalize_args "$@"
set -- ${JABS_ARGV[@]+"${JABS_ARGV[@]}"}

if (( $# == 0 )); then
    usage >&2
    exit 2
fi

while (( $# > 0 )); do
    case $1 in
        --config)       jabs_opt_value "$@"; CONFIG=$JABS_OPT_VALUE; shift 2 ;;
        -b|--behavior)  jabs_opt_value "$@"; BEHAVIOR=$JABS_OPT_VALUE; shift 2 ;;
        -o|--out-dir)   jabs_opt_value "$@"; JABS_OUT_DIR=$JABS_OPT_VALUE; shift 2 ;;
        *)
            if jabs_common_opt "$@"; then
                shift "$JABS_OPT_SHIFT"
            else
                jabs_usage_error "unknown option: $1"
            fi
            ;;
    esac
done

if [[ -z $CONFIG ]]; then
    jabs_usage_error "--config is required"
fi
if [[ ! -r $CONFIG ]]; then
    jabs_fail "cannot read config file: ${CONFIG}"
fi

jabs_validate_common

CONFIG=$(jabs_abspath "$CONFIG")
if [[ -n $JABS_OUT_DIR ]]; then
    mkdir -p -- "$JABS_OUT_DIR" || jabs_fail "cannot create ${JABS_OUT_DIR}"
    JABS_OUT_DIR=$(jabs_abspath "$JABS_OUT_DIR")
fi

jabs_build_manifest postprocess
if [[ -n $JABS_OUT_DIR ]]; then
    jabs_check_basename_collisions
fi
jabs_write_common_env
jabs_env_var JABS_CONFIG "$CONFIG"
jabs_env_var JABS_BEHAVIOR "$BEHAVIOR"

jabs_submit "${JABS_SCRIPT_DIR}/run_postprocess.sh"
