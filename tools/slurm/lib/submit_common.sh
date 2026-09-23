# Shared helpers for the JABS Slurm submit scripts.
#
# Sourced by submit_classify.sh and submit_postprocess.sh. Provides the option
# parsing for settings every array job shares, manifest creation, array
# sizing, the job environment file, and the sbatch invocation itself.
#
# Everything a worker needs is written to a job environment file rather than
# passed through `sbatch --export`, so values are free to contain commas and
# the exact settings of a submission stay on disk next to its manifest.
#
# shellcheck shell=bash

# ------------------------------------------------------------------
# Defaults shared by every job. Submit scripts may override any of
# these before parsing their command line.
# ------------------------------------------------------------------
JABS_INPUT_DIR=""
JABS_FILE_LIST=""
JABS_GLOB="*.h5"
JABS_RECURSIVE=""
JABS_FILES_PER_TASK=16
JABS_THROTTLE=50
JABS_WALLTIME="04:00:00"
JABS_MEM="4G"
JABS_CPUS=1
JABS_PARTITION=""
JABS_QOS=""
JABS_ACCOUNT=""
JABS_JOB_NAME=""
JABS_LOG_DIR="logs"
JABS_MANIFEST_DIR="manifests"
JABS_VENV="${HOME}/jabs.venv"
JABS_OUT_DIR=""
JABS_DRY_RUN=""
JABS_EXTRA_ARGS=()
JABS_SBATCH_ARGS=()

JABS_OPT_VALUE=""
JABS_OPT_SHIFT=0
JABS_MANIFEST=""
JABS_JOB_ENV=""

jabs_fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

jabs_usage_error() {
    printf 'ERROR: %s\n\n' "$*" >&2
    usage >&2
    exit 2
}

# Absolute path for a file or directory that may or may not exist yet.
jabs_abspath() {
    local path=$1
    if [[ -d $path ]]; then
        (cd -- "$path" && pwd)
    else
        local dir base
        dir=$(dirname -- "$path")
        base=$(basename -- "$path")
        [[ -d $dir ]] || jabs_fail "no such directory: ${dir}"
        printf '%s/%s\n' "$(cd -- "$dir" && pwd)" "$base"
    fi
}

# Rewrite --option=value into two words so the parse loops only handle one
# form. Result lands in JABS_ARGV.
jabs_normalize_args() {
    JABS_ARGV=()
    local arg
    for arg in "$@"; do
        if [[ $arg == --*=* ]]; then
            JABS_ARGV+=("${arg%%=*}" "${arg#*=}")
        else
            JABS_ARGV+=("$arg")
        fi
    done
}

# Pull the value of an option out of the remaining argument list.
# Call as `jabs_opt_value "$@"` from inside a case branch; the value lands in
# JABS_OPT_VALUE.
jabs_opt_value() {
    (( $# >= 2 )) || jabs_usage_error "option $1 requires a value"
    [[ -n $2 ]] || jabs_usage_error "option $1 requires a non-empty value"
    JABS_OPT_VALUE=$2
}

# Handle one option shared by all submit scripts. Returns non-zero if the
# option is not one of ours, so the caller can report it or handle it itself.
# On success JABS_OPT_SHIFT holds the number of words consumed.
jabs_common_opt() {
    JABS_OPT_SHIFT=2
    case $1 in
        -i|--input-dir)     jabs_opt_value "$@"; JABS_INPUT_DIR=$JABS_OPT_VALUE ;;
        -l|--file-list)     jabs_opt_value "$@"; JABS_FILE_LIST=$JABS_OPT_VALUE ;;
        -g|--glob)          jabs_opt_value "$@"; JABS_GLOB=$JABS_OPT_VALUE ;;
        -n|--files-per-task) jabs_opt_value "$@"; JABS_FILES_PER_TASK=$JABS_OPT_VALUE ;;
        -t|--throttle)      jabs_opt_value "$@"; JABS_THROTTLE=$JABS_OPT_VALUE ;;
        --time)             jabs_opt_value "$@"; JABS_WALLTIME=$JABS_OPT_VALUE ;;
        --mem)              jabs_opt_value "$@"; JABS_MEM=$JABS_OPT_VALUE ;;
        --cpus)             jabs_opt_value "$@"; JABS_CPUS=$JABS_OPT_VALUE ;;
        --partition)        jabs_opt_value "$@"; JABS_PARTITION=$JABS_OPT_VALUE ;;
        --qos)              jabs_opt_value "$@"; JABS_QOS=$JABS_OPT_VALUE ;;
        --account)          jabs_opt_value "$@"; JABS_ACCOUNT=$JABS_OPT_VALUE ;;
        --job-name)         jabs_opt_value "$@"; JABS_JOB_NAME=$JABS_OPT_VALUE ;;
        --log-dir)          jabs_opt_value "$@"; JABS_LOG_DIR=$JABS_OPT_VALUE ;;
        --manifest-dir)     jabs_opt_value "$@"; JABS_MANIFEST_DIR=$JABS_OPT_VALUE ;;
        --venv)             jabs_opt_value "$@"; JABS_VENV=$JABS_OPT_VALUE ;;
        --extra-arg)        jabs_opt_value "$@"; JABS_EXTRA_ARGS+=("$JABS_OPT_VALUE") ;;
        --sbatch-arg)       jabs_opt_value "$@"; JABS_SBATCH_ARGS+=("$JABS_OPT_VALUE") ;;
        -r|--recursive)     JABS_RECURSIVE=1;  JABS_OPT_SHIFT=1 ;;
        --no-venv)          JABS_VENV="";      JABS_OPT_SHIFT=1 ;;
        --dry-run)          JABS_DRY_RUN=1;    JABS_OPT_SHIFT=1 ;;
        -h|--help)          usage; exit 0 ;;
        *) return 1 ;;
    esac
    return 0
}

# Validate the settings every submit script shares.
jabs_validate_common() {
    if [[ -n $JABS_INPUT_DIR && -n $JABS_FILE_LIST ]]; then
        jabs_usage_error "--input-dir and --file-list are mutually exclusive"
    fi
    if [[ -z $JABS_INPUT_DIR && -z $JABS_FILE_LIST ]]; then
        jabs_usage_error "one of --input-dir or --file-list is required"
    fi
    if [[ -n $JABS_INPUT_DIR && ! -d $JABS_INPUT_DIR ]]; then
        jabs_fail "input directory does not exist: ${JABS_INPUT_DIR}"
    fi
    if [[ -n $JABS_FILE_LIST && ! -r $JABS_FILE_LIST ]]; then
        jabs_fail "cannot read file list: ${JABS_FILE_LIST}"
    fi
    [[ $JABS_FILES_PER_TASK =~ ^[0-9]+$ ]] && (( JABS_FILES_PER_TASK > 0 )) \
        || jabs_usage_error "--files-per-task must be a positive integer"
    [[ $JABS_THROTTLE =~ ^[0-9]+$ ]] && (( JABS_THROTTLE > 0 )) \
        || jabs_usage_error "--throttle must be a positive integer"
    [[ $JABS_CPUS =~ ^[0-9]+$ ]] && (( JABS_CPUS > 0 )) \
        || jabs_usage_error "--cpus must be a positive integer"
    if [[ -n $JABS_VENV && ! -r "${JABS_VENV}/bin/activate" ]]; then
        jabs_fail "no virtualenv activate script at ${JABS_VENV}/bin/activate (use --venv or --no-venv)"
    fi
}

# Freeze the list of input files into a manifest. Every array task reads this
# snapshot, so the slices agree even if files appear in the input directory
# while the array is running.
#
# $1: short label used in the manifest and job env file names.
jabs_build_manifest() {
    local label=$1

    mkdir -p -- "$JABS_MANIFEST_DIR" || jabs_fail "cannot create ${JABS_MANIFEST_DIR}"
    local manifest_dir stamp
    manifest_dir=$(jabs_abspath "$JABS_MANIFEST_DIR")
    stamp=$(date +%Y%m%d_%H%M%S)
    JABS_MANIFEST="${manifest_dir}/${label}_${stamp}.txt"
    JABS_JOB_ENV="${manifest_dir}/${label}_${stamp}.env"

    if [[ -n $JABS_FILE_LIST ]]; then
        # Copy rather than reference: the snapshot must not change under a
        # running array, and blank lines would shift every later index.
        grep -v '^[[:space:]]*$' -- "$JABS_FILE_LIST" > "$JABS_MANIFEST" || true
    else
        JABS_INPUT_DIR=$(jabs_abspath "$JABS_INPUT_DIR")
        local find_args=("$JABS_INPUT_DIR")
        if [[ -z $JABS_RECURSIVE ]]; then find_args+=(-maxdepth 1); fi
        find_args+=(-type f -name "$JABS_GLOB")
        find "${find_args[@]}" | sort > "$JABS_MANIFEST"
    fi

    JABS_NUM_FILES=$(wc -l < "$JABS_MANIFEST" | tr -d '[:space:]')
    if (( JABS_NUM_FILES == 0 )); then
        rm -f -- "$JABS_MANIFEST"
        if [[ -n $JABS_FILE_LIST ]]; then
            jabs_fail "file list is empty: ${JABS_FILE_LIST}"
        fi
        jabs_fail "no files matching '${JABS_GLOB}' found in ${JABS_INPUT_DIR}"
    fi
}

# Refuse to run when two input files share a base name. Output files are named
# after their input, so a collision would have tasks overwriting each other in
# the output directory. Only reachable with --recursive or a --file-list that
# spans directories.
jabs_check_basename_collisions() {
    local dupes
    dupes=$(sed -e 's#.*/##' -- "$JABS_MANIFEST" | sort | uniq -d)
    if [[ -n $dupes ]]; then
        printf 'ERROR: input files share base names, so their output would collide in %s:\n' \
            "$JABS_OUT_DIR" >&2
        printf '%s\n' "$dupes" | sed -e 's#^#  #' >&2
        printf 'Submit each source directory separately, or drop --recursive.\n' >&2
        exit 1
    fi
}

# Append a scalar to the job environment file, quoted so it can be sourced back.
jabs_env_var() {
    printf '%s=%q\n' "$1" "$2" >> "$JABS_JOB_ENV"
}

# Append an array to the job environment file.
jabs_env_array() {
    local name=$1
    shift
    printf '%s=(' "$name" >> "$JABS_JOB_ENV"
    if (( $# > 0 )); then printf '%q ' "$@" >> "$JABS_JOB_ENV"; fi
    printf ')\n' >> "$JABS_JOB_ENV"
}

# Write the job environment entries every worker reads. Submit scripts add
# their own entries with jabs_env_var after calling this.
jabs_write_common_env() {
    : > "$JABS_JOB_ENV"
    {
        printf '# JABS Slurm job settings written by %s on %s\n' \
            "$(basename -- "$0")" "$(date)"
        printf '# Sourced by the array worker. Edit and resubmit to re-run with changes.\n'
    } >> "$JABS_JOB_ENV"
    jabs_env_var JABS_SLURM_LIB "$JABS_LIB_DIR"
    jabs_env_var JABS_JOB_LABEL "$JABS_JOB_NAME"
    jabs_env_var JABS_MANIFEST "$JABS_MANIFEST"
    jabs_env_var JABS_FILES_PER_TASK "$JABS_FILES_PER_TASK"
    jabs_env_var JABS_VENV "$JABS_VENV"
    jabs_env_var JABS_OUT_DIR "$JABS_OUT_DIR"
    jabs_env_array JABS_EXTRA_ARGS ${JABS_EXTRA_ARGS[@]+"${JABS_EXTRA_ARGS[@]}"}
}

# Size the array against the cluster's MaxArraySize and submit the runner.
#
# $1: path to the run_*.sh worker script.
jabs_submit() {
    local runner=$1
    [[ -r $runner ]] || jabs_fail "cannot read runner script: ${runner}"

    local num_tasks last
    num_tasks=$(( (JABS_NUM_FILES + JABS_FILES_PER_TASK - 1) / JABS_FILES_PER_TASK ))
    last=$(( num_tasks - 1 ))

    local max_array=""
    if command -v scontrol > /dev/null 2>&1; then
        max_array=$(scontrol show config 2>/dev/null | awk '/^MaxArraySize/ {print $3}')
    fi
    if [[ $max_array =~ ^[0-9]+$ ]] && (( last >= max_array )); then
        printf 'ERROR: need array index %d but MaxArraySize is %d.\n' "$last" "$max_array" >&2
        printf 'Raise --files-per-task to at least %d.\n' \
            "$(( (JABS_NUM_FILES + max_array - 2) / (max_array - 1) ))" >&2
        exit 1
    fi

    mkdir -p -- "$JABS_LOG_DIR" || jabs_fail "cannot create ${JABS_LOG_DIR}"
    local log_dir
    log_dir=$(jabs_abspath "$JABS_LOG_DIR")

    local sbatch_args=(
        --job-name "$JABS_JOB_NAME"
        --output "${log_dir}/${JABS_JOB_NAME}_%A_%a.out"
        --error "${log_dir}/${JABS_JOB_NAME}_%A_%a.err"
        --time "$JABS_WALLTIME"
        --cpus-per-task "$JABS_CPUS"
        --mem "$JABS_MEM"
        --array "0-${last}%${JABS_THROTTLE}"
        --export "ALL,JABS_JOB_ENV=${JABS_JOB_ENV}"
    )
    if [[ -n $JABS_PARTITION ]]; then sbatch_args+=(--partition "$JABS_PARTITION"); fi
    if [[ -n $JABS_QOS ]]; then sbatch_args+=(--qos "$JABS_QOS"); fi
    if [[ -n $JABS_ACCOUNT ]]; then sbatch_args+=(--account "$JABS_ACCOUNT"); fi
    sbatch_args+=(${JABS_SBATCH_ARGS[@]+"${JABS_SBATCH_ARGS[@]}"})

    # `sbatch --export` splits on commas, so a comma in the job env path would
    # silently truncate it into a bogus variable.
    if [[ $JABS_JOB_ENV == *,* ]]; then
        jabs_fail "--manifest-dir must not contain a comma: ${JABS_JOB_ENV}"
    fi

    printf 'Manifest:  %s\n' "$JABS_MANIFEST"
    printf 'Job env:   %s\n' "$JABS_JOB_ENV"
    printf 'Logs:      %s/%s_%%A_%%a.{out,err}\n' "$log_dir" "$JABS_JOB_NAME"
    printf '%d file(s), %d per task -> array 0-%d%%%d\n' \
        "$JABS_NUM_FILES" "$JABS_FILES_PER_TASK" "$last" "$JABS_THROTTLE"

    if [[ -n $JABS_DRY_RUN ]]; then
        printf '\n--- job env ---\n'
        cat -- "$JABS_JOB_ENV"
        printf '\n--- would run ---\n'
        printf '%q ' sbatch "${sbatch_args[@]}" "$runner"
        printf '\n'
        return 0
    fi

    sbatch "${sbatch_args[@]}" "$runner"
}

# Usage text for the options jabs_common_opt handles. Submit scripts print
# this after their own job-specific options.
jabs_common_usage() {
    cat <<'USAGE'
Input selection:
  -i, --input-dir DIR      Directory to scan for input files
  -l, --file-list PATH     File containing one input path per line (instead of
                           --input-dir; useful for re-running failures)
  -g, --glob PATTERN       Glob used to select files in --input-dir (default: *.h5)
  -r, --recursive          Recurse into subdirectories of --input-dir

Batching:
  -n, --files-per-task N   Files each array task processes sequentially
  -t, --throttle N         Max concurrently running array tasks (default: 50)

Slurm resources (override the #SBATCH defaults in the runner):
      --time HH:MM:SS      Walltime per array task
      --mem SIZE           Memory per array task
      --cpus N             CPUs per array task (default: 1)
      --partition NAME     Partition to submit to
      --qos NAME           QOS to submit under
      --account NAME       Account to charge
      --job-name NAME      Slurm job name, also used in log file names
      --sbatch-arg ARG     Extra raw argument for sbatch (repeatable)

Environment and output:
      --venv PATH          Virtualenv to activate on the compute node
                           (default: ~/jabs.venv)
      --no-venv            Do not activate a virtualenv
      --log-dir DIR        Directory for Slurm logs (default: ./logs)
      --manifest-dir DIR   Directory for the manifest and job env file
                           (default: ./manifests)
      --extra-arg ARG      Extra argument passed through to the JABS command
                           (repeatable)
      --dry-run            Print what would be submitted and exit
  -h, --help               Show this help and exit
USAGE
}
