# Shared helpers for the JABS Slurm array workers.
#
# Sourced by run_classify.sh and run_postprocess.sh after they have sourced
# the job environment file written by their submit script. All settings come
# from that file; the runners contain no configuration of their own.
#
# shellcheck shell=bash

# Stable, locale-independent ordering and comparisons.
export LC_ALL=C

jabs_worker_fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

# Resolve the slice of the manifest this array task owns.
# Sets JABS_FILES, JABS_NUM_FILES, JABS_START and JABS_END. Exits 0 when the
# slice starts past the end of the manifest, which happens whenever the array
# was sized larger than the file list.
jabs_worker_slice() {
    [[ -n ${SLURM_ARRAY_TASK_ID:-} ]] \
        || jabs_worker_fail "SLURM_ARRAY_TASK_ID is not set. Submit this with the submit script."
    [[ -n ${JABS_MANIFEST:-} ]] || jabs_worker_fail "JABS_MANIFEST is not set in ${JABS_JOB_ENV}"
    [[ -r $JABS_MANIFEST ]] || jabs_worker_fail "cannot read manifest: ${JABS_MANIFEST}"

    # Read with a loop rather than mapfile: this has to work on the bash 3.2
    # that ships on macOS as well as on the cluster's bash 4+.
    JABS_FILES=()
    local line
    while IFS= read -r line || [[ -n $line ]]; do
        [[ -n $line ]] || continue
        JABS_FILES+=("$line")
    done < "$JABS_MANIFEST"
    JABS_NUM_FILES=${#JABS_FILES[@]}
    (( JABS_NUM_FILES > 0 )) || jabs_worker_fail "manifest is empty: ${JABS_MANIFEST}"

    JABS_START=$(( SLURM_ARRAY_TASK_ID * JABS_FILES_PER_TASK ))
    JABS_END=$(( JABS_START + JABS_FILES_PER_TASK ))
    if (( JABS_END > JABS_NUM_FILES )); then
        JABS_END=$JABS_NUM_FILES
    fi

    if (( JABS_START >= JABS_NUM_FILES )); then
        printf 'Task %s: start index %d is past the end of the manifest (%d files). Nothing to do.\n' \
            "$SLURM_ARRAY_TASK_ID" "$JABS_START" "$JABS_NUM_FILES"
        exit 0
    fi
}

# Activate the virtualenv recorded at submission time, if there is one.
jabs_worker_activate_venv() {
    [[ -n ${JABS_VENV:-} ]] || return 0
    [[ -r "${JABS_VENV}/bin/activate" ]] \
        || jabs_worker_fail "no virtualenv activate script at ${JABS_VENV}/bin/activate"
    # shellcheck disable=SC1091
    source "${JABS_VENV}/bin/activate"
}

# Print the banner that identifies this task in the Slurm log.
# Extra "Label: value" pairs may be passed as arguments.
jabs_worker_banner() {
    printf '==========================================================\n'
    printf 'Task %s on %s\n' "$SLURM_ARRAY_TASK_ID" "$(hostname)"
    printf 'Manifest: %s\n' "$JABS_MANIFEST"
    local pair
    for pair in "$@"; do
        printf '%s\n' "$pair"
    done
    printf 'Files %d..%d of %d (%d this task)\n' \
        "$JABS_START" "$(( JABS_END - 1 ))" "$JABS_NUM_FILES" "$(( JABS_END - JABS_START ))"
    printf 'Started:  %s\n' "$(date)"
    printf '==========================================================\n'
}

# Run the whole task: resolve the slice, prepare the environment, then call
# the caller's per-file function once per file in the slice.
#
# $1: name of a function taking a single input file path. It should return
#     non-zero to mark that file as failed.
# Remaining arguments are extra banner lines.
#
# Exits non-zero if any file in the slice failed, so `sacct` shows the task as
# failed and the failing files can be pulled out of the log.
jabs_worker_main() {
    local process_one=$1
    shift

    jabs_worker_slice
    jabs_worker_banner "$@"

    if [[ -n ${JABS_OUT_DIR:-} ]]; then
        mkdir -p -- "$JABS_OUT_DIR" || jabs_worker_fail "cannot create ${JABS_OUT_DIR}"
    fi

    jabs_worker_activate_venv

    local failed=0
    local failed_files=()
    local i input_file rc total
    total=$(( JABS_END - JABS_START ))

    for (( i = JABS_START; i < JABS_END; i++ )); do
        input_file=${JABS_FILES[$i]}
        printf '\n--- [%d/%d] index %d: %s\n' \
            "$(( i - JABS_START + 1 ))" "$total" "$i" "$(basename -- "$input_file")"
        printf -- '--- %s\n' "$(date +%H:%M:%S)"

        if [[ ! -r $input_file ]]; then
            printf -- '--- SKIPPED (unreadable): %s\n' "$input_file" >&2
            failed=$(( failed + 1 ))
            failed_files+=("$input_file")
            continue
        fi

        if "$process_one" "$input_file"; then
            printf -- '--- OK\n'
        else
            rc=$?
            printf -- '--- FAILED (exit %d): %s\n' "$rc" "$input_file" >&2
            failed=$(( failed + 1 ))
            failed_files+=("$input_file")
        fi
    done

    printf '\n==========================================================\n'
    printf 'Task %s finished: %s\n' "$SLURM_ARRAY_TASK_ID" "$(date)"
    printf 'Processed %d file(s), %d failure(s)\n' "$total" "$failed"
    if (( failed > 0 )); then
        printf 'FAILED: %s\n' "${failed_files[@]}" >&2
    fi
    printf '==========================================================\n'

    exit $(( failed > 0 ? 1 : 0 ))
}
