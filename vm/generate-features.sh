#!/bin/bash
#
#SBATCH --job-name=jabs-generate-features
#
#SBATCH --qos=batch
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G

# See behavior-classify-batch.sh for more detailed information
# This script is written in the same style, but only generates features

CLASSIFICATION_IMG=/projects/kumar-lab/JABS/JABS-Classify-current.sif

trim_sp() {
    local var="$*"
    # remove leading whitespace characters
    var="${var#"${var%%[![:space:]]*}"}"
    # remove trailing whitespace characters
    var="${var%"${var##*[![:space:]]}"}"
    echo -n "$var"
}

# If the provided file ends with the extension, return itself
# Otherwise, search in descending order
MAX_POSE_VERSION=6
MIN_POSE_VERSION=2
find_pose_file() {
    local in_file="$*"
    if [[ "${in_file##*.}" == 'h5' ]]; then
        echo -n ${in_file}
    else
        cur_pose_version=${MAX_POSE_VERSION}
        prefix="${in_file%.*}"
        while [[ cur_pose_version -ge ${MIN_POSE_VERSION} ]]
        do
            if [[ -f "${prefix}_pose_est_v${cur_pose_version}.h5" ]]; then
                echo -n "${prefix}_pose_est_v${cur_pose_version}.h5"
                break
            else
                ((cur_pose_version--))
            fi
        done
    fi
}

if [[ -z "${SLURM_JOB_ID}" ]]
then
    # The script is being run from command line. We should do a self-submit as an array job
    if [[ ( -f "${1}" ) && ( -n "${2}" ) ]]
    then
        echo "Preparing to submit feature generation on batch file: ${1}"
        batch_line_count=$(wc -l < "${1}")
        echo "Submitting an array job for ${batch_line_count} pose files"

        # Here we perform a self-submit
        sbatch --export=BATCH_FILE="${1}",FEATURE_FOLDER="${2}" --array="1-${batch_line_count}%500" "${0}"
    else
        echo "ERROR: missing classification and/or batch file." >&2
        echo "Expected usage:" >&2
        echo "generate-features.sh BATCH_FILE.txt FEATURE_FOLDER" >&2
        exit 1
    fi
else
    # the script is being run by slurm
    if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]
    then
        echo "ERROR: no SLURM_ARRAY_TASK_ID found. This job should be run as an array" >&2
        exit 1
    fi

    if [[ -z "${FEATURE_FOLDER}" ]]
    then
        echo "ERROR: the FEATURE_FOLDER environment variable is not defined" >&2
        exit 1
    fi

    if [[ -z "${BATCH_FILE}" ]]
    then
        echo "ERROR: the BATCH_FILE environment variable is not defined" >&2
        exit 1
    fi

    # here we use the array ID to pull out the right line from the batch file
    BATCH_LINE=$(trim_sp $(sed -n "${SLURM_ARRAY_TASK_ID}{p;q;}" < "${BATCH_FILE}"))
    echo "BATCH LINE FILE: ${BATCH_LINE}"
    # Try and trim the line to look like a video file (if it is a pose file)
    VIDEO_FILE=$(sed -E 's:(_pose_est_v[0-9]+)?\.(avi|mp4|h5):.avi:' <(echo ${BATCH_LINE}))

    # The batch file can either contain fully qualified paths for files to process OR local paths relative to where the batch file exists
    # Change the working directory to support local paths
    cd "$(dirname "${BATCH_FILE}")"

    # Detect the pose file based on the batch line provided
    POSE_FILE=$(find_pose_file ${BATCH_LINE})

    if [[ ! ( -f "${POSE_FILE}" ) ]]
    then
        echo "ERROR: failed to find pose file for ${BATCH_LINE}" >&2
        exit 1
    fi

    echo "BEGIN PROCESSING: ${POSE_FILE} for ${BATCH_LINE} (${POSE_FILE})"
    module load singularity
    singularity exec "${CLASSIFICATION_IMG}" jabs-features --pose-file "${POSE_FILE}" --feature-dir "${FEATURE_FOLDER}" --use-cm-distances

    echo "FINISHED PROCESSING: ${POSE_FILE}"
fi
