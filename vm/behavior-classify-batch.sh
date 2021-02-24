#!/bin/bash
#
#SBATCH --job-name=behavior-classify
#
#SBATCH --qos=batch
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

trim_sp() {
    local var="$*"
    # remove leading whitespace characters
    var="${var#"${var%%[![:space:]]*}"}"
    # remove trailing whitespace characters
    var="${var%"${var##*[![:space:]]}"}"
    echo -n "$var"
}

if [[ -z "${SLURM_JOB_ID}" ]]
then
    # the script is being run from command line. We should do a self-submit as an array job
    if [[ ( -f "${1}" ) && ( -f "${2}" ) ]]
    then
        # echo "${1} is set and not empty"
        echo "Preparing to submit classification using ${1} on batch file: ${2}"
        batch_line_count=$(wc -l < "${2}")
        echo "Submitting an array job for ${batch_line_count} videos"

        # Here we perform a self-submit
        sbatch --export=ROOT_DIR="$(dirname "${0}")",CLASSIFIER_FILE="${1}",BATCH_FILE="${2}" --array="1-${batch_line_count}" "${0}"
    else
        echo "ERROR: missing classification and/or batch file." >&2
        echo "Expected usage:" >&2
        echo "behavior-classify-batch.sh CLASSIFIER.h5 BATCH_FILE.txt" >&2
        exit 1
    fi
else
    # the script is being run by slurm
    if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]
    then
        echo "ERROR: no SLURM_ARRAY_TASK_ID found" >&2
        exit 1
    fi

    if [[ -z "${CLASSIFIER_FILE}" ]]
    then
        echo "ERROR: the CLASSIFIER_FILE environment variable is not defined" >&2
        exit 1
    fi

    if [[ -z "${BATCH_FILE}" ]]
    then
        echo "ERROR: the BATCH_FILE environment variable is not defined" >&2
        exit 1
    fi

    # here we use the array ID to pull out the right video from the batch file
    VIDEO_FILE=$(trim_sp $(sed -n "${SLURM_ARRAY_TASK_ID}{p;q;}" < "${BATCH_FILE}"))
    echo "BATCH VIDEO FILE: ${VIDEO_FILE}"

    # the "v1" is for output format versioning. If format changes this should be updated
    OUT_DIR="${VIDEO_FILE%.*}_behavior/v1"

    cd "$(dirname "${BATCH_FILE}")"
    POSE_FILE_V3="${VIDEO_FILE%.*}_pose_est_v3.h5"
    POSE_FILE="${POSE_FILE_V3}"

    if [[ ! ( -f "${POSE_FILE}" ) ]]
    then
        POSE_FILE_V2="${VIDEO_FILE%.*}_pose_est_v2.h5"
        POSE_FILE="${POSE_FILE_V2}"
    fi

    if [[ ! ( -f "${POSE_FILE}" ) ]]
    then
        echo "ERROR: failed to find either pose file (${POSE_FILE_V2} or ${POSE_FILE_V3}) for ${VIDEO_FILE}" >&2
        exit 1
    fi

    echo "DUMP OF CURRENT ENVIRONMENT:"
    env
    echo "BEGIN PROCESSING: ${POSE_FILE} for ${VIDEO_FILE}"
    module load singularity
    singularity run "${ROOT_DIR}/behavior-classifier.sif" --xgboost --training "${CLASSIFIER_FILE}" --input-pose "${POSE_FILE}" --out-dir "${OUT_DIR}"

    echo "FINISHED PROCESSING: ${POSE_FILE}"
fi
