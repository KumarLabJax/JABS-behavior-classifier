#!/bin/bash
#
#SBATCH --job-name=render-pose-v8-arr
#SBATCH --qos=batch
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --output=runs/inference-logs/render-slurm-%A_%a.txt

trim_sp() {
    local var="$*"
    # remove leading whitespace characters
    var="${var#"${var%%[![:space:]]*}"}"
    # remove trailing whitespace characters
    var="${var%"${var##*[![:space:]]}"}"
    echo -n "$var"
}

if [[ -n "${SLURM_JOB_ID}" ]]
then

    if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]
    then
        echo "ERROR: SLURM_ARRAY_TASK_ID is not set; this script must be run as part of a SLURM array job." >&2
        exit 1
    fi

    if [[ -z "${BATCH_FILE}" ]]
    then
        echo "ERROR: BATCH_FILE environment variable is not set; cannot proceed." >&2
        exit 1
    fi

    if [[ -z "${SCRIPT_DIR}" ]]
    then
        echo "ERROR: SCRIPT_DIR environment variable is not set; cannot proceed." >&2
        exit 1
    fi

    # here we use the array ID to pull out the right video
    VIDEO_FILE="$(trim_sp "$(sed -n "${SLURM_ARRAY_TASK_ID}{p;q;}" < "${BATCH_FILE}")")"
    POSE_FILE="${VIDEO_FILE%.*}_pose_est_v8.h5"
    RENDERED_VIDEO_FILE="${VIDEO_FILE%.*}_pose_est_v8_render.mp4"

    if [[ ! -f "${VIDEO_FILE}" ]]
    then
        echo "ERROR: Video file does not exist: ${VIDEO_FILE}" >&2
        exit 1
    fi

    if [[ ! -f "${POSE_FILE}" ]]
    then
        echo "ERROR: Pose file does not exist: ${POSE_FILE}" >&2
        exit 1
    fi

    module load apptainer
    export APPTAINERENV_PYTHONHTTPSVERIFY=0
    export APPTAINERENV_PYTHONPATH="${SCRIPT_DIR}/../src"
    apptainer run "${SCRIPT_DIR}/../mmpose.sif" \
        python "${SCRIPT_DIR}/render_pose_on_video.py" \
        "${VIDEO_FILE}" \
        "${POSE_FILE}" \
        "${RENDERED_VIDEO_FILE}" \
        --exclude-points 4

    if [[ ! -f "${RENDERED_VIDEO_FILE}" ]]
    then
        echo "ERROR: FAILED TO RENDER VIDEO: ${RENDERED_VIDEO_FILE}" >&2
        exit 1
    fi

    echo "FINISHED RENDERING POSE OVERLAY FOR: ${VIDEO_FILE}"
else
    # the script is being run from command line. We should do a self-submit as an array job
    if [[ -f "${1}" ]]
    then
        echo "Preparing to submit batch file: ${1}"
        test_count=$(wc -l < "${1}")
        echo "Submitting an array job for ${test_count} videos"

        BATCH_FILE="$(realpath "${1}")"
        BATCH_FILE_DIR="$(dirname "${BATCH_FILE}")"
        mkdir -p "${BATCH_FILE_DIR}/runs/inference-logs"

        SCRIPT_DIR="$(dirname "$(realpath "${0}")")"

        # Here we perform a self-submit
        array_value="1-${test_count}"
        if [[ -n "${2}" ]]
        then
            array_value="${2}"
        fi

        sbatch --chdir="${BATCH_FILE_DIR}" --output="runs/inference-logs/render-slurm-%A_%a.txt" --export="BATCH_FILE=${BATCH_FILE},SCRIPT_DIR=${SCRIPT_DIR}" --array="${array_value}" "${0}"
    else
        echo "ERROR: you need to provide a batch file to process. Eg: ./scripts/render_pose_v3_batch.sh batchfile.txt" >&2
        exit 1
    fi
fi
