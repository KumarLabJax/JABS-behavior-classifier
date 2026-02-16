#!/bin/bash
#
#SBATCH --job-name=infer-pose-v8-arr
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu_inference
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --output=runs/inference-logs/slurm-%A_%a.txt

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

    # the script is being run by slurm
    DET_CONFIG="${SCRIPT_DIR}/../configs/custom_pose/mouse_detector_retinanet.py"
    DET_CHECKPOINT="${SCRIPT_DIR}/../runs/mouse_detector/best_single_class_ap_AP@0.75_epoch_28.pth"
    POSE_CONFIG="${SCRIPT_DIR}/../configs/custom_pose/topdown_resnet18.py"
    POSE_CHECKPOINT="${SCRIPT_DIR}/../runs/custom_pose/best_PCK_epoch_60.pth"

    MAX_INSTANCES=3
    BATCH_SIZE=4

    # here we use the array ID to pull out the right video
    VIDEO_FILE="$(trim_sp "$(sed -n "${SLURM_ARRAY_TASK_ID}{p;q;}" < "${BATCH_FILE}")")"
    echo "VIDEO: ${VIDEO_FILE}"
    echo "SLURM JOB ID: ${SLURM_JOB_ID}"
    echo "SLURM ARRAY TASK ID: ${SLURM_ARRAY_TASK_ID}"
    echo "DET CONFIG: ${DET_CONFIG}"
    echo "DET CHECKPOINT: ${DET_CHECKPOINT}"
    echo "POSE CONFIG: ${POSE_CONFIG}"
    echo "POSE CHECKPOINT: ${POSE_CHECKPOINT}"
    echo "MAX INSTANCES: ${MAX_INSTANCES}"
    echo "BATCH SIZE: ${BATCH_SIZE}"
    echo ""
    echo ""
    echo "DUMP OF CURRENT ENVIRONMENT:"
    env
    echo ""
    echo ""
    echo "BEGIN PROCESSING: ${VIDEO_FILE}"

    # ---- Logging setup ----
    OUTPUT_DIR="runs/inference-logs"
    mkdir -p "${OUTPUT_DIR}"
    
    if [[ ! -f "${VIDEO_FILE}" ]]
    then
        echo "ERROR: Video file does not exist: ${VIDEO_FILE}" >&2
        exit 1
    fi

    # start logging GPU usage
    GPU_LOG_FILE="${OUTPUT_DIR}/gpu-usage-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
        --format=csv,nounits -l 5 > "$GPU_LOG_FILE" 2>&1 &
    NVIDIA_SMI_PID=$!
    trap "kill $NVIDIA_SMI_PID 2>/dev/null" EXIT INT TERM

    H5_OUT_FILE="${VIDEO_FILE%.*}_pose_est_v8.h5"
    module load apptainer
    export APPTAINERENV_PYTHONHTTPSVERIFY=0
    export APPTAINERENV_PYTHONPATH="${SCRIPT_DIR}/../src"
    apptainer run --nv "${SCRIPT_DIR}/../mmpose.sif" \
        python "${SCRIPT_DIR}/vid_infer_detector_then_pose.py" \
        "${VIDEO_FILE}" \
        --det-config "${DET_CONFIG}" \
        --det-checkpoint "${DET_CHECKPOINT}" \
        --pose-config "${POSE_CONFIG}" \
        --pose-checkpoint "${POSE_CHECKPOINT}" \
        --max-instances "${MAX_INSTANCES}" \
        --batch-size "${BATCH_SIZE}" \
        --out-pose-v8 "${H5_OUT_FILE}"

    # stop logging GPU on exit (should be redundant but just in case)
    kill $NVIDIA_SMI_PID 2>/dev/null

    if [[ ! -f "${H5_OUT_FILE}" ]]
    then
        echo "ERROR: FAILED TO GENERATE OUTPUT FILE"
        exit 1
    fi

    echo "FINISHED PROCESSING: ${VIDEO_FILE}"
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
        sbatch --chdir="${BATCH_FILE_DIR}" --output="runs/inference-logs/slurm-%A_%a.txt" --export="BATCH_FILE=${BATCH_FILE},SCRIPT_DIR=${SCRIPT_DIR}" --array="1-${test_count}%24" "${0}"
    else
        echo "ERROR: you need to provide a batch file to process. Eg: ./scripts/infer_pose_v3_batch.sh batchfile.txt" >&2
        exit 1
    fi
fi
