#!/bin/bash
#
#SBATCH --job-name=behavior-classify
#
#SBATCH --qos=batch
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G

# This is a self-submitting SLURM script which can be used to classify a batch
# of video poses on JAX's sumner cluster. This script assumes that there will be
# a Singularity VM image located on the cluster (CLASSIFICATION_IMG) to use.
# You can build this VM using the "behavior-classifier-vm.def" Singularity
# definition in the repository or use one that has been pre-built.
#
# This script expects two positional command line arguments:
# 1) a fully qualified path to an exported JABS classifier HDF5 file
# 2) a fully qualified path to a batch file. A batch file is a plain
#    text file which contains newline separated network IDs for each
#    pose file that should be classified. Note that one can provide
#    either a pose file ending with a *.h5 extension or a video file
#    with the assumption that *.avi also possesses a *_pose_est_v##.h5.
#
# Example script usage:
# 
#   /projects/kumar-lab/JABS/behavior-classify-batch.sh \
#       /projects/kumar-lab/JABS/classifiers/Rearing_supported_training_20210216_122124.h5 \
#       /projects/kumar-lab/JABS/batches/test-batch.txt
#
# And if we look at the contents of test-batch.txt:
#
#   head -n 4 /projects/kumar-lab/JABS/batches/test-batch.txt
#
# we see:
#
#   /projects/kumar-lab/video-data/LL1-B2B/2016-05-05_SPD/LL1-3_100492-F-AX27-9-42416-3-S111.avi
#   /projects/kumar-lab/video-data/LL1-B2B/2017-01-01_SPD/LL1-4_002105-M-AX12-5.28571428571429-42640-4-S331.avi
#   /projects/kumar-lab/video-data/LL1-B2B/2017-01-12_SPD/LL1-4_001144-F-F29-4-42661-4-S344.avi
#   /projects/kumar-lab/video-data/LL1-B2B/2016-05-23_SPD/B6J_Male_S6730806_ep3-PSY.avi
# 
# Performance Notes for adjusting job requests:
#
# Inputs: Features:
#   base + social + landmark (pose_v5)
#   3 animals
#   xgboost classifier
# Cluster Specs:
#   2.7GHz cpu
#   DDR4 RAM
#   Isilon NAS Storage (~5GB/s read/write speed)
# Expected Resources: 
#   22-28m time to compute
#   1.3-2.2GB RAM usage
# Max Resources:
#   35m time to compute
#   3GB RAM usage

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
MAX_POSE_VERSION=5
MIN_POSE_VERSION=2
find_pose_file() {
    local in_file="$*"
    if [[ "${in_file##*.}" == 'h5' ]]; then
        echo -n ${in_file}
    else
        cur_pose_version=${MAX_POSE_VERSION}
        prefix="${in_file%.*}"
        while [[ cur_pose_version -gt ${MIN_POSE_VERSION} ]]
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
    if [[ ( -f "${1}" ) && ( -f "${2}" ) ]]
    then
        # echo "${1} is set and not empty"
        echo "Preparing to submit classification using ${1} on batch file: ${2}"
        batch_line_count=$(wc -l < "${2}")
        echo "Submitting an array job for ${batch_line_count} videos"

        # Here we perform a self-submit
        sbatch --export=CLASSIFIER_FILE="${1}",BATCH_FILE="${2}" --array="1-${batch_line_count}%500" "${0}"
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
        echo "ERROR: no SLURM_ARRAY_TASK_ID found. This job should be run as an array" >&2
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

    # here we use the array ID to pull out the right line from the batch file
    BATCH_LINE=$(trim_sp $(sed -n "${SLURM_ARRAY_TASK_ID}{p;q;}" < "${BATCH_FILE}"))
    echo "BATCH LINE FILE: ${BATCH_LINE}"
    # Try and trim the line to look like a video file (if it is a pose file)
    VIDEO_FILE=$(sed -E 's:(_pose_est_v[0-9]+)?\.(avi|h5):.avi:' <(echo ${BATCH_LINE}))

    # the "v1" is for output format versioning. If format changes this should be updated
    OUT_DIR="${VIDEO_FILE%.*}_behavior/v1"

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

    echo "DUMP OF CURRENT ENVIRONMENT:"
    env
    echo "BEGIN PROCESSING: ${POSE_FILE} for ${BATCH_LINE} (${POSE_FILE}"
    module load singularity
    singularity run "${CLASSIFICATION_IMG}" classify --training "${CLASSIFIER_FILE}" --input-pose "${POSE_FILE}" --out-dir "${OUT_DIR}"

    echo "FINISHED PROCESSING: ${POSE_FILE}"
fi
