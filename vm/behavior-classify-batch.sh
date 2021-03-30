#!/bin/bash
#
#SBATCH --job-name=behavior-classify
#
#SBATCH --qos=batch
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# This is a self-submitting SLURM script which can be used to classify a batch
# of video poses on JAX's sumner cluster. This script assumes that there will be
# a "behavior-classifier.sif" Singularity VM image in the same directory as this
# script. You can build this VM using the "behavior-classifier-vm.def" Singularity
# definition in the repository or use one that has been pre-built.
#
# This script expects two positional command line arguments:
# 1) a fully qualified path to an exported Rotta classifier HDF5 file
# 2) a fully qualified path to a batch file. A batch file is a plain
#    text file which contains newline separated network IDs for each
#    pose file that should be classified. Note that even though we
#    are performing classification on poses, the batch file is
#    expected to contain "*.avi" entries in order to conform to
#    our network IDs. This batch file should be placed at the root
#    directory of all poses that we want to process. So for example,
#    if we have a bunch of poses in "/a/b/c" one of which is
#    "/a/b/c/d/e_pose_est_v3.h5" we should have a batch file
#    "/a/b/c/mybatch.txt" that contains a "d/e.avi" entry.
#
# Example script usage:
# 
#   ~/behavior-classify-batch.sh \
#       /projects/kumar-lab/USERS/sheppk/temp/rearing-batch-2021-02-23-leilani/rearing-classifiers-2020-02-18/leinani-hession/Rearing_supported_training_20210216_122124.h5 \
#       /projects/kumar-lab/USERS/sheppk/temp/rearing-batch-2021-02-23-leilani/batch.txt
#
# And if we look at the contents of batch.txt:
#
#   head -n 4 /projects/kumar-lab/USERS/sheppk/temp/rearing-batch-2021-02-23-leilani/batch.txt
#
# we see:
#
#   LL1-B2B/2016-05-05_SPD/LL1-3_100492-F-AX27-9-42416-3-S111.avi
#   LL1-B2B/2017-01-01_SPD/LL1-4_002105-M-AX12-5.28571428571429-42640-4-S331.avi
#   LL1-B2B/2017-01-12_SPD/LL1-4_001144-F-F29-4-42661-4-S344.avi
#   LL1-B2B/2016-05-23_SPD/B6J_Male_S6730806_ep3-PSY.avi

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
        sbatch --export=ROOT_DIR="$(dirname "${0}")",CLASSIFIER_FILE="${1}",BATCH_FILE="${2}" --array="1-${batch_line_count}%500" "${0}"
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
    singularity run "${ROOT_DIR}/behavior-classifier.sif" classify --xgboost --training "${CLASSIFIER_FILE}" --input-pose "${POSE_FILE}" --out-dir "${OUT_DIR}"

    echo "FINISHED PROCESSING: ${POSE_FILE}"
fi
