#!/bin/bash
#
#SBATCH --job-name=train-topdown-resnet18
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu_training
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=runs/slurm-output-%j.txt

echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU(s) allocated:"
nvidia-smi -L
echo "GPU(s) on node (unmasked):"
CUDA_VISIBLE_DEVICES= nvidia-smi -L
echo "Job ID: ${SLURM_JOB_ID}"
start_time=$(date +%s)

# ---- Logging setup ----
# SLURM's `#SBATCH --output=...` path must exist when the job starts, so we write to a stable
# location under `runs/` first, then hardlink that file into the per-job run directory and
# unlink the original path. SLURM continues writing to the same inode, so the log effectively
# "moves" into `runs/topdown_resnet18/${SLURM_JOB_ID}/` (same-filesystem required).
RUNS_JOB_DIR="runs/topdown_resnet18/${SLURM_JOB_ID}"
mkdir -p "$RUNS_JOB_DIR"

LOCAL_SLURM_OUT="runs/slurm-output-${SLURM_JOB_ID}.txt"
RUNS_SLURM_OUT="${RUNS_JOB_DIR}/slurm-output-${SLURM_JOB_ID}.txt"

while [ ! -f "$LOCAL_SLURM_OUT" ]; do
    echo "Waiting for local SLURM output file to be created: $LOCAL_SLURM_OUT"
    sleep 1
done

if ln "$LOCAL_SLURM_OUT" "$RUNS_SLURM_OUT" 2>/dev/null; then
    rm -f "$LOCAL_SLURM_OUT"
else
    echo "WARN: failed to hardlink '$LOCAL_SLURM_OUT' -> '$RUNS_SLURM_OUT' (same filesystem required); leaving local output file in place."
fi

# start logging GPU usage
GPU_LOG_FILE="${RUNS_JOB_DIR}/gpu-usage-${SLURM_JOB_ID}.csv"
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,nounits -l 5 > "$GPU_LOG_FILE" 2>&1 &
NVIDIA_SMI_PID=$!

# stop logging GPU on exit/preemption
trap "kill $NVIDIA_SMI_PID 2>/dev/null" EXIT INT TERM

echo "========== STARTING TRAINING =========="

# Run container with the config file passed in
module load apptainer
export APPTAINERENV_PYTHONHTTPSVERIFY=0
apptainer run --nv mmpose.sif python src/train_custom_pose.py --config configs/custom_pose/topdown_resnet18.py

# stop logging GPU on exit (should be redundant but just in case)
kill $NVIDIA_SMI_PID 2>/dev/null

# Time logging
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "========== HYDRA TRAINING COMPLETE =========="
echo "End time: $(date)"
echo "Elapsed time: $((elapsed / 3600))h $((elapsed % 3600 / 60))m $((elapsed % 60))s"
