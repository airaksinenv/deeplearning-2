#!/bin/bash
#SBATCH --job-name=flower_NN
#SBATCH --account=project_2018566
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --gres=gpu:v100:1,nvme:20
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Own Puhti project and data location.
export PROJECT_ID="${PROJECT_ID:-project_2018566}"
export SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/scratch/${PROJECT_ID}/data/flower_photos}"

# Model must end up here for the assignment.
export MODEL_PATH="${MODEL_PATH:-/projappl/${PROJECT_ID}/models/flower_model_NN.keras}"

# Use one physical GPU split into four logical TensorFlow GPUs.
export EPOCHS="${EPOCHS:-30}"
export LOGICAL_GPU_MEMORY_MB="${LOGICAL_GPU_MEMORY_MB:-2048}"

# Load TensorFlow module if the environment uses CSC modules.
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load tensorflow || true
fi

# Copy data from scratch to node-local disk. This replaces the Allas copy step,
# because this self-study setup has the data in the user's own scratch project.
LOCAL_BASE="${LOCAL_SCRATCH:-${TMPDIR:-/tmp/${USER}/flower_${SLURM_JOB_ID:-manual}}}"
LOCAL_DATA_PARENT="${LOCAL_BASE}/flower_job_data"
LOCAL_DATA_DIR="${LOCAL_DATA_PARENT}/flower_photos"

mkdir -p "${LOCAL_DATA_PARENT}"
mkdir -p "$(dirname "${MODEL_PATH}")"

echo "Job started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Host: $(hostname)"
echo "Project: ${PROJECT_ID}"
echo "Source data: ${SOURCE_DATA_DIR}"
echo "Local data: ${LOCAL_DATA_DIR}"
echo "Model path: ${MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

echo "Copying data to local disk..."
rm -rf "${LOCAL_DATA_DIR}"
cp -a "${SOURCE_DATA_DIR}" "${LOCAL_DATA_DIR}"
export FLOWER_DATA_DIR="${LOCAL_DATA_DIR}"

echo "Starting training_script.py..."
python training_script.py

echo "Starting test_script.py..."
python test_script.py

echo "Training log tail:"
tail -n 30 train_log.out || true

echo "Test log tail:"
tail -n 30 test_log.out || true

echo "Job finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
