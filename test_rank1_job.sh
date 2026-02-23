#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --account=12345678
#SBATCH --partition=accel

# This script is called by submit_all_evaluations.sh with dynamic parameters
# Usage: test_rank1_job.sh <model_name> <model_path> <conda_env> <num_gpus> <retriever> <split> <direction>

MODEL_NAME=$1
MODEL_PATH=$2
CONDA_ENV=$3
NUM_GPUS=$4
RETRIEVER=$5
SPLIT=$6
DIRECTION=$7

if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_PATH" ] || [ -z "$CONDA_ENV" ] || [ -z "$NUM_GPUS" ] || [ -z "$RETRIEVER" ] || [ -z "$SPLIT" ] || [ -z "$DIRECTION" ]; then
    echo "Error: Missing parameters"
    echo "Usage: $0 <model_name> <model_path> <conda_env> <num_gpus> <retriever> <split> <direction>"
    exit 1
fi

# Setup
module load Miniconda3/22.11.1-1
export PS1=\$
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh
conda deactivate &>/dev/null

echo "Conda environments: $(conda info --envs)"
echo "EBROOTMINCONDA3: ${EBROOTMINICONDA3}"

conda activate "$CONDA_ENV"

# Load Python environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Print debug information
echo "=== Debug Information ==="
echo "Hostname: $(hostname)"
echo "Current directory: $(pwd)"
echo "Date: $(date)"
echo "Model: $MODEL_NAME"
echo "Model Path: $MODEL_PATH"

# Print GPU information
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n=== GPU Information ==="
    nvidia-smi
else
    echo "nvidia-smi not found - no GPU information available"
fi

# Print memory information
echo -e "\n=== Memory Information ==="
free -h

# Print CPU information
echo -e "\n=== CPU Information ==="
lscpu | grep "Model name"
lscpu | grep "CPU(s):"

echo -e "\n=== Environment ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "=== End Debug Information ===\n"

# Use GPU count and retriever passed from submission script
echo "Running evaluation for: $MODEL_NAME"
echo "Using $NUM_GPUS GPUs"
echo "Using retriever: $RETRIEVER"
echo "Split: $SPLIT"
echo "Direction: $DIRECTION"
echo "=========================================="

# Run evaluation with all parameters
python test_rank1.py -d "ids-supp" -n "$NUM_GPUS" -m "$MODEL_PATH" -r "$RETRIEVER" -s "$SPLIT" --direction "$DIRECTION" -p

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully completed: $MODEL_NAME"
    exit 0
else
    echo ""
    echo "✗ Failed: $MODEL_NAME"
    exit 1
fi
