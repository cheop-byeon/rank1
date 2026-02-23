#!/bin/bash

# Master script to submit evaluation jobs for all models with appropriate GPU allocation
# This script determines GPU requirements based on model size and submits SLURM jobs

set -e

# Define models and their GPU requirements (80GB memory per GPU)
declare -A MODEL_GPUS
declare -A MODEL_PATHS
declare -A MODEL_RETRIEVERS

retriever_model="bm25s"  # Default retriever for all models, can be customized per model if needed
# 7B models: ~14-16GB (1 GPU sufficient)
MODEL_GPUS[rank1_7b]=1
MODEL_PATHS[rank1_7b]="./jhu-clsp/rank1-7b"
MODEL_RETRIEVERS[rank1_7b]=$retriever_model

MODEL_GPUS[qwen_7b]=1
MODEL_PATHS[qwen_7b]="./Qwen/Qwen2.5-7B-Instruct"
MODEL_RETRIEVERS[qwen_7b]=$retriever_model

# 14B models: ~28-32GB (1-2 GPUs, use 1 for inference, 2 for batch)
MODEL_GPUS[rank1_14b]=2
MODEL_PATHS[rank1_14b]="./jhu-clsp/rank1-14b"
MODEL_RETRIEVERS[rank1_14b]=$retriever_model

MODEL_GPUS[qwen_14b]=2
MODEL_PATHS[qwen_14b]="./Qwen/Qwen2.5-14B-Instruct"
MODEL_RETRIEVERS[qwen_14b]=$retriever_model

# 32B models: ~64-70GB (2 GPUs with tensor parallelism)
MODEL_GPUS[rank1_32b]=2
MODEL_PATHS[rank1_32b]="./jhu-clsp/rank1-32b"
MODEL_RETRIEVERS[rank1_32b]=$retriever_model

MODEL_GPUS[qwen_32b]=2
MODEL_PATHS[qwen_32b]="./Qwen/Qwen2.5-32B-Instruct"
MODEL_RETRIEVERS[qwen_32b]=$retriever_model
MODEL_GPUS[qwq_32b]=2
MODEL_PATHS[qwq_32b]="./Qwen/QwQ-32B"
MODEL_RETRIEVERS[qwq_32b]=$retriever_model

# 72B models: ~140GB+ (4 GPUs with tensor parallelism for good throughput)
MODEL_GPUS[qwen_72b]=4
MODEL_PATHS[qwen_72b]="./Qwen/Qwen2.5-72B-Instruct"
MODEL_RETRIEVERS[qwen_72b]=$retriever_model

# Other large models
MODEL_GPUS[nvidia_49b]=4
MODEL_PATHS[nvidia_49b]="./nvidia/Llama-3_3-Nemotron-Super-49B-v1"
MODEL_RETRIEVERS[nvidia_49b]=$retriever_model


# Configuration
ACCOUNT="12345678"
PARTITION="accel"
TIME="02-12:59:00"
CPUS=1
MEM_PER_CPU="32G"
DATASET="ids-supp" # or "ids"
CONDA_ENV="/path/to/.conda/envs/rank1"
SPLIT="test"           # Data split: test or dev
DIRECTION="i2c"        # Direction: i2c or c2i

echo "=========================================="
echo "Submitting evaluation jobs for all models"
echo "=========================================="
echo ""

JOBS_SUBMITTED=0
JOBS_FAILED=0

for model in "${!MODEL_GPUS[@]}"; do
    num_gpus=${MODEL_GPUS[$model]}
    model_path=${MODEL_PATHS[$model]}
    retriever=${MODEL_RETRIEVERS[$model]}
    
    echo "Submitting job for: $model"
    echo "  GPU allocation: $num_gpus"
    echo "  Model path: $model_path"
    echo "  Retriever: $retriever"
    echo "  Split: $SPLIT"
    echo "  Direction: $DIRECTION"
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Submit SLURM job with dynamic GPU allocation
    JOB_ID=$(sbatch \
        --job-name="eval_${model}" \
        --account=$ACCOUNT \
        --time=$TIME \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=$CPUS \
        --ntasks-per-node=1 \
        --mem-per-cpu=$MEM_PER_CPU \
        --partition=$PARTITION \
        --gpus=$num_gpus \
        --output="logs/eval_${model}_%j.log" \
        --error="logs/eval_${model}_%j.err" \
        test_rank1_job.sh "$model" "$model_path" "$CONDA_ENV" "$num_gpus" "$retriever" "$SPLIT" "$DIRECTION" \
        | awk '{print $4}')
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Job submitted with ID: $JOB_ID"
        ((JOBS_SUBMITTED++))
    else
        echo "  ✗ Failed to submit job"
        ((JOBS_FAILED++))
    fi
    echo ""
done

echo "=========================================="
echo "Summary:"
echo "  Jobs submitted: $JOBS_SUBMITTED"
echo "  Jobs failed: $JOBS_FAILED"
echo "=========================================="

if [ $JOBS_FAILED -gt 0 ]; then
    exit 1
fi
