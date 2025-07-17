#!/bin/bash
set -e

# Arguments
STEP_DIR="$1"
STEP_NUM="$2"
RESULTS_DIR="${3:-$(dirname "$STEP_DIR")/outputs}"
DATASET="${4:-"judgebench"}"

# Config
GPFS="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL"
# CONTAINER="/lustre/fsw/portfolios/llmservice/users/yizhuj/nemorl/containers/anyscale+ray+2.43.0-py312-cu125_uv.sqsh"
CONTAINER="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/container/nemo-rl:main-3e5481f.squashfs"

# SLURM Config
ACCOUNT="llmservice_modelalignment_ppo"
PARTITION="batch,batch_short,interactive"
TIME="01:00:00"
NODES=1
GPUS=8

# Paths
mkdir -p "$RESULTS_DIR/logs"
LOG="$RESULTS_DIR/logs/step_${STEP_NUM}.out"
ERR="$RESULTS_DIR/logs/step_${STEP_NUM}.err"
OUTPUT="$RESULTS_DIR/step_${STEP_NUM}_${DATASET}_results.json"

# Check if already done
if [ -f "$OUTPUT" ]; then
    # Check if the output file is valid JSON
    if jq empty "$OUTPUT" >/dev/null 2>&1; then
        echo "✓ Step $STEP_NUM already evaluated"
        exit 0
    else
        echo "⚠ Output file exists but contains invalid JSON, will regenerate"
    fi
fi

# Set up environment and submit job using the new format
cd "$GPFS"

HF_HOME=/lustre/fsw/portfolios/llmservice/users/yizhuj/hf_cache \
COMMAND="uv run python examples/run_eval_genrm.py --dataset=${DATASET} ++generation.model_name=${STEP_DIR} ++eval.output_file=${OUTPUT} ++eval.batch_size=1024 ++generation.vllm_cfg.tensor_parallel_size=1 ++generation.vllm_cfg.gpu_memory_utilization=0.7 ++cluster.gpus_per_node=${GPUS} ++cluster.num_nodes=1" \
CONTAINER="$CONTAINER" \
MOUNTS="$GPFS:$GPFS,/lustre:/lustre" \
sbatch \
    --nodes=$NODES \
    --account=$ACCOUNT \
    --job-name=eval_step$STEP_NUM \
    --partition=$PARTITION \
    --time=$TIME \
    --gres=gpu:$GPUS \
    --output=$LOG \
    --error=$ERR \
    ray.sub

echo "📤 Submitted job for step_$STEP_NUM"