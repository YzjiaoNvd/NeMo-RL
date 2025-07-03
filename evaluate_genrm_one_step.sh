#!/bin/bash
set -e

# Arguments
STEP_DIR="$1"
STEP_NUM="$2"
RESULTS_DIR="${3:-$(dirname "$STEP_DIR")/outputs}"
DATASET="${4:-"judgebench"}"


# Config
GPFS="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL"
CONTAINER="/lustre/fsw/portfolios/llmservice/users/yizhuj/nemorl/containers/anyscale+ray+2.43.0-py312-cu125_uv.sqsh"

# SLURM

ACCOUNT="llmservice_modelalignment_ppo"
PARTITION="batch_short"
TIME="01:00:00"
NODES=1
GPUS=1


# Paths
mkdir -p "$RESULTS_DIR/logs"
LOG="$RESULTS_DIR/logs/step_${STEP_NUM}.out"
ERR="$RESULTS_DIR/logs/step_${STEP_NUM}.err"
OUTPUT="$RESULTS_DIR/step_${STEP_NUM}_${DATASET}_results.json"

# Generate unique identifiers for this job
UNIQUE_ID=$(shuf -i 1000-9999 -n 1)
RAY_TEMP_DIR="/tmp/ray_${UNIQUE_ID}"

# Check if already done
if [ -f "$OUTPUT" ]; then
    echo "✓ Step $STEP_NUM already evaluated"
    exit 0
fi



# Submit job
JOB=$( sbatch -A $ACCOUNT -J eval_step$STEP_NUM -p $PARTITION -t $TIME \
    -N $NODES --gpus-per-node=$GPUS -o $LOG -e $ERR << EOF
#!/bin/bash
export HF_HOME=/lustre/fsw/portfolios/llmservice/users/yizhuj/hf_cache

srun --container-image="$CONTAINER" \
    --container-mounts="${GPFS}:${GPFS},/lustre:/lustre" bash -c '
    set -ex
    cd "$GPFS"
    
    # Set unique venv suffix to avoid conflicts between jobs
    export NEMO_RL_VENV_SUFFIX="_${UNIQUE_ID}"
    
    # Ray configuration
    export RAY_DISABLE_IMPORT_WARNING=1 
    export RAY_worker_register_timeout_seconds=60 
    export RAY_worker_startup_timeout_seconds=180 
    export RAY_ADDRESS="" 
    export RAY_TMPDIR="${RAY_TEMP_DIR}" 
    mkdir -p "${RAY_TEMP_DIR}" 

    # Run evaluation - let NeMo-RL handle venv creation automatically
    uv run python examples/run_eval_genrm.py \
        --dataset="${DATASET}" \
        ++generation.model_name="$STEP_DIR" \
        ++eval.output_file="$OUTPUT" \
        ++eval.batch_size=256 \
        ++generation.vllm_cfg.tensor_parallel_size=1 \
        ++generation.vllm_cfg.gpu_memory_utilization=0.7 \
        ++cluster.gpus_per_node=1 \
        ++cluster.num_nodes=1 
    
    # Cleanup
    rm -rf "${RAY_TEMP_DIR}" 2>/dev/null || true

    
'
EOF
)

echo "📤 Submitted job ${JOB##* } for step_$STEP_NUM"
sleep 60


