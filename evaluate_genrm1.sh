#!/bin/bash

set -e

# Configuration
GPFS="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL"
CONTAINER="/lustre/fsw/portfolios/llmservice/users/yizhuj/nemorl/containers/anyscale+ray+2.43.0-py312-cu125_uv.sqsh"

# SLURM configuration
ACCOUNT="llmservice_modelalignment_ppo"
PARTITION="batch"
TIME_LIMIT="04:00:00"
NODES=1
CPUS_PER_TASK=8
GPUS_PER_NODE=1

# Base directory for your specific checkpoint structure
BASE_DIR="${1:-/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/results/grpo_hs3_16K_step240_clip_max_0.28_llama3.1_8B_lr_2e-6_temp_1_kl_0.001_grpo_bs_64_rollout_8_num_prompts_64}"
HF_DIR="${BASE_DIR}/HF"  # Your checkpoints are under HF/step_X
RESULTS_DIR="${2:-/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/outputs/genrm_eval_results/grpo_hs3_16K_step240_clip_max_0.28_llama3.1_8B_lr_2e-6_temp_1_kl_0.001_grpo_bs_64_rollout_8_num_prompts_64}"

# Create results directory
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/logs"

echo "Submitting GenRM Evaluation Jobs"
echo "Base Directory: $BASE_DIR"
echo "HF Directory: $HF_DIR"
echo "Results Directory: $RESULTS_DIR"
echo "Container image: $CONTAINER"
echo "----------------------------------------"

# Track submitted jobs
SUBMITTED_JOBS=()
SKIPPED_STEPS=()


# Function to submit a single evaluation job
submit_evaluation_job() {
    local step_dir="$1"
    local step_num="$2"
    
    local job_name="eval_genrm_step${step_num}"
    local log_file="$RESULTS_DIR/logs/step_${step_num}.out"
    local err_file="$RESULTS_DIR/logs/step_${step_num}.err"
    
    # Set up container mounts
    local mounts="--container-mounts=${GPFS}:${GPFS},/lustre:/lustre"
    
    # Define paths as they will appear INSIDE the container
    local container_model_path="$step_dir"
    local container_output_file="$RESULTS_DIR/step_${step_num}_judgebench_results.json"

    # Submit SLURM job
    local job_output
    job_output=$(sbatch <<EOF
#!/bin/bash
#SBATCH -A $ACCOUNT
#SBATCH -J $job_name
#SBATCH -p $PARTITION
#SBATCH -t $TIME_LIMIT
#SBATCH -N $NODES
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH -o $log_file
#SBATCH -e $err_file

export HF_HOME=/lustre/fsw/portfolios/llmservice/users/yizhuj/hf_cache
export NCCL_ALGO=Tree

srun --container-image="$CONTAINER" $mounts bash -c '
    # Enable error-exit and execution tracing inside the container
    set -e
    set -x

    # Change to the NeMo-RL directory and run evaluation (exact same pattern as convert script)
    cd "'"$GPFS"'" \
    && ulimit -c 0 \
    && pwd \
    && ls -la examples/ | head -10 \
    && which uv \
    && uv --version \
    && uv run python examples/run_eval_genrm.py \
        --dataset judgebench \
        ++generation.model_name='"$container_model_path"' \
        ++eval.output_file='"$container_output_file"' \
        ++eval.batch_size=256 \
        ++generation.vllm_cfg.tensor_parallel_size=1 \
        ++generation.vllm_cfg.gpu_memory_utilization=0.7 \
        ++cluster.gpus_per_node=1 \
        ++cluster.num_nodes=1

    echo "Evaluation successful for step_'"$step_num"'."
'
EOF
)
    
    local job_id=$(echo "$job_output" | grep -oE '[0-9]+' | tail -1)
    echo "  📤 Submitted job $job_id for step_$step_num"
    SUBMITTED_JOBS+=("$job_id:step_$step_num")
}

# Find all step directories under HF/ and sort them numerically
echo "Finding checkpoints in $HF_DIR..."
if [ -d "$HF_DIR" ]; then
    STEP_DIRS=$(find $HF_DIR -type d -name "step_*" | sort -V)
else
    echo "HF directory not found. Looking for step directories in base directory..."
    STEP_DIRS=$(find $BASE_DIR -type d -name "step_*" | sort -V)
fi

if [ -z "$STEP_DIRS" ]; then
    echo "No checkpoint directories found!"
    exit 1
fi

echo "Found checkpoints:"
echo "$STEP_DIRS"
echo ""

# Process each step directory and submit jobs
for step_dir in $STEP_DIRS; do
    if [ -d "$step_dir" ]; then
        # Extract step number from directory name (anchor to end of string)
        step_basename=$(basename "$step_dir")
        step_num=$(echo "$step_basename" | grep -Eo '[0-9]+$')
        
        if [ ! -z "$step_num" ]; then
            echo "Processing directory: $step_dir"
            echo "Step number: $step_num"
            
            # Check if output already exists
            output_file="$RESULTS_DIR/step_${step_num}_judgebench_results.json"
            if [ -f "$output_file" ]; then
                echo "  ⚠️  Output file already exists: $output_file"
                echo "  ⏭️  Skipping step_$step_num"
                SKIPPED_STEPS+=("step_$step_num (already exists)")
                continue
            fi
            
            # Submit evaluation job
            submit_evaluation_job "$step_dir" "$step_num"
            
        else
            echo "Could not extract step number from $step_basename"
            SKIPPED_STEPS+=("$step_basename (invalid name)")
        fi
    fi
done

echo ""
echo "========================================"
echo "Job Submission Summary"
echo "========================================"
echo "Submitted jobs: ${#SUBMITTED_JOBS[@]}"
for job_info in "${SUBMITTED_JOBS[@]}"; do
    echo "  - $job_info"
done

if [ ${#SKIPPED_STEPS[@]} -gt 0 ]; then
    echo ""
    echo "Skipped steps: ${#SKIPPED_STEPS[@]}"
    for skip_info in "${SKIPPED_STEPS[@]}"; do
        echo "  - $skip_info"
    done
fi

echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: $RESULTS_DIR/logs/"
echo ""
