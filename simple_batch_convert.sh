#!/bin/bash

set -e

# Input arguments
CHECKPOINT_DIR="$1"
THRESHOLD="${2:-0}"
OUTPUT_BASE_DIR="${3:-${CHECKPOINT_DIR}/HF}"

# Container configuration
CONTAINER="${NEMO_RL_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/yizhuj/nemorl/containers/anyscale+ray+2.43.0-py312-cu125_uv.sqsh}"

# SLURM configuration
ACCOUNT="llmservice_modelalignment_ppo"
PARTITION="batch"
TIME_LIMIT="04:00:00"
NODES=1
CPUS_PER_TASK=4
MEM_PER_CPU="8G"
GPUS_PER_NODE=8

# Validate inputs
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not provided"
    echo "Usage: $0 <checkpoint_dir> [threshold] [output_base_dir]"
    echo "  checkpoint_dir: Directory containing step_* subdirectories"
    echo "  threshold: Optional minimum step number to process (default: 0)" 
    echo "  output_base_dir: Optional base directory for HF outputs (default: <checkpoint_dir>/HF)"
    echo ""
    echo "Environment variables for SLURM:"
    echo "  SLURM_ACCOUNT: SLURM account (default: nemo_rl)"
    echo "  SLURM_PARTITION: SLURM partition (default: batch)"
    echo "  SLURM_TIME: Time limit (default: 1:00:00)"
    echo "  SLURM_GPUS: GPUs per node (default: 0)"
    echo "  NEMO_RL_CONTAINER: Container image path (required)"
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory '$CHECKPOINT_DIR' does not exist"
    exit 1
fi

if [ ! -f "$CONTAINER" ]; then
    echo "Error: Container image not found at '$CONTAINER'"
    echo "Set NEMO_RL_CONTAINER environment variable to the correct path"
    exit 1
fi

# Find the convert_dcp_to_hf.py script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_SCRIPT="$SCRIPT_DIR/examples/convert_dcp_to_hf.py"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "Error: convert_dcp_to_hf.py not found at $CONVERT_SCRIPT"
    echo "Make sure this script is in the same directory as convert_dcp_to_hf.py"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$OUTPUT_BASE_DIR/logs"

echo "Submitting DCP to HuggingFace conversion jobs"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Step threshold: $THRESHOLD"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Container image: $CONTAINER"
echo "SLURM account: $ACCOUNT"
echo "SLURM partition: $PARTITION"
echo "Time limit: $TIME_LIMIT"
echo "GPUs per node: $GPUS_PER_NODE"
echo "----------------------------------------"

# Track submitted jobs
SUBMITTED_JOBS=()
SKIPPED_STEPS=()

# Function to submit a single conversion job
submit_conversion_job() {
    local step_dir="$1"
    local step_num="$2"
    local config_path="$3"
    local dcp_weights_path="$4"
    local hf_output_path="$5"
    
    local job_name="dcp2hf_step${step_num}"
    local log_file="$OUTPUT_BASE_DIR/logs/step_${step_num}.out"
    local err_file="$OUTPUT_BASE_DIR/logs/step_${step_num}.err"
    
    # Set up container mounts
    local mounts="--container-mounts=$CHECKPOINT_DIR:/checkpoint_dir,$OUTPUT_BASE_DIR:/output_dir,$SCRIPT_DIR:/scripts"
    
    # Convert absolute paths to container paths
    local container_config_path="/checkpoint_dir/$(basename "$step_dir")/config.yaml"
    local container_dcp_path="/checkpoint_dir/$(basename "$step_dir")/policy/weights"
    local container_hf_path="/output_dir/step_${step_num}"
    local container_script_path="/scripts/convert_dcp_to_hf.py"
    
    # Submit SLURM job
    local job_output=$(sbatch <<EOF
#!/bin/bash
#SBATCH -A $ACCOUNT
#SBATCH -J $job_name
#SBATCH -p $PARTITION
#SBATCH -t $TIME_LIMIT
#SBATCH -N $NODES
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem-per-cpu=$MEM_PER_CPU
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH -o $log_file
#SBATCH -e $err_file

set -x

read -r -d '' cmd <<SCRIPT_EOF
set -e

echo "Converting DCP checkpoint step_$step_num"
echo "Config: $container_config_path"
echo "DCP weights: $container_dcp_path" 
echo "HF output: $container_hf_path"
echo "Started at: \\\$(date)"

# Create output directory
mkdir -p "$container_hf_path"

# Run conversion
python "$container_script_path" \\\\
    --config "$container_config_path" \\\\
    --dcp-ckpt-path "$container_dcp_path" \\\\
    --hf-ckpt-path "$container_hf_path"

echo "Completed at: \\\$(date)"
echo "Conversion successful for step_$step_num"
SCRIPT_EOF

srun --container-image="$CONTAINER" $mounts bash -c "\\\${cmd}"
set +x
EOF
)
    
    local job_id=$(echo "$job_output" | grep -oE '[0-9]+' | tail -1)
    echo "  📤 Submitted job $job_id for step_$step_num"
    SUBMITTED_JOBS+=("$job_id:step_$step_num")
}

# Process each step directory
for step_dir in "$CHECKPOINT_DIR"/step_*; do
    if [ -d "$step_dir" ]; then
        # Extract step number from directory name
        step_basename=$(basename "$step_dir")
        step_num=$(echo "$step_basename" | grep -Eo "[0-9]+" | head -n1)
        
        if [ ! -z "$step_num" ] && [ "$step_num" -gt "$THRESHOLD" ]; then
            echo "Processing directory: $step_dir"
            echo "Step number: $step_num"
            
            # Define paths
            config_path="$step_dir/config.yaml"
            dcp_weights_path="$step_dir/policy/weights"
            hf_output_path="$OUTPUT_BASE_DIR/step_$step_num"
            
            # Validate required files exist
            if [ ! -f "$config_path" ]; then
                echo "  ❌ Error: config.yaml not found at $config_path"
                SKIPPED_STEPS+=("step_$step_num (missing config.yaml)")
                continue
            fi
            
            if [ ! -d "$dcp_weights_path" ]; then
                echo "  ❌ Error: DCP weights directory not found at $dcp_weights_path"
                SKIPPED_STEPS+=("step_$step_num (missing weights)")
                continue
            fi
            
            # Check if output already exists
            if [ -d "$hf_output_path" ]; then
                echo "  ⚠️  Output directory already exists: $hf_output_path"
                read -p "  Overwrite? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo "  ⏭️  Skipping step_$step_num"
                    SKIPPED_STEPS+=("step_$step_num (already exists, user skipped)")
                    continue
                else
                    rm -rf "$hf_output_path"
                fi
            fi
            
            # Submit conversion job
            submit_conversion_job "$step_dir" "$step_num" "$config_path" "$dcp_weights_path" "$hf_output_path"
            
        else
            echo "Skipping $step_basename (step $step_num <= threshold $THRESHOLD)"
            SKIPPED_STEPS+=("$step_basename (below threshold)")
        fi
    fi
done

# Print summary
echo ""
echo "🚀 JOB SUBMISSION SUMMARY"
echo "========================"
echo "Total jobs submitted: ${#SUBMITTED_JOBS[@]}"
echo "Total steps skipped: ${#SKIPPED_STEPS[@]}"

if [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
    echo ""
    echo "📋 Submitted jobs:"
    for job_info in "${SUBMITTED_JOBS[@]}"; do
        job_id="${job_info%%:*}"
        step="${job_info##*:}"
        echo "  - Job $job_id: $step"
    done
    
    echo ""
    echo "💡 Monitor job progress with:"
    echo "   squeue -u \$USER"
    echo ""
    echo "📁 Logs will be saved to: $OUTPUT_BASE_DIR/logs/"
    echo ""
fi

if [ ${#SKIPPED_STEPS[@]} -gt 0 ]; then
    echo ""
    echo "⏭️  Skipped steps:"
    for skipped_step in "${SKIPPED_STEPS[@]}"; do
        echo "  - $skipped_step"
    done
fi

echo ""
echo "🎯 All jobs submitted! Use 'squeue -u \$USER' to monitor progress."