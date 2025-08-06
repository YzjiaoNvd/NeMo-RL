#!/bin/bash
set -e

# Base directory (from argument or default)
BASE_DIR="${1:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/NeMo-RL/results/2grpo_hs3_16K_step240_clip_max_0.28_llama3.1_8B_lr_2e-6_temp_1_kl_0.001_grpo_bs_256_rollout_16_num_prompts_128}"
HF_DIR="${BASE_DIR}/HF"
RESULTS_DIR="${BASE_DIR}/outputs"
DATASET="${2:-rmbench}"


# Find evaluation script
EVAL_SCRIPT="$(dirname "$0")/evaluate_genrm_one_step_w_fact.sh"
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: evaluate_genrm_one_step.sh not found"
    exit 1
fi

echo "GenRM Evaluation for: $BASE_DIR"
echo "Results: $RESULTS_DIR"
echo "----------------------------------------"

# Find step directories
STEP_DIRS=$(find ${HF_DIR} -type d -name "step_*" 2>/dev/null || find ${BASE_DIR} -type d -name "step_*")

if [ -z "$STEP_DIRS" ]; then
    echo "No checkpoint directories found!"
    exit 1
fi

# Process each step
for step_dir in $(echo "$STEP_DIRS" | sort -V); do
    step_num=$(basename "$step_dir" | grep -Eo '[0-9]+$')
    
    if [ -z "$step_num" ]; then
        echo "⚠️  Invalid: $(basename "$step_dir")"
        continue
    fi
    
    echo "Processing step_$step_num..."
    bash "$EVAL_SCRIPT" "$step_dir" "$step_num" "$RESULTS_DIR" ""$DATASET""

done
