#!/bin/bash
#SBATCH -N 1 --gpus-per-node=1 --ntasks-per-node 1 -A llmservice_modelalignment_ppo -p batch --job-name eval_genrm_custom -t 04:00:00 --dependency=singleton 

export NCCL_ALGO=Tree
set -x

NAME="$1"


# Configuration
GPFS="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL"
CONTAINER="/lustre/fsw/portfolios/llmservice/users/yizhuj/nemorl/containers/anyscale+ray+2.43.0-py312-cu125_uv.sqsh"
export HF_HOME=/lustre/fsw/portfolios/llmservice/users/yizhuj/hf_cache


# Base directory for your specific checkpoint structure
DATASET="rmbench"
BASE_DIR="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/results/${NAME}"
HF_DIR="${BASE_DIR}/HF"  # Your checkpoints are under HF/step_X
RESULTS_DIR="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/outputs/genrm_eval_results/${NAME}"

# Create results directory
mkdir -p $RESULTS_DIR

# Job info
JOB_LOG_DIR=${RESULTS_DIR}
mkdir -p $JOB_LOG_DIR

MOUNTS="--container-mounts=${GPFS}:${GPFS},/lustre:/lustre"

# Create a summary file
SUMMARY_FILE="${JOB_LOG_DIR}/evaluation_summary.txt"
echo "GenRM Evaluation on ${DATASET} Dataset" > $SUMMARY_FILE
echo "======================================" >> $SUMMARY_FILE
echo "Base Directory: $BASE_DIR" >> $SUMMARY_FILE
echo "HF Directory: $HF_DIR" >> $SUMMARY_FILE
echo "Start Time: $(date)" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE


# Find all step directories under HF/ and sort them numerically
echo "Finding checkpoints in $HF_DIR..." | tee -a $SUMMARY_FILE
if [ -d "$HF_DIR" ]; then
    STEP_DIRS=$(find $HF_DIR -type d -name "step_*" | sort -V)
else
    echo "HF directory not found. Looking for step directories in base directory..." | tee -a $SUMMARY_FILE
    STEP_DIRS=$(find $BASE_DIR -type d -name "step_*" | sort -V)
fi

if [ -z "$STEP_DIRS" ]; then
    echo "No checkpoint directories found!" | tee -a $SUMMARY_FILE
    exit 1
fi

echo "Found checkpoints:" | tee -a $SUMMARY_FILE
echo "$STEP_DIRS" | tee -a $SUMMARY_FILE
echo "" | tee -a $SUMMARY_FILE

# Evaluate each checkpoint
for STEP_DIR in $STEP_DIRS; do
    STEP_NAME=$(basename $STEP_DIR)
    echo "----------------------------------------" | tee -a $SUMMARY_FILE
    echo "Evaluating checkpoint: $STEP_NAME" | tee -a $SUMMARY_FILE
    echo "Path: $STEP_DIR" | tee -a $SUMMARY_FILE
    
    # The model path is the step directory itself in your case
    MODEL_PATH="$STEP_DIR"
    
    # Define output file for this step
    OUTPUT_FILE="${JOB_LOG_DIR}/${STEP_NAME}_${DATASET}_results.json"
    LOG_FILE="${JOB_LOG_DIR}/${STEP_NAME}_eval.log"
    ERR_FILE="${JOB_LOG_DIR}/${STEP_NAME}_eval.err"
    
    # Run evaluation
    read -r -d '' cmd_eval <<EOF
cd ${GPFS} \
&& ulimit -c 0 \
&& uv run python examples/run_eval_genrm.py \
    --dataset rmbench \
    ++generation.model_name=${MODEL_PATH} \
    ++eval.output_file=${OUTPUT_FILE} \
    ++eval.batch_size=1024 \
    ++generation.vllm_cfg.tensor_parallel_size=1 \
    ++generation.vllm_cfg.gpu_memory_utilization=0.7 \
    ++cluster.gpus_per_node=1 \
    ++cluster.num_nodes=1 \
&& uv run python examples/run_eval_genrm.py \
    --dataset judgebench \
    ++generation.model_name=${MODEL_PATH} \
    ++eval.output_file=${OUTPUT_FILE} \
    ++eval.batch_size=1024 \
    ++generation.vllm_cfg.tensor_parallel_size=1 \
    ++generation.vllm_cfg.gpu_memory_utilization=0.7 \
    ++cluster.gpus_per_node=1 \
    ++cluster.num_nodes=1 \
&& uv run python examples/run_eval_genrm.py \
    --dataset hs3local \
    ++generation.model_name=${MODEL_PATH} \
    ++eval.output_file=${OUTPUT_FILE} \
    ++eval.batch_size=1024 \
    ++generation.vllm_cfg.tensor_parallel_size=1 \
    ++generation.vllm_cfg.gpu_memory_utilization=0.7 \
    ++cluster.gpus_per_node=1 \
    ++cluster.num_nodes=1

EOF

    echo "Running evaluation for $STEP_NAME..." | tee -a $SUMMARY_FILE
    echo "Output will be saved to: $OUTPUT_FILE" | tee -a $SUMMARY_FILE
    
    # Execute the evaluation
    srun -o $LOG_FILE -e $ERR_FILE --container-image=${CONTAINER}  -A llmservice_modelalignment_ppo -p batch --job-name eval_genrm_custom  -N 1 --gpus-per-node=1 --ntasks-per-node 1  $MOUNTS bash -c "${cmd_eval}"
    
done
