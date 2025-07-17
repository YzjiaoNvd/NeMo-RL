#!/bin/bash

set -x

GPFS="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL"
CONTAINER="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/container/nemo-rl:main-3e5481f.squashfs"
export HF_HOME=/lustre/fsw/portfolios/llmservice/users/yizhuj/hf_cache


# Number of nodes for the job
NUM_ACTOR_NODES=8

# Model and training configuration
FSDP2=True
#MODEL="Qwen/Qwen2.5-7B-Instruct"
#MODEL_NAME="qwen25_7b"
#MODEL="meta-llama/Llama-3.1-8B-Instruct"
#MODEL_NAME="llama3.1_8B"
MODEL="Qwen/Qwen3-8B"
MODEL_NAME="qwen3_8b"


ACT_CKPT=True
CPU_OFFLOAD=True
TP=1
project_name="yizhu_rlhf"
lr=2e-6
temp=1
grpo_bs=256
prompts_per_step=128
rollouts_per_prompt=$((8 * NUM_ACTOR_NODES))
kl=0.001
reward="r0"
data_version="_base"

NAME="grpo_hs3_16K_step240_clip_max_0.28_${MODEL_NAME}_lr_${lr}_temp_${temp}_kl_${kl}_grpo_bs_${grpo_bs}_rollout_${rollouts_per_prompt}_num_prompts_${prompts_per_step}_${reward}"

RESULTS_DIR="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/results/${NAME}${data_version}"
mkdir -p $RESULTS_DIR

ACTOR_LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p $ACTOR_LOG_DIR
PPO_ERRFILE="${ACTOR_LOG_DIR}/%j_%t.err"
PPO_OUTFILE="${ACTOR_LOG_DIR}/%j_%t.log"



# Construct the command to run
COMMAND="cd ${GPFS} && ulimit -c 0 && uv run examples/run_grpo_genrm.py \
    ++logger.wandb.name=${NAME} \
    ++logger.wandb_enabled=True \
    logger.wandb.project=${project_name} \
    ++checkpointing.checkpoint_dir=${RESULTS_DIR} \
    ++cluster.num_nodes=${NUM_ACTOR_NODES} \
    policy.dtensor_cfg.enabled=${FSDP2} \
    policy.dtensor_cfg.tensor_parallel_size=1 \
    ++policy.dtensor_cfg.context_parallel_size=1 \
    policy.dtensor_cfg.activation_checkpointing=${ACT_CKPT} \
    policy.dtensor_cfg.cpu_offload=${CPU_OFFLOAD} \
    ++cluster.gpus_per_node=8 \
    grpo.num_prompts_per_step=${prompts_per_step} \
    grpo.num_generations_per_prompt=${rollouts_per_prompt} \
    grpo.val_period=5 \
    grpo.max_val_samples=16 \
    grpo.val_batch_size=16 \
    data.train_data_path="/lustre/fsw/portfolios/llmservice/users/yizhuj/datasets/hs3_genrm/train_data${data_version}.jsonl" \
    data.val_data_path="/lustre/fsw/portfolios/llmservice/users/yizhuj/datasets/hs3_genrm/val_data${data_version}.jsonl" \
    ++data.shuffle_seed_for_training=-1 \
    loss_fn.reference_policy_kl_penalty=${kl} \
    loss_fn.use_on_policy_kl_approximation=False \
    loss_fn.use_importance_sampling_correction=False \
    checkpointing.keep_top_k=50 \
    checkpointing.save_period=5 \
    loss_fn.ratio_clip_min=0.2 \
    loss_fn.ratio_clip_max=0.28 \
    policy.model_name=${MODEL} \
    policy.make_sequence_length_divisible_by=8 \
    policy.generation.vllm_cfg.tensor_parallel_size=${TP} \
    policy.train_global_batch_size=${grpo_bs} \
    policy.train_micro_batch_size=1 \
    policy.generation_batch_size=2 \
    policy.logprob_batch_size=2 \
    policy.max_total_sequence_length=8192 \
    policy.optimizer.kwargs.lr=${lr} \
    policy.optimizer.kwargs.weight_decay=0 \
    policy.generation.temperature=${temp} \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.8 \
    data.dataset_name="hs3" \
    ++env.genrm.reward_design=${reward} \
"

# Set up mounts
MOUNTS="${GPFS}:${GPFS},/lustre:/lustre"

# Submit job using ray.sub
COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER}" \
MOUNTS="${MOUNTS}" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=llmservice_modelalignment_ppo \
    --job-name=grpo_genrm_hs3_${MODEL_NAME}_${reward}${data_version} \
    --partition=batch \
    --time=4:00:00 \
    --gres=gpu:8 \
    --mem=0 \
    --dependency=singleton \
    -o $PPO_OUTFILE \
    -e $PPO_ERRFILE \
    ray.sub
#!/bin/bash

set -x

GPFS="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL"
CONTAINER="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/container/nemo-rl:main-3e5481f.squashfs"
export HF_HOME=/lustre/fsw/portfolios/llmservice/users/yizhuj/hf_cache



# Number of nodes for the job
NUM_ACTOR_NODES=8

# Model and training configuration
FSDP2=True
#MODEL="Qwen/Qwen2.5-7B-Instruct"
#MODEL_NAME="qwen25_7b"
#MODEL="meta-llama/Llama-3.1-8B-Instruct"
#MODEL_NAME="llama3.1_8B"
MODEL="Qwen/Qwen3-8B"
MODEL_NAME="qwen3_8b"


ACT_CKPT=True
CPU_OFFLOAD=True
TP=1
project_name="yizhu_rlhf"
lr=2e-6
temp=1
grpo_bs=256
prompts_per_step=128
rollouts_per_prompt=$((8 * NUM_ACTOR_NODES))
kl=0.001
reward="r0"
data_version="_base"

NAME="grpo_hs3_16K_step240_clip_max_0.28_${MODEL_NAME}_lr_${lr}_temp_${temp}_kl_${kl}_grpo_bs_${grpo_bs}_rollout_${rollouts_per_prompt}_num_prompts_${prompts_per_step}_${reward}"

RESULTS_DIR="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/results/${NAME}${data_version}"
mkdir -p $RESULTS_DIR

ACTOR_LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p $ACTOR_LOG_DIR
PPO_ERRFILE="${ACTOR_LOG_DIR}/%j_%t.err"
PPO_OUTFILE="${ACTOR_LOG_DIR}/%j_%t.log"



# Construct the command to run
COMMAND="cd ${GPFS} && ulimit -c 0 && uv run examples/run_grpo_genrm.py \
    ++logger.wandb.name=${NAME} \
    ++logger.wandb_enabled=True \
    logger.wandb.project=${project_name} \
    ++checkpointing.checkpoint_dir=${RESULTS_DIR} \
    ++cluster.num_nodes=${NUM_ACTOR_NODES} \
    policy.dtensor_cfg.enabled=${FSDP2} \
    policy.dtensor_cfg.tensor_parallel_size=1 \
    ++policy.dtensor_cfg.context_parallel_size=1 \
    policy.dtensor_cfg.activation_checkpointing=${ACT_CKPT} \
    policy.dtensor_cfg.cpu_offload=${CPU_OFFLOAD} \
    ++cluster.gpus_per_node=8 \
    grpo.num_prompts_per_step=${prompts_per_step} \
    grpo.num_generations_per_prompt=${rollouts_per_prompt} \
    grpo.val_period=5 \
    grpo.max_val_samples=16 \
    grpo.val_batch_size=16 \
    data.train_data_path="/lustre/fsw/portfolios/llmservice/users/yizhuj/datasets/hs3_genrm/train_data${data_version}.jsonl" \
    data.val_data_path="/lustre/fsw/portfolios/llmservice/users/yizhuj/datasets/hs3_genrm/val_data${data_version}.jsonl" \
    ++data.shuffle_seed_for_training=-1 \
    loss_fn.reference_policy_kl_penalty=${kl} \
    loss_fn.use_on_policy_kl_approximation=False \
    loss_fn.use_importance_sampling_correction=False \
    checkpointing.keep_top_k=50 \
    checkpointing.save_period=5 \
    loss_fn.ratio_clip_min=0.2 \
    loss_fn.ratio_clip_max=0.28 \
    policy.model_name=${MODEL} \
    policy.make_sequence_length_divisible_by=8 \
    policy.generation.vllm_cfg.tensor_parallel_size=${TP} \
    policy.train_global_batch_size=${grpo_bs} \
    policy.train_micro_batch_size=1 \
    policy.generation_batch_size=2 \
    policy.logprob_batch_size=2 \
    policy.max_total_sequence_length=8192 \
    policy.optimizer.kwargs.lr=${lr} \
    policy.optimizer.kwargs.weight_decay=0 \
    policy.generation.temperature=${temp} \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.8 \
    data.dataset_name="hs3" \
    ++env.genrm.reward_design=${reward} \
"

# Set up mounts
MOUNTS="${GPFS}:${GPFS},/lustre:/lustre"

# Submit job using ray.sub
COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER}" \
MOUNTS="${MOUNTS}" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=llmservice_modelalignment_ppo \
    --job-name=grpo_genrm_hs3_${MODEL_NAME}_${reward}${data_version} \
    --partition=batch \
    --time=4:00:00 \
    --gres=gpu:8 \
    --mem=0 \
    --dependency=singleton \
    -o $PPO_OUTFILE \
    -e $PPO_ERRFILE \
    ray.sub