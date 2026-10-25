#!/usr/bin/env bash
# =============================================================================
# Solver GRPO Training for scene-Math
#
# Uses verl GRPO to train the solver on difficulty-calibrated scene images.
# Optimized for 3× A100-80GB setup.
#
# Usage:
#   bash rl/train_solver.sh
#
#   # Custom paths:
#   ROUND=0 \
#   TRAIN_FILE=/path/to/solver_grpo_train.parquet \
#   MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
#   bash rl/train_solver.sh
# =============================================================================
set -euo pipefail

# =============================================================================
# Networking — force torch.distributed to use localhost
# =============================================================================
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29500}
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true


# =============================================================================
# Configuration
# =============================================================================
ROUND=${ROUND:-0}
VICO_DIR=${VICO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}

# Data
TRAIN_FILE=${TRAIN_FILE:-${VICO_DIR}/round_${ROUND}/solver_grpo_train.parquet}

# Model: use previous round's solver or original
if [ ${ROUND} -gt 0 ]; then
    PREV_ROUND=$((ROUND - 1))
    DEFAULT_MODEL=$(ls -d ${VICO_DIR}/round_${PREV_ROUND}/solver_merged_step* 2>/dev/null | sort -t'p' -k2 -n | tail -1 || true)
    if [ -z "${DEFAULT_MODEL}" ]; then
        DEFAULT_MODEL="/path/to/models/Qwen3-VL-8B-Instruct"
    fi
else
    DEFAULT_MODEL="/path/to/models/Qwen3-VL-8B-Instruct"
fi
MODEL_PATH=${MODEL_PATH:-${DEFAULT_MODEL}}

# Checkpoints
CKPTS_DIR=${CKPTS_DIR:-${VICO_DIR}/round_${ROUND}/solver_ckpt}

# Training hyperparameters (auto-scaled to GPU count)
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
LR=${LR:-1e-6}
N_ROLLOUTS=${N_ROLLOUTS:-9}

# Hardware
NUM_GPUS=${NUM_GPUS:-3}
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:--1}

# Batch sizes: auto-scale to GPU count if not overridden
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$(( NUM_GPUS * 4 ))}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-${NUM_GPUS}}
PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-1}

ENGINE=vllm
project_name='vico_coevolve_solver'
exp_name="vico-solver-round${ROUND}-grpo"

mkdir -p "${CKPTS_DIR}"

echo "============================================================"
echo "scene Solver GRPO Training — Round ${ROUND}"
echo "  Train data:   ${TRAIN_FILE}"
echo "  Model:        ${MODEL_PATH}"
echo "  Checkpoints:  ${CKPTS_DIR}"
echo "  GPUs:         ${NUM_GPUS}"
echo "  Epochs:       ${TOTAL_EPOCHS}"
echo "  Batch size:   ${TRAIN_BATCH_SIZE}"
echo "  N rollouts:   ${N_ROLLOUTS}"
echo "  LR:           ${LR}"
echo "  Mini-batch:   ${PPO_MINI_BATCH}"
echo "  Micro-batch:  ${PPO_MICRO_BATCH}"
echo "============================================================"

# Verify data exists
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: Training data not found: ${TRAIN_FILE}"
    exit 1
fi

# Print data stats
python3 -c "
import pandas as pd
df = pd.read_parquet('${TRAIN_FILE}')
print(f'  Data: {len(df)} samples')
print(f'  Columns: {list(df.columns)}')
"

# =============================================================================
# Run GRPO Training
# =============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TRAIN_FILE/.parquet/_val.parquet}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH:-1048} \
    data.max_response_length=${MAX_RESPONSE_LENGTH:-3824} \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.image_key=images \
    data.shuffle=True \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF:-0} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE} \
    \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN:-6896} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS:-16392} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-0.7} \
    actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    \
    custom_reward_function.path="${VICO_DIR}/rl/reward.py" \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=vico_reward \
    \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.critic_warmup=0 \
    trainer.resume_mode=auto \
    trainer.val_before_train=True \
    "trainer.logger=[console,wandb]" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 "$@"

echo "============================================================"
echo "scene Solver GRPO Training Complete — Round ${ROUND}"
echo "  Checkpoint: ${CKPTS_DIR}"
echo "============================================================"
