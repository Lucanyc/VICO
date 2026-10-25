#!/bin/bash
set -x

# =============================================================================
# Solver Tool-Call GRPO Training
#
# Uses verl-tool framework with multi-tool agent:
#   - chartmoe, easyocr, sam3_detect, calculator, pixel_reasoner
#   - Tool router at port 5556 (must be running before starting)
#
# Usage:
#   # First start tool servers (separate terminals):
#   bash start_tool_servers.sh
#
#   # Then start training:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_solver_tool.sh
# =============================================================================

# ═══════════════════════════════════════════════════════════════
#  Environment
# ═══════════════════════════════════════════════════════════════

export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29500}
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

# Clean up any stale Ray session on this node (previous job leftover) so that
# ray.init() starts a fresh local cluster instead of trying to connect to a
# dead GCS at an old head-node IP (e.g. 10.49.160.5:6379).
unset RAY_ADDRESS
ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray /tmp/ray_* 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════

VICO_DIR=${VICO_DIR:-/path/to/VICO}
VERL_TOOL_DIR="${VERL_TOOL_DIR:-${VICO_DIR}/vendor/verl-tool}"

# Model — SFT-trained with tool-call capability
model_name=${MODEL_PATH:-${VICO_DIR}/solver_tool_sft_merged}

# Data — tool-agent GRPO data with tool system prompt
train_data=${TRAIN_DATA:-"[${VICO_DIR}/vico_tool_agent_grpo.parquet]"}
val_data=${VAL_DATA:-"[${VICO_DIR}/vico_tool_agent_grpo_val.parquet]"}

# Checkpoints
# Checkpoint dir — lustre may have quota issues, keep on lustre but monitor
checkpoint_dir=${CKPTS_DIR:-${VICO_DIR}/solver_tool_grpo_ckpt}
mkdir -p "$checkpoint_dir"

# ═══════════════════════════════════════════════════════════════
#  Tool Router — must already be running on port 5556
# ═══════════════════════════════════════════════════════════════

TOOL_ROUTER_URL="${TOOL_ROUTER_URL:-http://localhost:5556/get_observation}"

echo "========================================="
echo "Checking Tool Router..."
echo "========================================="
ROUTER_HOST=$(echo "$TOOL_ROUTER_URL" | sed 's|http://||;s|/.*||')
if curl -s "http://${ROUTER_HOST}/health" 2>/dev/null | grep -q "healthy"; then
    echo "✅ Router is healthy (${ROUTER_HOST})"
    # Show tool status
    curl -s "http://${ROUTER_HOST}/check_tools" 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'  Tools: {data[\"summary\"]}')
for tool, info in data['tools'].items():
    icon = '✅' if info['status'] == 'online' else '❌'
    print(f'    {icon} {tool}: port {info[\"port\"]} ({info[\"status\"]})')
" 2>/dev/null || true
else
    echo "❌ Router not running at ${ROUTER_HOST}! Start it first: python tool_router.py"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
#  Training Configuration
# ═══════════════════════════════════════════════════════════════

# Algorithm
rl_alg=grpo
strategy="fsdp2"

# Resources
n_gpus_per_node=${NUM_GPUS:-4}
n_nodes=1
n=${N_ROLLOUTS:-8}                    # responses per prompt
batch_size=${TRAIN_BATCH_SIZE:-8}     # small batch to match tool server throughput
ppo_mini_batch_size=${PPO_MINI_BATCH:-${n_gpus_per_node}}

# Sequence lengths
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-10000}
max_action_length=4096                # max tokens for one action (think+tool or think+answer)
max_obs_length=4096                   # max tokens for tool observation
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)

# Sampling
temperature=${ROLLOUT_TEMPERATURE:-0.7}
top_p=1.0

# Agent config
enable_agent=True
max_turns=5                           # allow up to 5 tool calls: e.g. chartmoe → calculator → ... → answer
max_concurrent_trajectories=16            # match ChartMoE throughput: 8 batch × 8 rollout = 64 total, 16 concurrent
action_stop_tokens='</tool_call>'
additional_eos_token_ids=[151645]      # <|im_end|>
mask_observations=True                # ← critical: don't train on tool output
enable_mtrl=True                      # multi-turn RL
mtrl_role=user                        # LlamaFactory qwen3_vl uses "user" role for tool responses

# Optimizer
lr=${LR:-1e-6}
kl_loss_coef=0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl

# Reward
reward_manager=${REWARD_MANAGER:-vl_reasoning_qwen3}

# Memory
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
tensor_model_parallel_size=1
gpu_memory_utilization=${GPU_MEM_UTIL:-0.65}
do_offload=False
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1
max_num_batched_tokens=24000
rollout_mode='async'

# Training schedule
total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-50}

# Naming
run_name="solver-tool-grpo-$(date +%m%d)"

# ═══════════════════════════════════════════════════════════════
#  Action stop tokens file
# ═══════════════════════════════════════════════════════════════

action_stop_tokens_file="$(mktemp)"
echo -e -n "$action_stop_tokens" > $action_stop_tokens_file

# ═══════════════════════════════════════════════════════════════
#  Print config
# ═══════════════════════════════════════════════════════════════

echo ""
echo "========================================="
echo "Solver Tool-Call GRPO Training"
echo "========================================="
echo "  Model:        $model_name"
echo "  Train data:   $train_data"
echo "  Checkpoints:  $checkpoint_dir"
echo "  GPUs:         $n_gpus_per_node"
echo "  Tool router:  $TOOL_ROUTER_URL"
echo ""
echo "  Agent config:"
echo "    max_turns:           $max_turns"
echo "    mask_observations:   $mask_observations"
echo "    action_stop_tokens:  $action_stop_tokens"
echo "    enable_mtrl:         $enable_mtrl"
echo ""
echo "  Training:"
echo "    n (rollouts):        $n"
echo "    batch_size:          $batch_size"
echo "    lr:                  $lr"
echo "    epochs:              $total_epochs"
echo "    reward:              $reward_manager"
echo "========================================="

# ═══════════════════════════════════════════════════════════════
#  Launch Training
# ═══════════════════════════════════════════════════════════════

cd ${VERL_TOOL_DIR}
export PYTHONPATH="${VICO_DIR}/vendor/verl:${VERL_TOOL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=36 \
    data.dataloader_num_workers=4 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.image_key=images \
    data.shuffle=True \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$TOOL_ROUTER_URL \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.mtrl_role=$mtrl_role \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.max_concurrent_trajectories=$max_concurrent_trajectories \
    actor_rollout_ref.agent.tool_call_timeout=120 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name='solver_tool_grpo' \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.resume_mode=auto \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs

# Cleanup
rm -f $action_stop_tokens_file

echo "========================================="
echo "Training complete!"
echo "  Checkpoints: $checkpoint_dir"
echo "========================================="
