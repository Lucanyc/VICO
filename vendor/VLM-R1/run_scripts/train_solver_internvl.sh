#!/usr/bin/env bash
# =============================================================================
# VLM-R1 GRPO Training: VICO Solver on SFT'd InternVL3-8B
#
# Adapted from run_grpo_rec_internvl.sh (kept untouched), rewired for:
#   - SFT'd InternVL3-8B vico solver (LoRA merged)
#   - vico GenEnv + original mix data (jsonl from
#     build_solver_data_internvl.py)
#   - default accuracy + format rewards (grpo_jsonl.py defaults handle
#     numeric / text / MCQ for VICO)
#
# Usage:
#   conda activate vlm-r1
#   bash VLM-R1/run_scripts/train_solver_internvl.sh
# =============================================================================
set -xeuo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"

# ---- data ------------------------------------------------------------------
# jsonl produced by build_solver_data_internvl.py (absolute paths)
data_paths="${TRAIN_FILE:?Set TRAIN_FILE}"
# images are absolute in the jsonl, so use empty image_folder.
image_folders=""

# ---- model ------------------------------------------------------------------
# Direct absolute path. (VLM-R1 InternVL module's "InternVL" string check has
# been patched to case-insensitive "internvl", see vlm_modules/internvl_module.py)
model_path="${MODEL_PATH:?Set MODEL_PATH}"

# ---- experiment -------------------------------------------------------------
export EXP_NAME="InternVL3-8B-vico-solver-grpo"
TASK_TYPE="vico"           # falls through to default in InternVL module's
                              # get_question_template (uses generic template)
is_reward_customized_from_vlm_module=False  # use grpo_jsonl.py default rewards

cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true"
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

# ---- launch ----------------------------------------------------------------
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path "${model_path}" \
    --data_file_paths "${data_paths}" \
    --image_folders "${image_folders}" \
    --is_reward_customized_from_vlm_module ${is_reward_customized_from_vlm_module} \
    --task_type ${TASK_TYPE} \
    --max_anyres_num 12 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 5 \
    --save_total_limit 3 \
    --num_generations 8 \
    --max_completion_length 8192 \
    --max_prompt_length 1024 \
    --reward_funcs accuracy format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name vico-solver \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json

echo "Training completed for ${EXP_NAME}"