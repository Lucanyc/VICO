#!/usr/bin/env bash
# =============================================================================
# VLM-R1 GRPO Training: ChartQA Solver on SFT'd InternVL3-14B (LoRA merged).
#
# Variant of run_grpo_chartqa_internvl_vlm-r1.sh (8B). Changes vs 8B:
#   - model_path  -> chartqa_internvl3-14b_solver_lora_merged
#   - EXP_NAME    -> InternVL3-14B-chartqa-solver-grpo  (separate ckpts/runs)
#   - master_port -> 12350 (avoid collision with the 8B run)
#   - per_device_train_batch_size 4 -> 1, gradient_accumulation_steps 4 -> 8
#       14B at num_generations=8 + max_completion 8192 will OOM at batch 4 on
#       80GB GPUs. Per-GPU rollout: 1 prompt * 8 generations = 8 sequences.
#       Optimizer-step batch (8 GPUs): 1 * 8 grad_accum * 8 = 64 prompts/step
#       (8B was 128). Halving step batch is fine for GRPO.
#
# If still OOM: drop max_anyres_num 12 -> 6, OR max_completion_length 8192 ->
# 4096, OR switch deepspeed to zero3.
#
# Usage:
#   conda activate vlm-r1
#   bash VLM-R1/run_scripts/run_grpo_chartqa_internvl_vlm-r1_14b.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"

# ---- data ------------------------------------------------------------------
# jsonl produced by build_solver_grpo_data_chartqa_vlm-r1.py (absolute paths)
data_paths="${TRAIN_FILE:?Set TRAIN_FILE}"
# images are absolute in the jsonl, so use empty image_folder.
image_folders=""

# ---- model ------------------------------------------------------------------
# Direct absolute path. (VLM-R1 InternVL module's "InternVL" string check has
# been patched to case-insensitive "internvl", see vlm_modules/internvl_module.py)
model_path="${MODEL_PATH:?Set MODEL_PATH}"

# ---- experiment -------------------------------------------------------------
export EXP_NAME="InternVL3-14B-chartqa-solver-grpo"
TASK_TYPE="chartqa"           # falls through to default in InternVL module's
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
    --master_port="12350" \
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
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
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
    --dataset-name chartqa-solver \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json

echo "Training completed for ${EXP_NAME}"