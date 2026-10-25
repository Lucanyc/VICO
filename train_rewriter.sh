#!/usr/bin/env bash
# =============================================================================
# vico Rewriter RWR SFT Training (LlamaFactory LoRA)
#
# Supports ALL rounds:
#   Round 0: base Qwen3-VL-8B-Instruct → SFT on GPT-5 trajectories
#   Round 1+: previous round's merged rewriter → RWR-weighted SFT
#
# Usage:
#   # Round 0 (from base model, using GPT-5 trajectory data):
#   ROUND=0 \
#   GPUS=0,1,2,3 \
#   bash train_rewriter.sh
#
#   # Round 1 (from Round 0 merged rewriter):
#   ROUND=1 \
#   GPUS=0,1,2,3 \
#   bash train_rewriter.sh
#
#   # Custom paths:
#   ROUND=0 \
#   TRAIN_DATA=/path/to/rewriter_rwr_train_llama.json \
#   VAL_DATA=/path/to/rewriter_rwr_val_llama.json \
#   REWRITER_BASE=/path/to/base_model \
#   GPUS=0,1,2,3 \
#   bash train_rewriter.sh
# =============================================================================
set -euo pipefail

ROUND=${ROUND:-0}
VICO_DIR=${VICO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
LLAMA_FACTORY_DIR=${LLAMA_FACTORY_DIR:-$(python3 -c "import llamafactory; import os; print(os.path.dirname(os.path.dirname(llamafactory.__file__)))" 2>/dev/null || echo "/path/to/LlamaFactory")}
GPUS=${GPUS:-"0,1,2,3"}

# ── Data paths ──
TRAIN_DATA=${TRAIN_DATA:-${VICO_DIR}/vico_rewriter_rwr_data/rewriter_rwr_train_llama.json}
VAL_DATA=${VAL_DATA:-${VICO_DIR}/vico_rewriter_rwr_data/rewriter_rwr_val_llama.json}

# ── Model selection ──
if [ ${ROUND} -eq 0 ]; then
    # Round 0: train from base model
    REWRITER_BASE=${REWRITER_BASE:-/path/to/models/Qwen3-VL-8B-Instruct}
else
    # Round 1+: train from previous round's merged rewriter
    PREV_ROUND=$((ROUND - 1))
    REWRITER_BASE=${REWRITER_BASE:-${VICO_DIR}/vico_rewriter_round${PREV_ROUND}_merged}
fi

# ── Hyperparameters ──
LR=${LR:-5e-6}
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
SAVE_STEPS=${SAVE_STEPS:-50}
EVAL_STEPS=${EVAL_STEPS:-50}

# ── Output ──
OUTPUT_DIR=${OUTPUT_DIR:-${LLAMA_FACTORY_DIR}/saves/qwen3vl-8b/lora/vico_rewriter_rwr_round${ROUND}}

# ── Validation ──
if [ ! -d "${REWRITER_BASE}" ]; then
    echo "ERROR: Base model not found: ${REWRITER_BASE}"
    exit 1
fi
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: Training data not found: ${TRAIN_DATA}"
    exit 1
fi

# Count GPUs
NUM_GPUS=$(echo ${GPUS} | tr ',' '\n' | wc -l)

echo "============================================================"
echo "vico Rewriter RWR SFT - Round ${ROUND}"
echo "============================================================"
echo "  Base model:   ${REWRITER_BASE}"
echo "  Train data:   ${TRAIN_DATA}"
echo "  Val data:     ${VAL_DATA}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  GPUs:         ${GPUS} (${NUM_GPUS} GPUs)"
echo "  LR:           ${LR}"
echo "  Epochs:       ${EPOCHS}"
echo "  LoRA:         rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Batch:        ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS} = $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "============================================================"

# Print data stats
python3 -c "
import json
for name, path in [('Train', '${TRAIN_DATA}'), ('Val', '${VAL_DATA}')]:
    try:
        with open(path) as f:
            data = json.load(f)
        print(f'  {name}: {len(data)} samples')
    except Exception as e:
        print(f'  {name}: ERROR - {e}')
"

# ── Register datasets in dataset_info.json ──
DATASET_INFO=${LLAMA_FACTORY_DIR}/data/dataset_info.json

if [ -f "${DATASET_INFO}" ]; then
    python3 -c "
import json

info_path = '${DATASET_INFO}'
with open(info_path) as f:
    info = json.load(f)

for name, path in [
    ('vico_rewriter_rwr_round${ROUND}_train', '${TRAIN_DATA}'),
    ('vico_rewriter_rwr_round${ROUND}_val', '${VAL_DATA}'),
]:
    info[name] = {
        'file_name': path,
        'formatting': 'sharegpt',
        'columns': {'messages': 'conversations', 'images': 'images'},
        'tags': {
            'role_tag': 'from', 'content_tag': 'value',
            'user_tag': 'human', 'assistant_tag': 'gpt', 'system_tag': 'system'
        }
    }

with open(info_path, 'w') as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print('  Registered datasets in dataset_info.json')
"
else
    echo "  WARNING: ${DATASET_INFO} not found, creating minimal version"
    mkdir -p "$(dirname ${DATASET_INFO})"
    python3 -c "
import json
info = {}
for name, path in [
    ('vico_rewriter_rwr_round${ROUND}_train', '${TRAIN_DATA}'),
    ('vico_rewriter_rwr_round${ROUND}_val', '${VAL_DATA}'),
]:
    info[name] = {
        'file_name': path,
        'formatting': 'sharegpt',
        'columns': {'messages': 'conversations', 'images': 'images'},
        'tags': {
            'role_tag': 'from', 'content_tag': 'value',
            'user_tag': 'human', 'assistant_tag': 'gpt', 'system_tag': 'system'
        }
    }
with open('${DATASET_INFO}', 'w') as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print('  Created dataset_info.json')
"
fi

# ── Generate YAML config ──
YAML_PATH=${LLAMA_FACTORY_DIR}/examples/vico_rewriter_rwr_round${ROUND}.yaml
mkdir -p "$(dirname ${YAML_PATH})"
cat > ${YAML_PATH} << EOF
### Model
model_name_or_path: ${REWRITER_BASE}
template: qwen3_vl
### Method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: ${LORA_RANK}
lora_alpha: ${LORA_ALPHA}
### Dataset
dataset: vico_rewriter_rwr_round${ROUND}_train
eval_dataset: vico_rewriter_rwr_round${ROUND}_val
dataset_dir: data
preprocessing_num_workers: 16
### Output
output_dir: ${OUTPUT_DIR}
logging_steps: 5
save_steps: ${SAVE_STEPS}
overwrite_output_dir: true
### Train
per_device_train_batch_size: ${BATCH_SIZE}
gradient_accumulation_steps: ${GRAD_ACCUM}
learning_rate: ${LR}
num_train_epochs: ${EPOCHS}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
### Eval
per_device_eval_batch_size: ${BATCH_SIZE}
eval_strategy: steps
eval_steps: ${EVAL_STEPS}
save_total_limit: 3
load_best_model_at_end: true
metric_for_best_model: eval_loss
EOF

echo "  Generated YAML: ${YAML_PATH}"

# ── Clean bad images ──
echo "Cleaning bad images..."
python3 -c "
import json
from PIL import Image

for path in ['${TRAIN_DATA}', '${VAL_DATA}']:
    try:
        with open(path) as f:
            data = json.load(f)
    except:
        print(f'  SKIP (cannot load): {path}')
        continue
    clean = []
    bad = 0
    for s in data:
        ok = True
        for img_path in s.get('images', []):
            try:
                img = Image.open(img_path)
                img.verify()
            except:
                ok = False
                break
        if ok:
            clean.append(s)
        else:
            bad += 1
    if bad > 0:
        with open(path, 'w') as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        print(f'  {path}: {len(data)} -> {len(clean)} (removed {bad} bad images)')
    else:
        print(f'  {path}: all {len(data)} images OK')
"

# ── Train ──
echo ""
echo "Starting training..."
cd ${LLAMA_FACTORY_DIR}
CUDA_VISIBLE_DEVICES=${GPUS} llamafactory-cli train ${YAML_PATH} "$@"

# ── Done ──
echo ""
echo "============================================================"
echo "vico Rewriter RWR SFT Complete - Round ${ROUND}"
echo "  LoRA saved: ${OUTPUT_DIR}"
echo ""
echo "  To merge LoRA into full model:"
echo "    llamafactory-cli export \\"
echo "      --model_name_or_path ${REWRITER_BASE} \\"
echo "      --adapter_name_or_path ${OUTPUT_DIR} \\"
echo "      --template qwen3_vl \\"
echo "      --finetuning_type lora \\"
echo "      --export_dir ${VICO_DIR}/vico_rewriter_round${ROUND}_merged"
echo "============================================================"
