#!/usr/bin/env bash
# =============================================================================
# vico Solver SFT Training with Qwen3-VL-235B-Thinking generated data
#
# Same pipeline as scene solver SFT, adapted for vico:
#   Step 1: Build SFT data (already done via build_solver_sft_data.py)
#   Step 2: Register dataset in LlamaFactory
#   Step 3: Generate YAML config
#   Step 4: Validate images
#   Step 5: LlamaFactory LoRA SFT training
#
# Usage:
#   GPUS="0,1,2,3,4,5,6,7" bash sft/train_solver_sft.sh
#   GPUS="0,1,2,3,4,5,6,7" NUM_EPOCHS=10 bash sft/train_solver_sft.sh
# =============================================================================
set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
VICO_DIR=${VICO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
LLAMA_FACTORY_DIR=${LLAMA_FACTORY_DIR:-/path/to/LLaMA-Factory}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}

# Base model
BASE_MODEL=${BASE_MODEL:-/path/to/models/Qwen3-VL-8B-Instruct}

# SFT data (already built by build_solver_sft_data.py)
SFT_DATA_DIR=${SFT_DATA_DIR:-${VICO_DIR}/vico_solver_sft_235b}
TRAIN_DATA=${SFT_DATA_DIR}/solver_sft_train.json

# Training output
OUTPUT_DIR=${OUTPUT_DIR:-${LLAMA_FACTORY_DIR}/saves/qwen3vl-8b/lora/vico_solver_sft_qwen3vl235bthinking}

# Hyperparameters — aligned with scene solver SFT
NUM_EPOCHS=${NUM_EPOCHS:-10}
LR=${LR:-5.0e-6}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}

echo "============================================================"
echo "vico Solver SFT — Qwen3-VL-235B-Thinking Data"
echo "  Base model:      ${BASE_MODEL}"
echo "  Train data:      ${TRAIN_DATA}"
echo "  Output:          ${OUTPUT_DIR}"
echo "  GPUs:            ${GPUS}"
echo "  Epochs:          ${NUM_EPOCHS}"
echo "  LR:              ${LR}"
echo "  Batch (per GPU): ${BATCH_SIZE}"
echo "  Grad accum:      ${GRAD_ACCUM}"
echo "  Effective batch:  $((BATCH_SIZE * GRAD_ACCUM))"
echo "  LoRA rank/alpha: ${LORA_RANK}/${LORA_ALPHA}"
echo "============================================================"

# Verify
if [ ! -d "${BASE_MODEL}" ]; then
    echo "ERROR: Base model not found: ${BASE_MODEL}"
    exit 1
fi

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: Train data not found: ${TRAIN_DATA}"
    echo "  Run build_solver_sft_data.py first"
    exit 1
fi

TRAIN_COUNT=$(python3 -c "import json; print(len(json.load(open('${TRAIN_DATA}'))))")
echo "  Train samples: ${TRAIN_COUNT}"

# =============================================================================
# Step 1: Register dataset in LlamaFactory
# =============================================================================
echo ""
echo "============================================================"
echo "Step 1: Registering dataset in LlamaFactory"
echo "============================================================"

DATASET_INFO=${LLAMA_FACTORY_DIR}/data/dataset_info.json
python3 -c "
import json

info_path = '${DATASET_INFO}'
with open(info_path) as f:
    info = json.load(f)

info['vico_solver_sft_qwen3vl235bthinking_train'] = {
    'file_name': '${TRAIN_DATA}',
    'formatting': 'sharegpt',
    'columns': {'messages': 'conversations', 'images': 'images'},
    'tags': {
        'role_tag': 'from', 'content_tag': 'value',
        'user_tag': 'human', 'assistant_tag': 'gpt', 'system_tag': 'system'
    }
}

with open(info_path, 'w') as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print('  Registered dataset in dataset_info.json')
"

# =============================================================================
# Step 2: Generate YAML config
# =============================================================================
echo ""
echo "============================================================"
echo "Step 2: Generating training YAML"
echo "============================================================"

YAML_PATH=${LLAMA_FACTORY_DIR}/examples/vico_solver_sft_qwen3vl235bthinking.yaml
cat > ${YAML_PATH} << EOF
### Model — always from base
model_name_or_path: ${BASE_MODEL}
template: qwen3_vl

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: ${LORA_RANK}
lora_alpha: ${LORA_ALPHA}

### Dataset
dataset: vico_solver_sft_qwen3vl235bthinking_train
dataset_dir: data
preprocessing_num_workers: 16

### Output
output_dir: ${OUTPUT_DIR}
logging_steps: 5
save_steps: 50
overwrite_output_dir: true

### Train
per_device_train_batch_size: ${BATCH_SIZE}
gradient_accumulation_steps: ${GRAD_ACCUM}
learning_rate: ${LR}
num_train_epochs: ${NUM_EPOCHS}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
max_grad_norm: 0.5

### Eval — no val set, all data used for training
eval_strategy: "no"
save_total_limit: 3

### Logging
report_to: wandb
run_name: vico_solver_sft_qwen3vl235bthinking
EOF

echo "  Generated: ${YAML_PATH}"

# =============================================================================
# Step 3: Validate images
# =============================================================================
echo ""
echo "============================================================"
echo "Step 3: Validating images"
echo "============================================================"

python3 -c "
import json
from PIL import Image

path = '${TRAIN_DATA}'
with open(path) as f:
    data = json.load(f)
clean = []
bad = 0
for s in data:
    try:
        for img_path in s.get('images', []):
            img = Image.open(img_path)
            img.verify()
        clean.append(s)
    except:
        bad += 1
if bad > 0:
    with open(path, 'w') as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f'  {path}: {len(data)} -> {len(clean)} (removed {bad} bad)')
else:
    print(f'  {path}: all {len(data)} images OK')
"

# =============================================================================
# Step 4: Train
# =============================================================================
echo ""
echo "============================================================"
echo "Step 4: Training"
echo "============================================================"

cd ${LLAMA_FACTORY_DIR}
CUDA_VISIBLE_DEVICES=${GPUS} llamafactory-cli train ${YAML_PATH} "$@"

echo ""
echo "============================================================"
echo "vico Solver SFT Training Complete"
echo "  LoRA saved: ${OUTPUT_DIR}"
echo "  Next: merge LoRA with base model:"
echo "    llamafactory-cli export \\"
echo "      --model_name_or_path ${BASE_MODEL} \\"
echo "      --adapter_name_or_path ${OUTPUT_DIR}/checkpoint-XXX \\"
echo "      --template qwen3_vl \\"
echo "      --finetuning_type lora \\"
echo "      --export_dir ${VICO_DIR}/vico_solver_sft_merged"
echo "============================================================"
