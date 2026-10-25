# Commands for the bounded GPT-5 reference workflow

These commands use the compact public entry points. Supply existing models, seeds,
tables, and images externally. Use fresh output paths and the compatible original
environments; the repository is not a replacement for those GPU environments.
The full 41,495-record rewritten archive is separate from Git.

## 1. Generate and build data

Activate the original `verl-tool-env-qwen3` environment. Set `OPENAI_API_KEY`
privately, `OPENAI_BASE_URL` to the authorized provider, and `AMPLIFIER_MODEL`
to the model supported by that provider. The reference used
`openai/openai/gpt-5.4` for both Planner and editor API requests.

```bash
export VICO_DIR="$PWD"
export RUN_DIR=/path/to/new-run
export DATA_ROOT=/path/to/dataset/train
export SOLVER_MODEL=/path/to/previous_solver
export REWRITER_MODEL=/path/to/previous_rewriter
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_PLUGINS=""
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0,1 python3 run_evolution.py \
  --data_json "$DATA_ROOT/rewriter_seeds.json" \
  --tables_dir "$DATA_ROOT/tables" --png_dir "$DATA_ROOT/png" \
  --start 0 --n 4 --model_path "$SOLVER_MODEL" \
  --planner_mode gpt5 --planner_model "$AMPLIFIER_MODEL" \
  --n_rollouts 8 --max_turns 5 \
  --output_dir "$RUN_DIR/images" \
  --output_json "$RUN_DIR/trajectories_0_4.json" --resume

python3 build_solver_data.py \
  --traj_files "$RUN_DIR/trajectories_0_4.json" \
  --original_data "$DATA_ROOT/train_augmented.json" \
  --png_dir "$DATA_ROOT/png" --n_original 0 \
  --output "$RUN_DIR/fresh_solver_grpo.parquet"

python3 build_rewriter_data.py \
  --traj_files "$RUN_DIR/trajectories_0_4.json" \
  --png_dir "$DATA_ROOT/png" --output_dir "$RUN_DIR/rewriter_data" \
  --beta 4.0 --max_repeats 5
```

On a scheduler, retain its assigned GPU visibility instead of replacing it with
the example device indices. The source seeds require the original schema and
`tier` / `answer_cell` metadata; a rewritten QA archive alone is not a seed file.

## 2. Combine existing rewritten data and new edits

This is a metadata adapter around the original data-building functions, matching
the reference experiment. It does not implement a different filtering policy.
Only generated turns and the rewritten archive are added; seed images are not.

```bash
export REWRITTEN_DATASET=/path/to/rewritten-dataset
python3 - <<'PY'
import json, os, random
from pathlib import Path
from build_solver_data import extract_from_genenv_trajectories, samples_to_parquet

root = Path(os.environ['REWRITTEN_DATASET']).resolve()
run = Path(os.environ['RUN_DIR']).resolve()
samples = []
with (root / 'samples.jsonl').open() as handle:
    for line in handle:
        row = json.loads(line)
        image = (root / row['image']).resolve()
        assert image.is_file() and root in image.parents
        samples.append({
            'image_path': str(image), 'question': row['question'],
            'answer': str(row['answer']), 'sample_id': row['id'],
            'p_hat': row['provenance'][0].get('p_hat', -1), 'source': 'genenv',
        })
old_count = len(samples)
fresh, stats = extract_from_genenv_trajectories([str(run / 'trajectories_0_4.json')])
samples.extend(fresh)
random.seed(44)
random.shuffle(samples)
samples_to_parquet(samples, str(run / 'data/solver_grpo.parquet'))
print({'existing_rewrites': old_count, 'fresh': len(fresh), 'total': len(samples),
       'fresh_builder_stats': dict(stats)})
PY
```

The recorded run had 41,495 existing plus 17 fresh records. A new stochastic API
run need not produce the same 17 edits or rewards. Validation produced by this
writer overlaps the training pool, so the bounded GRPO check disables it.

## 3. Solver GRPO: two steps

Use the active compatible `verl` installation. `train_solver.sh` loads the
repository's `reward.py` through verl's custom-module loader and selects
`vico_reward`. This preserves the original scoring and manager implementation
under a neutral registry name. No framework copy is bundled in this repository.

```bash
export ROUND=4 TRAIN_FILE="$RUN_DIR/data/solver_grpo.parquet"
export CKPTS_DIR="$RUN_DIR/solver_ckpt" MODEL_PATH="$SOLVER_MODEL"
export NUM_GPUS=8 TRAIN_BATCH_SIZE=32 TOTAL_EPOCHS=3 ROLLOUT_TEMPERATURE=0.7
export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=2048 MAX_MODEL_LEN=4096

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_solver.sh \
  trainer.total_training_steps=2 trainer.save_freq=2 trainer.test_freq=-1 \
  trainer.val_before_train=False trainer.resume_mode=disable \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  data.filter_overlong_prompts_workers=12 data.image_patch_size=16 data.seed=44 \
  data.train_max_samples=-1 data.val_max_samples=16 \
  ray_kwargs.ray_init.num_cpus=56 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  data.dataloader_num_workers=0 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model]' \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" 'trainer.logger=[console]'
```

The reference allocated 64 CPUs and eight A100 80 GB GPUs for this stage. The
two-step cap overrides the three-epoch setting; removing it starts a longer run.

## 4. Rewriter RWR/SFT: two steps

Activate the original LLaMA-Factory environment. `LLAMA_FACTORY_DIR` must be an
independent working copy: the source launcher writes its dataset registration
and generated YAML there and can clean unreadable images from its input JSON.
The reference used the newly generated inputs in the independent run directory.

```bash
export LLAMA_FACTORY_DIR=/path/to/independent/LLaMA-Factory
export ROUND=1 REWRITER_BASE="$REWRITER_MODEL" GPUS=0
export TRAIN_DATA="$RUN_DIR/rewriter_data/rewriter_rwr_train_llama.json"
export VAL_DATA="$RUN_DIR/rewriter_data/rewriter_rwr_val_llama.json"
export OUTPUT_DIR="$RUN_DIR/rewriter_adapter"
export BATCH_SIZE=1 GRAD_ACCUM=1 SAVE_STEPS=1 EVAL_STEPS=1

bash train_rewriter.sh \
  max_steps=2 logging_steps=1 preprocessing_num_workers=2 \
  cutoff_len=2048 image_max_pixels=262144 flash_attn=sdpa report_to=none
```

Warm-up gives the first step zero learning rate. Inspect `checkpoint-2` to verify
the update; the original best-validation selection may restore `checkpoint-1`
to the final adapter root, as it did in the reference check. These commands do
not claim that the two-step Rewriter is ready to replace GPT-5.

## Solver SFT initialization

When supervised initialization is needed before GRPO, use the single API
generator and SFT builder below. Existing SFT records can go directly to the
builder or launcher in their respective original formats.

```bash
python3 generate_solver_sft.py \
  --data_json "$DATA_ROOT/train_augmented.json" --png_dir "$DATA_ROOT/png" \
  --start 0 --n 500 --model your-provider-model \
  --output_dir "$RUN_DIR/solver_sft_raw" --min_think_words 200 --resume

python3 build_solver_sft_data.py \
  --raw_files "$RUN_DIR/solver_sft_raw/solver_sft_raw_*.json" \
  --output_dir "$RUN_DIR/solver_sft" --min_think_words 200

BASE_MODEL=/path/to/base_model SFT_DATA_DIR="$RUN_DIR/solver_sft" \
LLAMA_FACTORY_DIR=/path/to/independent/LLaMA-Factory \
OUTPUT_DIR="$RUN_DIR/solver_sft_adapter" GPUS=0 \
bash train_solver_sft.sh max_steps=2
```

## Evaluation

The retained evaluation entry point reads JSON records with `id`, `image_path`,
`question`, and `answer`, and uses the original answer-tag extraction and relaxed
numeric matching. It is distinct from the format-and-answer training reward.

```bash
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
  --model_path /path/to/solver --data_json /path/to/test.json \
  --image_root /path/to/images --tp 1 --n 2 \
  --output "$RUN_DIR/evaluation.json"
```
