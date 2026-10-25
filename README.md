# VICO

[Paper](https://arxiv.org/abs/2610.10782)

This release organizes the original table-environment research code for visual
co-evolution. It includes the Solver and Rewriter training paths, their data
builders, evaluation scripts, and the existing InternVL and tool-training
branches. Dataset-specific file names have been generalized.

The research implementations are retained. There is no replacement feedback
loop, reward policy, or data-selection algorithm. Publication changes are listed
in [docs/SOURCE.md](docs/SOURCE.md); file hashes are in `SOURCE_PROVENANCE.json`.

## Code map

| Stage | Source-derived entry point |
| --- | --- |
| Generate editing trajectories | `run_evolution.py` |
| Table edits and rendering | `table_editor.py` |
| Trained / API Rewriter | `local_planner.py`, `api_planner.py` |
| Rewriter reward-weighted data | `build_rewriter_data.py` |
| Rewriter LoRA SFT | `train_rewriter.sh` |
| Solver SFT generation | `generate_solver_sft_api.py`, `generate_solver_sft_local.py` |
| Solver SFT data and training | `build_solver_sft_data.py`, `train_solver_sft.sh` |
| Solver GRPO data and training | `build_solver_data.py`, `train_solver.sh` |
| Original Solver reward | `reward.py` |
| InternVL JSONL data | `build_solver_data_internvl.py` |
| InternVL GRPO | `vendor/VLM-R1/run_scripts/train_solver_internvl.sh`, `train_solver_internvl_14b.sh` |
| Tool training | `build_solver_tool_data.py`, `train_solver_tool.sh`, `tool_router.py` |
| Evaluation | `evaluate.py`, `evaluate_answer_tag.py`, `evaluate_rethink.py`, benchmark-specific scripts |

The continuous, calibrated, and typed loop variants retain their original
implementations. The shared local Rewriter implementation is included because
the table loop imports it.

## Environment

Use the original compatible environments: Linux/CUDA with Python 3.10+, vLLM/verl
for Qwen training, a separate LLaMA-Factory environment for SFT, and VLM-R1 for
InternVL. `environment-reference.json` records an observed environment, not a
dependency lockfile. CPU rendering dependencies are in `requirements-data.txt`.
The Qwen GRPO launcher places `vendor/verl` on `PYTHONPATH`.

Set model, data, and output paths explicitly. Machine-specific examples in source
files use `/path/to` placeholders. API configuration uses `OPENAI_API_KEY`,
`OPENAI_BASE_URL`, and `AMPLIFIER_MODEL`; supply credentials outside the repository.

The original SFT launchers register datasets inside `LLAMA_FACTORY_DIR` and may
rewrite their input JSON when removing unreadable images. Run them with an
independent LLaMA-Factory working directory and copied inputs when preserving
existing files. Their behavior has not been silently changed.

## Generation and Rewriter training

Use two allocated GPUs for the original loop's separate Rewriter and Solver.
Each worker receives its own `--start`, `--n`, output directory, and JSON file.
The original JSON schema and `tier` / `answer_cell` metadata are required.

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_evolution.py \
  --data_json /path/to/rewriter_seeds.json \
  --tables_dir /path/to/tables --png_dir /path/to/png \
  --start 0 --n 200 \
  --model_path /path/to/previous_solver \
  --rewriter_model_path /path/to/previous_rewriter \
  --planner_mode local --n_rollouts 8 --max_turns 5 \
  --output_dir outputs/round3/images_0-200 \
  --output_json outputs/round3/trajectories_0-200.json --resume

python build_rewriter_data.py \
  --traj_files 'outputs/round3/trajectories_*.json' \
  --png_dir /path/to/png --output_dir outputs/round3/rewriter_data \
  --beta 4.0 --max_repeats 5

ROUND=3 REWRITER_BASE=/path/to/previous_rewriter \
TRAIN_DATA="$PWD/outputs/round3/rewriter_data/rewriter_rwr_train_llama.json" \
VAL_DATA="$PWD/outputs/round3/rewriter_data/rewriter_rwr_val_llama.json" \
LLAMA_FACTORY_DIR=/path/to/independent/LLaMA-Factory \
OUTPUT_DIR="$PWD/outputs/round3/rewriter_adapter" GPUS=0,1,2,3 \
bash train_rewriter.sh
```

The table Rewriter builder takes the arguments shown above. Flags such as
`--round`, `--gpt5_traj_files`, `--prev_traj_files`, and `--k_max` belong to a
different environment's builder and are not added here.

## Solver SFT

```bash
python generate_solver_sft_api.py \
  --data_json /path/to/train.json --png_dir /path/to/png \
  --start 0 --n 500 --model your-provider-model \
  --output_dir outputs/solver_sft_raw --min_think_words 200 --resume

python build_solver_sft_data.py \
  --raw_files 'outputs/solver_sft_raw/solver_sft_raw_*.json' \
  --output_dir outputs/solver_sft --min_think_words 200

BASE_MODEL=/path/to/base_model SFT_DATA_DIR="$PWD/outputs/solver_sft" \
LLAMA_FACTORY_DIR=/path/to/independent/LLaMA-Factory \
OUTPUT_DIR="$PWD/outputs/solver_sft_adapter" GPUS=0,1,2,3,4,5,6,7 \
bash train_solver_sft.sh
```

## Solver GRPO and checkpoint merge

The original builder retains all valid edited turns regardless of historical
pass rate. It also supports mixing original records and already-calibrated seed
images. Those original images are excluded from the separately packaged release;
see [docs/DATA.md](docs/DATA.md) for using that release with the original builder.

```bash
python build_solver_data.py \
  --traj_files 'outputs/round1/trajectories_*.json' 'outputs/round2/trajectories_*.json' \
  --original_data /path/to/train.json --png_dir /path/to/png \
  --n_original 2000 --output outputs/round2/solver_grpo.parquet

ROUND=2 TRAIN_FILE="$PWD/outputs/round2/solver_grpo.parquet" \
MODEL_PATH=/path/to/previous_solver CKPTS_DIR="$PWD/outputs/round2/solver_ckpt" \
NUM_GPUS=8 TRAIN_BATCH_SIZE=32 TOTAL_EPOCHS=3 ROLLOUT_TEMPERATURE=0.7 \
bash train_solver.sh

PYTHONPATH="$PWD/vendor/verl" python -m verl.model_merger merge \
  --backend fsdp --local_dir outputs/round2/solver_ckpt/global_step_380/actor \
  --target_dir outputs/round2/solver_merged
```

The first round uses the base model; later rounds pass the preceding selected
checkpoint explicitly. The supplied fourth-round recipe uses
`MAX_PROMPT_LENGTH=1648 MAX_RESPONSE_LENGTH=5824 MAX_MODEL_LEN=8896`.
The bounded gradient check and its results are documented in
[docs/VALIDATION.md](docs/VALIDATION.md).

For InternVL, set `TRAIN_FILE` to the JSONL produced by
`build_solver_data_internvl.py`, set `MODEL_PATH`, and run the corresponding
VLM-R1 launcher in the original compatible environment. These scripts use their
own `accuracy` and `format` rewards; they do not use the Qwen binary reward.
Tool training additionally requires the original tool environments, weights,
services, and `TOOL_ROUTER_URL`. Its launcher performs node-local Ray cleanup;
use it only on a node allocated to that run.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_answer_tag.py \
  --model_path /path/to/solver --data_json /path/to/test.json \
  --image_root /path/to/images --tp 4 --output outputs/evaluation.json
```

Benchmark-specific evaluation scripts keep their original flags and answer
checkers. `mathvision_inference.py` uses `--model` and `--test_file`, whereas
`eval_mathvista.py` uses `--model_path` and `--data_json`. Model merging requires
the complete `.../actor` path on the same command line.

No datasets, model weights, credentials, or raw experiment logs are committed.
The rewritten archive contains 41,495 QA records and 41,481 images, with zero
original seed images. Third-party licenses remain with their source packages.
