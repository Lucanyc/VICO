# VICO

**Visual Environments Co-Evolving for Vision-Language Model Reasoning**

[Paper](https://arxiv.org/abs/2610.10782)

VICO couples a visual editing Planner, an executable environment editor, and a
Solver. The Solver's pass rate guides subsequent edits; the resulting trajectories
support Rewriter reward-weighted training and Solver GRPO.

The main interfaces are independent of dataset names. Data fields, prompts,
answer metrics, model paths, and editor factories are configurable. This release
includes a **CSV-backed table/figure editor**. Other visual environments need an
editor that returns an image and its valid answer; a generic interface alone does
not implement scene-graph or protected-region editing.

## Layout

| Component | Code |
| --- | --- |
| Co-evolution entry point | `run_evolution.py` |
| Dataset-independent feedback loop | `vico/pipeline.py` |
| JSON/JSONL schema adapter | `vico/data.py` |
| Editor adapter and custom factories | `vico/backends.py` |
| CSV editing, label extraction, rendering | `table_editor.py` |
| Local Planner / API table Planner | `local_planner.py`, `api_planner.py` |
| Solver inference and configurable scoring | `vico/solver.py`, `vico/metrics.py` |
| Rewriter data and RWR training | `build_rewriter_data.py`, `train_rewriter.sh` |
| Solver data and GRPO training | `tools/prepare_rewritten_training.py`, `train_solver.sh` |
| Training reward and framework | `reward.py`, `vendor/verl/` |
| Evaluation | `evaluate.py` |

`table_loop*.py`, `table_editor_typed.py`, and `api_planner_calibrated.py` retain
earlier table-specific variants for research comparison. `build_seed_batch.py`
prepares table seeds. Scripts ending in `_legacy.py` preserve older data/evaluation
paths; use the main entry points above for new runs. In particular, the legacy
Solver builder mixes seed data and creates overlapping train/validation data.

## Setup

Use Python 3.10+. CPU data processing and rendering:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt
python -m unittest discover -s tests -v
python -m unittest discover -s tools -p 'test_*.py' -v
```

Inference requires Linux, CUDA, a compatible local VLM checkpoint, and vLLM.
The runtime checks use torch 2.8.0, vLLM 0.11.0, and transformers 4.57.3.
`environment-reference.json` records the observed training environment; it is a
reference, not a resolved lockfile. The vendored framework metadata has older
NumPy constraints than that environment, so do not blindly resolve its dependencies
over an existing working installation. The training launcher puts the vendored
framework on `PYTHONPATH`.

Rewriter training uses an external LLaMA-Factory checkout. Configure paths and
credentials locally using `.env.example`; scripts do not automatically load it.

## Dataset interface

Supply a JSON array or JSONL with one record per question:

```json
{"id":"example-1","image":"images/example.png","question":"What value is shown?","answer":"20"}
```

Image paths are relative to `--image-root`. The adapter also accepts `image_path`,
`imgname`, `query`, and `label`. For other schemas, pass `--field-map mapping.json`:

```json
{"id":"uid","image":"picture","question":"prompt","answer":"target"}
```

For the included editor, also provide `table` (CSV path relative to the dataset
file) and `tier`. Tier 2 changes rendering while keeping values and labels fixed.
Tier 1 edits data and requires `answer_cell`, e.g. `{"row":1,"col":"Value"}`;
its answer is recomputed from that cell's entity. Generated CSVs are saved beside
images and used in the next edit, so the image, table, and answer share one state.
The table backend assumes questions refer to values/entities, rather than visual
features that style changes could invalidate.

Datasets and model weights are supplied separately. The prepared rewritten-only
archive contains 41,495 unique QA records and 41,481 unique images; 36,088 records
belong to the main co-evolution subset. The archive contains no original seed
images. It is not stored in Git. See `docs/DATA.md` for the schema and split policy.

## Run a small evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model /path/to/model \
  --data /path/to/samples.jsonl --image-root /path/to/dataset \
  --prompt-style boxed --metric relaxed --tolerance 0.05 \
  --n 2 --batch-size 1 --max-tokens 256 --max-model-len 4096 \
  --temperature 0 --tp 1 --enforce-eager \
  --output outputs/smoke/evaluation.json
```

Prompt styles: `boxed`, `answer_tag`, or `direct`. Use `--metric exact` for
case-insensitive exact text matching, or `relaxed` for a configurable relative
numeric tolerance. Neither is a universal answer verifier. Custom instructions
can be supplied with `--system-prompt-file`; keep their output format consistent
with `--prompt-style`. `--resume` checks configuration before reusing results.

## Generate and train

```bash
CUDA_VISIBLE_DEVICES=0 python run_evolution.py \
  --data /path/to/seeds.jsonl --image-root /path/to/seeds \
  --model /path/to/solver --editor table --planner simple \
  --n 2 --n-rollouts 4 --max-turns 2 \
  --output-dir outputs/round0

python build_rewriter_data.py \
  --traj_files outputs/round0/trajectories.json \
  --output_dir outputs/rewriter-data

REWRITER_BASE=/path/to/rewriter LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory \
TRAIN_DATA="$PWD/outputs/rewriter-data/rewriter_rwr_train_llama.json" \
VAL_DATA="$PWD/outputs/rewriter-data/rewriter_rwr_val_llama.json" \
bash train_rewriter.sh

python tools/prepare_rewritten_training.py \
  --dataset-root /path/to/rewritten-dataset --subset coevolution \
  --output-dir outputs/solver-data

MODEL_PATH=/path/to/solver NUM_GPUS=4 \
TRAIN_FILE="$PWD/outputs/solver-data/train.parquet" \
VAL_FILE="$PWD/outputs/solver-data/val.parquet" \
bash train_solver.sh
```

The RWR builder needs at least two source-image groups to create disjoint splits.
A two-seed smoke run may produce too few usable groups; use a larger generation
run to prepare training data. New trajectories preserve full questions and the
exact image/answer presented to each Planner turn.

The simple Planner uses feedback rules. `--planner api` uses the included table
Planner and `OPENAI_API_KEY`; data-changing table edits also require this key.
The local Planner accepts `--planner-config '{"model_path":"/path/to/rewriter"}'`.
Local Planner and Solver share the visible device pool: set the Planner's
`gpu_memory_utilization` in `--planner-config` and the Solver's
`--gpu-memory-utilization` so their combined budgets fit the hardware. For example,
two 8B models on one 80 GB GPU can start with budgets of 0.4 and 0.5 respectively;
actual capacity depends on model, context, and image sizes.

For another environment, pass `--editor your_module:factory` and, if needed,
`--planner your_module:factory`; JSON keyword arguments go in `--editor-config`
and `--planner-config`. See `docs/EXTENDING.md`. Training accepts `REWARD_PATH`,
`REWARD_FUNCTION`, and `REWARD_MANAGER` to select another verifier. Extra arguments
to `train_solver.sh` are forwarded as Hydra overrides.

## Validation and provenance

`docs/VALIDATION.md` records the actual smoke-test scope. CPU tests exercise real
rendering, schema handling, cumulative edits, RWR conversion, and grouped splits.
`python tools/smoke_training.py` verifies vendored verl reward loading and tensor
placement in an environment with training dependencies.

This is a refactored research-code release, not a frozen reproduction of every
paper experiment. `docs/CHANGES.md` identifies publication changes;
`SOURCE_PROVENANCE.json` records source hashes without private workspace paths.
Third-party notices and licenses are retained in `vendor/verl/`. No new license
is assigned to project-specific code or separately distributed derivative data.
