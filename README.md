# VICO

[Paper](https://arxiv.org/abs/2610.10782)

VICO couples a visual editing Planner, an executable editor, and a Solver.
Solver feedback guides new edits; the resulting data trains the Solver and
Rewriter. This repository contains one source-derived workflow, with neutral
entry-point names and the original reward and training rules.

The included editor works with CSV-backed charts. It can modify table values
and rendering, or change rendering while preserving values. Supporting a
different visual environment requires a corresponding editor and answer rules.

## Core code

| Stage | Files |
| --- | --- |
| Generate editing trajectories | `run_evolution.py` |
| Edit tables and render images | `table_editor.py` |
| API and local Planner | `api_planner.py`, `local_planner.py` |
| Rewriter RWR data and training | `build_rewriter_data.py`, `train_rewriter.sh` |
| Solver SFT data generation | `generate_solver_sft.py` |
| Solver SFT data and training | `build_solver_sft_data.py`, `train_solver_sft.sh` |
| Solver GRPO data and training | `build_solver_data.py`, `train_solver.sh` |
| Original scoring and reward manager | `reward.py` |
| Evaluation | `evaluate.py` |

There are **13 core scripts**. Each stage has one implementation; historical
loop variants, duplicate evaluation scripts, optional training branches, and
copied training frameworks are omitted.

## Environment and execution

Use the original compatible Linux/CUDA environment with vLLM and verl for
inference/GRPO, and LLaMA-Factory for SFT. These engines are external dependencies.
`environment-reference.json` records observed versions, not a complete lockfile.
CPU data/rendering dependencies are in `requirements-data.txt`.

Set paths explicitly through CLI flags or launcher environment variables.
`.env.example` lists API/model configuration and must be loaded explicitly.
Credentials, models, datasets, generated images, and logs are supplied externally.

The GRPO launcher uses the active installed verl. Its existing custom-module
loader imports this repository's `reward.py` and registers `vico_reward`; no
external reward implementation is substituted.

Start with the verified API-Planner route:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_evolution.py \
  --data_json /path/to/rewriter_seeds.json \
  --tables_dir /path/to/tables --png_dir /path/to/png \
  --start 0 --n 4 --model_path /path/to/solver \
  --planner_mode gpt5 --planner_model "$AMPLIFIER_MODEL" \
  --n_rollouts 8 --max_turns 5 \
  --output_dir outputs/images --output_json outputs/trajectories.json
```

[Run commands](docs/RUN_COMMANDS.md) cover data construction, both training
paths, and small-run overrides. [Data notes](docs/DATA.md) describe the separate
rewritten archive and the required seed/trajectory inputs. Original SFT
launchers write dataset registration and may clean unreadable images from their
input JSON; use independent working directories and copied inputs.

## Reward and validation

The Solver receives **+1** when its extracted answer is correct and all four
thinking/answer tag strings are present; otherwise it receives **-1**. Numeric
answers and the original near-zero rule. Rewriter RWR
uses `exp(-beta * (p_hat - 0.5)^2)` and stochastic sample repetition, favoring
edits near 50% Solver success.

The original reference run produced 17 valid edits from four seeds and completed
two Solver GRPO steps and two Rewriter SFT steps. Mean Solver rewards were
0.5972 and 0.7014. These are bounded runtime checks, not benchmark or improvement
claims. Details and known local-Planner limits are in
[validation notes](docs/VALIDATION.md).

`SOURCE_PROVENANCE.json` records the source and publication hashes of the 13
selected implementations. Public changes are naming, path configuration, and
launcher wiring; the scoring and data-selection algorithms are retained.
