# Runtime validation

Validation date: 2026-10-22. Tests ran in an isolated publication copy; the
original research source and model checkpoints were not modified.

## Passed

- Python syntax parsing for all 341 Python source files and Bash syntax checks
  for both training launchers.
- Eight core/backend tests plus one grouped-split test. These cover custom field
  mappings, duplicate IDs, image-root validation, answer extraction, configurable
  metrics, failed edits, full questions, and per-turn state propagation.
- Real Matplotlib rendering and label extraction, including persistence and reuse
  of an edited CSV. The external API response is substituted in the data-edit
  regression; the style-only rendering test needs no API.
- RWR conversion from generated trajectories, including the exact image and
  answer supplied to each Planner turn.
- The vendored verl reward loader registers `vico_reward`, applies a configured
  exact-match scorer, and places rewards on the last valid response token.
- `bash train_solver.sh --cfg job` successfully composed the full Hydra training
  configuration using real train/validation Parquet paths and the vendored
  framework, with exit code 0. This prints configuration and does not run an
  optimizer step. The login-node check used `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`; earlier runs with the inherited
  threading environment exceeded the check timeout during dependency startup.
- Actual single-GPU inference through `evaluate.py`: an existing local Qwen3-VL-8B
  Solver checkpoint on one NVIDIA A100-SXM4-80GB, tensor parallelism 1, BF16,
  eager execution, maximum context 4096, output limit 256, and temperature 0.
  Both selected rewritten samples produced saved generations. The scheduler
  reported completion with exit code 0; total job time was 3m34s, including
  environment startup/model loading, and the two-example inference loop took
  15.6s.

The GPU check used only two samples. One matched under the configured 5% numeric
tolerance; the other reached the short output limit without a final boxed answer.
This checks runtime behavior and is not a benchmark score. The installed runtime
also emitted tokenizer-regex warnings and a third-party plugin shutdown message;
model/tokenizer files were left unchanged.

## Scope

Full GRPO training, Rewriter SFT, external API editing, and complete benchmark
reproduction were not run. CPU feedback-loop tests substitute Solver pass rates;
real model inference was tested separately. Historical loop variants are retained
for inspection and are not covered by an end-to-end GPU run.

Raw test outputs, datasets, checkpoints, and private filesystem paths are not
included in Git. The commands and tests in this repository allow the checks to
be repeated with locally configured model and data paths.
