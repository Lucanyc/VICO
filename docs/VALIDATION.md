# Validation

The reference experiments used the original research implementations and
independent outputs. The current release retains only their main generation,
Solver, Rewriter, and evaluation paths with neutral names. Naming/path changes
do not imply a new full training experiment.

## API generation

The reference used `--planner_mode gpt5`, eight Solver rollouts per image, and
five maximum editing turns. The provider model was `openai/openai/gpt-5.4`.
Four tier-1 seeds produced 18 attempted edits and 17 valid image/QA samples,
with zero Planner fallbacks. One trajectory reached the target difficulty band;
three ended as partial.

One generated table declared six columns but supplied five values per row.
The original editor rejected it and continued. Re-running with `--resume`
processed zero new samples and left the saved trajectories unchanged.

## Solver training

The original builder retained all 17 valid edits, including historical success
rates 0 and 1. Combining them with the 41,495 packaged rewritten records gave
41,512 training rows; the 2048-token input filter retained 41,508.

The GRPO check used eight A100 80 GB GPUs, batch size 32, nine rollouts per prompt,
and temperature 0.7. Two optimizer steps sampled 64 prompts and generated 576
answers from the pool.

| Step | +1 rewards | -1 rewards | Mean reward | Gradient norm |
| --- | ---: | ---: | ---: | ---: |
| 1 | 230 | 58 | 0.5972222 | 0.2311042 |
| 2 | 245 | 43 | 0.7013889 | 0.3675249 |

The mean reward is `(positive_count - negative_count) / 288`, not an accuracy
percentage. Different training batches do not establish improved generalization.
The original binary reward required both format and answer correctness.

A checkpoint spot check found 1,023 of 1,024 sampled attention-query values
changed, with maximum absolute difference 3.5651e-6; all were finite. The job
completed successfully in 15m13s. One answer reached the 2048-token output limit.
The original builder's validation sample overlaps training, so GRPO validation
was disabled for this check.

A separate original Solver SFT check completed two steps on 1,070 existing
records, with gradient norms 0.9838 / 1.6094 and training loss 0.6467.

## Rewriter training

The original RWR builder used beta 4 and max repeats 5, producing 37 training
and two validation rows from the new trajectories. The SFT check initialized an
existing Rewriter and retained LoRA rank 64 and alpha 128.

| Step | Learning rate | Loss | Gradient norm | Validation loss |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 1.1699 | 1.7555708 | 1.3297648 |
| 2 | 0.000005 | 1.2070 | 1.5020014 | 1.3304036 |

Warm-up gave step 1 zero learning rate. In checkpoint 2, all 1,024 sampled LoRA-B
values were nonzero, with maximum magnitude 5.0064e-6. The original best-model
selection restored checkpoint 1 to the final adapter root because its validation
loss was lower. Inspect checkpoint 2 for the parameter update; this two-example
validation set does not establish improved Rewriter quality.

## Publication checks and limits

The selected implementations retain the original scoring, sample selection,
rendering, and optimization rules. Public names and labels were neutralized.
The renamed reward retains the original 23 embedded cases, and its custom reward
manager is loaded through the active installed verl rather than a copied engine.
Source hashes in `docs/SOURCE_PROVENANCE.json` cover only the selected files.

After organizing the scripts into `vico/`, `sft/`, and `rl/`, checks in an
isolated directory of the original server environment passed for six CLI entry
points, module imports, image rendering, and the installed verl reward loader.
Rebuilding the saved reference trajectories produced 17 Solver rows and 37/2
Rewriter training/validation rows. All three training launchers resolved paths
and forwarded overrides from a different working directory; their GPU training
commands were intercepted for this path check. No new optimizer steps were run
for the directory reorganization. All ten Python implementations retain the
same executable syntax trees as the preceding compact release.

Generation uses a separate numeric checker with 0.5% tolerance and scale
comparisons. It is intentionally not replaced by the training reward's 5% rule.
The loop still truncates stored questions to 100 characters.

The retained local Planner has the original scene-oriented tool list and
untagged-JSON parser limitations. A newly trained Rewriter replacing the API
Planner for a complete cycle remains unvalidated. Full multi-round training and
optimizer-state resume are not established by these bounded checks. Existing
runtime dependency warnings were not fixed by changing the original algorithms.
