# Runtime validation — 2026-10-25

These are bounded execution checks, not full epochs or benchmark reproduction.
All runs used independent outputs. Original source files, datasets, installed
framework files, and research checkpoints were not overwritten. Credentials,
raw logs, training records, images, and model weights are not included here.

## Main reference: GPT-5 generation and both training paths

The original `planner_executor_loop_chartqa_genenv.py` ran with
`--planner_mode gpt5 --planner_model openai/openai/gpt-5.4`, eight Solver rollouts
per image, five maximum editing turns, and `--resume`. Initialization was the
existing `chartqa_round3_solver_merged_step110` Solver. GPT-5 and the original
editor generated changes; the original rendering code produced the images.

Four tier-1 seeds produced 18 attempted edits and 17 valid image/QA samples.
There were zero Planner fallbacks. One trajectory reached the target band and
three ended as `partial`. One generated table declared six columns but supplied
five row values; original validation rejected that edit and continued.
This was a handled invalid edit, not a claim of 18 successful edits.

The original Solver builder retained all 17 valid edits, including examples with
historical success rates 0 and 1. The original Rewriter builder with beta 4 and
max repeats 5 produced 37 training and two validation rows. Generation plus
Rewriter training completed successfully in 7m04s. Re-running generation with
`--resume` found four completed trajectories, processed zero new samples, and
left the trajectory file unchanged.

## Solver GRPO: actual rewards and updates

The training pool combined all 41,495 packaged rewritten records with the 17
new edits: 41,512 records. The 2,048-token input filter retained 41,508. No
historical pass-rate filtering was applied. The builder's overlapping validation
sample was not used for validation.

`run_solver_grpo_chartqa.sh` used the original binary `chartqa_reward.py`, verified
byte-for-byte against the installed reward module. It used eight A100 80 GB GPUs,
batch size 32, nine rollouts per prompt, and temperature 0.7. Two optimizer steps
sampled 64 prompts and produced 576 answers, not a complete pass over the pool.

| Step | +1 rewards | -1 rewards | Mean reward | Gradient norm |
| --- | ---: | ---: | ---: | ---: |
| 1 | 230 | 58 | 0.5972222 | 0.2311042 |
| 2 | 245 | 43 | 0.7013889 | 0.3675249 |

The mean is `(positive_count - negative_count) / 288`, not an accuracy
percentage. Rewards require both answer correctness and format. These two
different batches do not establish a generalization improvement.

Advantages included positive and negative values in both steps. A checkpoint
spot check found 1,023 of 1,024 sampled attention-query values changed, with
maximum absolute difference 3.5651e-6; all were finite. The job completed with
exit code 0 in 15m13s and saved the step-2 Hugging Face checkpoint. Maximum
allocated GPU memory was 56.45 GB. One answer reached the 2,048-token output cap.

The only runtime launcher edit was forwarding `"$@"`. Overrides bounded the run
to two steps, disabled validation/resume, used 2048/2048 input/output limits and
4096 model context, enabled optimizer offload, and saved the model at step 2.
The relevant override recipe is in [RUN_COMMANDS.md](RUN_COMMANDS.md).
The active installed verl framework was used; it differs from the optional
vendored snapshot. See [SOURCE.md](SOURCE.md) and `environment-reference.json`.

## Rewriter RWR/SFT

`run_chartqa_rewriter_rwr.sh` initialized the original
`chartqa_rewriter_round0_merged`, then used the 37/2 rows built from the new GPT-5
trajectories. It retained original LoRA rank 64 and alpha 128. The builder used
beta 4 and max repeats 5. Test overrides used one GPU, batch size 1, accumulation
1, two steps, SDPA, a 2048-token cutoff, and 262144 maximum image pixels.

| Step | Learning rate | Training loss | Gradient norm | Validation loss |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 1.1699 | 1.7555708 | 1.3297648 |
| 2 | 0.000005 | 1.2070 | 1.5020014 | 1.3304036 |

Step 1 had zero learning rate because of the original warm-up. In checkpoint 2,
all 1,024 sampled LoRA-B values were nonzero, with maximum magnitude 5.0064e-6.
The original `load_best_model_at_end` selected checkpoint 1 because its validation
loss was lower, so the final adapter root is the first checkpoint, not the
updated second checkpoint. The validation set has only two examples; this check
establishes execution and a parameter update, not improved Rewriter quality.

## Earlier bounded checks

- Original Solver GRPO: four steps on the 41,495-record rewritten pool completed
  successfully; all 1,024 sampled checkpoint values changed.
- Original Solver SFT: two steps on 1,070 existing SFT rows completed with
  gradient norms 0.9838 / 1.6094 and training loss 0.6467. Those SFT records are
  separate from the rewritten-image archive and are not published here.
- Original Rewriter SFT: two steps on 1,668/88 rows from an existing trajectory
  shard completed with gradient norms 4.1675 / 4.0783.
- An existing trained local Rewriter generated two rendered edits in a limited
  tier-2 check. A separate checkpoint reload evaluated two examples, one matched;
  this is a reload check, not a benchmark score.

## Limits and publication checks

The current GPT-5 path succeeded with a working supplied credential. Earlier
authentication failures with other supplied configurations do not describe this
successful run; no credentials are included in the release.

The shared local Planner retains its scene-oriented prompt/tool list and original
untagged-JSON parser limitations. The main loop truncates stored questions to
100 characters. A newly trained local Rewriter replacing GPT-5 for a complete
co-evolution cycle remains unvalidated. Full multi-round training, InternVL GRPO,
tool-assisted training, and optimizer-state resume are also not established.

The runtime emitted tokenizer-regex and optional-kernel/process-cleanup warnings;
dependencies and tokenizer files were not changed. Original source behavior has
not been silently repaired to remove warnings or handled editing failures.

Publication validation checks syntax, preserved source hashes, local module
imports, and the original 23 embedded reward cases. The restored reward is
byte-identical to the GPU-tested source. Restoring script names does not
constitute a complete new GPU run of every optional branch.
