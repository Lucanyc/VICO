# Runtime validation — 2026-10-25

All runs used independent output directories. Original code, datasets, installed
framework files, and research checkpoints were not overwritten. This document
reports bounded execution checks, not full epochs or benchmark reproduction.

## Original reward and full rewritten training pool

The original training environment loaded the original reward module, verified
byte-for-byte against the project source. The original `samples_to_parquet`
function converted all 41,495 packaged records without a historical pass-rate
cutoff. Its full training table and 2,074-row sampled validation table were kept
as produced. The training framework's 2,048-token input-length check retained
41,491 training rows; four were over the limit.

Training started from the existing third-round 8B Solver and used 8 A100 80 GB
GPUs, batch size 32, 9 rollouts per prompt, temperature 0.7, and the original
binary format-and-answer reward. Four steps sampled 128 prompts and generated
1,152 answers from the full training pool. No answer-only reward was substituted.

| Step | +1 rewards | -1 rewards | Gradient norm |
| --- | ---: | ---: | ---: |
| 1 | 208 | 80 | 0.8041199 |
| 2 | 259 | 29 | 0.4241900 |
| 3 | 220 | 68 | 1.0488333 |
| 4 | 272 | 16 | 0.2960535 |

Advantages had nonzero positive and negative values in every step. The job
completed with exit code 0 in 18m58s and saved the step-4 model and Hugging Face
checkpoint. One answer reached the 2,048-token output limit. No validation
accuracy claim is made from the source builder's overlapping validation sample.
An independent checkpoint spot check compared 1,024 values from the first
language attention query matrix against the initialization: all 1,024 changed,
with maximum absolute difference 7.7931e-6. This confirms an actual parameter
update, without establishing an accuracy improvement.

The research launcher was copied with one addition: forwarding `"$@"` to its
existing training command. Bounded-run overrides set four steps, save frequency,
disabled validation, input/output limits of 2048/2048, model context 4096,
rollout memory fraction 0.55, image patch size 16, optimizer offload, zero loader
subprocesses, and console logging. The reward and training algorithm were not
edited. The previous six-row check gave only +1 rewards and zero gradients;
that small sample was insufficient to test learning on this trained Solver.

## Original SFT paths

Both source launchers ran two optimizer steps with their original LoRA training
logic. The only source-copy change was forwarding CLI overrides. Test settings
used batch size 1, accumulation 1, SDPA, a 2,048-token cutoff, and 262,144 maximum
image pixels; LoRA rank 64 and alpha 128 retained source defaults.

| Path | Data | Observed result |
| --- | --- | --- |
| Rewriter RWR/SFT | Original builder, beta 4, max repeats 5; 1,668 train / 88 validation rows from one original trajectory shard | Gradient norms 4.1675 / 4.0783; training loss 1.7148; step-2 evaluation loss 1.9397 |
| Solver SFT | 1,070 original SFT records copied into the independent test directory | Gradient norms 0.9838 / 1.6094; training loss 0.6467 |

Both saved adapters and completed successfully; the combined job exited 0 in
3m27s. The first warm-up step had learning rate zero; the second used 5e-6.
Dataset registration and image validation operated on independent configuration
directories and copied/generated training JSON.

## Original generation and checkpoint inference

The original local-Planner loop loaded an existing trained Rewriter and the new
step-4 Solver on two GPUs. It processed two original tier-2 seeds, with eight
Solver answers per image and one editing turn. Both images were actually
re-rendered. Their pass rates changed from 1.0 to 0.0, leaving both trajectories
outside the target band (`partial`). The saved plans used the shared Planner's
`scene_rerender` tool. No new fallback/parser logic was inserted.

The original evaluation entry point then loaded the saved Solver checkpoint and
processed two packaged examples. One answer matched. This is a reload check,
not a benchmark score. The combined job completed with exit code 0 in 6m23s.

## API and untested branches

A real call through the original API Planner entered its fallback path. A
minimal request with the supplied NVIDIA configuration confirmed HTTP 401
(`AuthenticationError`). The other supplied provider configuration also produced
a fallback plan. Neither call completed a validated API table edit; no API
success is claimed and no credentials or provider response bodies are published.

The shared local Planner retains its original scene-oriented prompt/tool list,
and its untagged-JSON parser does not handle nested parameter objects. The main
loop also truncates stored questions to 100 characters. These source behaviors
remain visible rather than being silently repaired. Tier-1 API editing, InternVL
GRPO, tool-assisted training, optimizer-state resume, and full multi-round
training were not validated by these checks.

The runtime emitted tokenizer-regex and optional-kernel/process-cleanup warnings.
Existing dependency versions and tokenizer files were not changed. External
vLLM plugins were disabled for the Qwen checks.

## Publication checks

The renamed publication copy passed the original 23 embedded reward cases,
actual vendored verl reward registration and response-token placement checks,
and CLI import/help checks for generation and the data builders. Its evaluation
entry point loaded the saved Solver and processed the same two examples, matching
the original script's result. The publication validation job exited 0 in 2m34s.
All 472 included Python files passed Python 3.10 syntax parsing, and shell syntax
checks passed. The publication scan found no dataset-specific names, credentials,
or private source-workspace paths. GPU execution of the InternVL and tool paths
is not implied by these checks.
