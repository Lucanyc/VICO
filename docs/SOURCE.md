# Original sources and publication adjustments

This release uses the author's original research script names and imports.
The verified source snapshot was read on 2026-10-25. The generation loop, editor,
data builders, and reward come directly from that snapshot. Each retained source
file has its original and published SHA-256 in `SOURCE_PROVENANCE.json`.
Restored entry points also record their original relative `source_file`.

Publication adjustments are limited to the per-file list in that manifest:

- Machine-specific paths become `/path/to` placeholders. Shell project-directory
  defaults use the script directory and keep the original environment-variable
  names, including `CO_EVOLVE_DIR`.
- `run_solver_grpo_chartqa.sh`, `run_chartqa_rewriter_rwr.sh`, and
  `run_solver_sft_chartqa.sh` forward extra command-line arguments, matching the
  isolated launchers used for bounded checks. Shell tracing is disabled.
- The main GRPO launcher uses the active installed `verl`, matching the validated
  research environment. It no longer overrides it with the older vendored
  framework. Reward registration remains `chartqa_reward`.
- Optional tool and InternVL launchers retain external path configuration and
  the repository's `vendor/` layout. Their GPU training is not claimed as tested.
- Previously published supporting framework files remain source-derived copies;
  their earlier naming/path adjustments are recorded in the manifest.

The research behavior is unchanged:

- Solver reward: +1 only if all four `<think>`, `</think>`, `<answer>`, and
  `</answer>` strings are present and the extracted answer is correct; -1
  otherwise. This is a tag-presence check, not a strict XML validator.
  Numbers use 5% relative tolerance and the original near-zero rule. Text uses
  original normalization and the short-reference prefix rule.
- Generation feedback: its separate checker uses 0.5% numeric tolerance and
  scale comparisons. It has not been replaced by the training reward.
- Rewriter: `exp(-beta * (p_hat - 0.5)^2)`, beta 4 by default, followed by
  stochastic repetition with max repeats 5 and LoRA supervised training.
- Solver data: valid edited turns are retained regardless of historical
  difficulty. Image, answer, execution, and pass-rate metadata checks remain.
  The original builder also supports mixing original examples.
- Splits: the Solver builder writes the full training table and samples
  validation from it. This overlapping sample is not an independent test set.
- InternVL: its VLM-R1 accuracy and format rewards are distinct from Qwen's
  binary format-and-answer reward.

`chartqa_reward.py` is byte-identical to both the original source and the reward
module loaded in the completed GPU runs, with SHA-256:

`a8ec266de31c900d27624660b9342e5f8dbd9a76c17c5dc26aaf3655ff1e1c73`

`qwen3vl_planner_clevr.py` retains its original name because the ChartQA loop
imports this shared module. This does not add a separate CLEVR or ZwZ training
pipeline. The main validated route uses `--planner_mode gpt5`.

For a new training installation, first ensure that its reward registry contains
`chartqa_reward` and points to this original implementation. The optional vendored
snapshot has passed reward-loading checks, but differs from the installed
framework used for the recorded GRPO updates. With that snapshot, the existing
custom-reward loader can register the original module using
`custom_reward_function.path="$PWD/chartqa_reward.py"` and
`custom_reward_function.name=compute_score` as launcher overrides. Do not assume
this substitutes for validation of a different training environment.
