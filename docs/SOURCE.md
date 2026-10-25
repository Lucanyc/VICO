# Source and publication changes

The original workspace was read directly on 2026-10-25. This release replaces the
previous generalized implementation with the actual research entry points.
Each retained file has its source and publication SHA-256 in
`SOURCE_PROVENANCE.json`.

Changes are limited to module/class/dataset labels, machine-specific path
placeholders, external model/data configuration, and launcher wiring. The Qwen
launcher loads the included verl source and registers the original reward module
through verl's existing custom-module mechanism. SFT and GRPO launchers forward
extra command-line arguments to support bounded checks. These changes are made
only in the publication copy.
Unused reward registrations for other research environments are excluded from
the vendored framework; the table and tool reward implementations are retained.

The following research behaviors are preserved:

- Qwen reward: +1 only if all four thinking/answer tag strings are present and the
  extracted answer is correct; -1 otherwise. Numbers use 5% relative tolerance
  and the original near-zero rule. Text uses original normalization and the
  short-reference prefix rule. The original reward manager is retained too.
- Generation feedback: its own answer checker uses 0.5% numeric tolerance and
  scale comparisons. It is not replaced by the training reward.
- Rewriter weighting: `exp(-beta * (p_hat - 0.5)^2)`, beta 4 by default, followed
  by stochastic repetition up to 5 and LoRA supervised training.
- Solver data: no historical difficulty cutoff. Successful edited turns are
  kept even when their recorded pass rate is 0 or 1. The source also checks image,
  answer, and pass-rate metadata, and supports original-example mixing.
- Split behavior: the original Solver builder writes the full training table,
  then samples validation rows from it. The original Rewriter split is retained.
  These splits must not be presented as an independent generalization test.
- InternVL: the VLM-R1 accuracy and format reward functions remain distinct from
  Qwen's binary format-and-answer reward.

The original reward module and the module registered in the research training
environment were byte-identical, with SHA-256:

`a8ec266de31c900d27624660b9342e5f8dbd9a76c17c5dc26aaf3655ff1e1c73`

The source's 23 embedded reward cases are retained. The earlier generalized loop,
configurable replacement scorer, grouped-split training builders, and new Planner
fallback behavior are not used by this release. Bugs or limitations discovered
in the original implementation are reported in `VALIDATION.md`; algorithmic
fixes are not silently introduced.
