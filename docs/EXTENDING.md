# Add an environment

The common pipeline consumes a dictionary with `id`, `image`, `question`, and
`answer`. Additional keys are passed through to your editor.

An editor factory returns an object with:

```python
def edit(self, sample, plan, output_path, p_hat):
    # Render the requested modification and verify/recompute its label.
    return {**sample, "image": str(output_path), "answer": verified_answer}
```

Raise an exception when an edit cannot be executed or verified. The loop records
the failed attempt and keeps the last valid state. Return updated environment
metadata when needed for subsequent edits. The editor is responsible for semantic
label validity; the generic loop only checks the returned structure/image path.

A Planner implements `get_plan(image_path, question, answer, p_hat_original,
scene_objects, history)` and returns a dictionary containing `tool`, `params`, and
optionally `reasoning`. A Solver implements `evaluate(image, question, answer)`
and returns a finite pass rate in [0, 1]. It must raise inference errors instead
of silently converting them into a zero pass rate.

Pass your factories to `run_evolution.py` as `package.module:factory`, with JSON
keyword arguments in `--editor-config` / `--planner-config`. Direct Python callers
can inject any compatible Solver into `vico.pipeline.evolve`.

The included table API Planner accepts a `system_prompt` override. The local
Planner accepts both `system_prompt` and `valid_tools`. Use a prompt describing
the capabilities and restrictions of your actual editor. `build_rewriter_data.py`
accepts `--system-prompt-file` so training inputs can match inference instructions.

The included reward is binary: +1 for the expected response format and a correct
answer, -1 otherwise. Configure `custom_reward_function.reward_kwargs.metric`
(`exact` or `relaxed`), `tolerance`, and `require_format` through Hydra. A new task
with symbolic, multi-answer, or semantic labels needs its own verifier. Supply
it through `REWARD_PATH`, `REWARD_FUNCTION`, and a compatible `REWARD_MANAGER`.
