# Publication changes

- Replaced dataset-bound file names, class names, registry keys, environment
  variables, example paths, and output names with reusable component names.
- Added a common JSON/JSONL schema adapter, configurable answer extraction and
  metrics, a dataset-independent feedback loop, and editor/Planner factories.
- Retained table editing as an explicit backend. Earlier loop/evaluation variants
  remain available for comparison; they are not the recommended generic entry.
- Preserved full trajectory questions and exact per-turn Planner inputs. Saved
  modified CSV state alongside images so subsequent edits use consistent data.
- Replaced unrelated local-Planner scene-tool defaults with a configurable edit
  interface. Its old domain-specific demo is excluded.
- Removed fixed GPU-index selection from the primary inference path. It uses the
  process's CUDA allocation and configurable tensor parallelism.
- Made main evaluation reject missing images and mismatched resume settings;
  moved heavy inference imports after argument parsing and input validation.
- Grouped RWR splits by source image and rewritten Solver splits by connected
  seed/generated-image groups to prevent group overlap.
- Registered the reward through verl's custom hook, honored the configured score
  function, exposed exact/numeric matching, removed permissive text-prefix
  matches, and guarded empty-response token placement.
- Made launchers portable and configurable. Retained third-party license notices
  and removed two unrelated locally added reward managers from the vendor copy.
- Inspected training data through Parquet metadata instead of loading the full
  table for startup statistics. Scoped data/model artifact ignore rules to the
  repository root so vendored model implementations and configs remain tracked.
- Excluded private export utilities/manifests, raw logs, checkpoints, credentials,
  original datasets, and dataset archives. Source SHA-256 provenance is retained.

All changes were made in publication/test copies. The original research workspace
was not modified. These fixes can change behavior relative to historical runs;
no equivalence to the paper's benchmark numbers is claimed.
