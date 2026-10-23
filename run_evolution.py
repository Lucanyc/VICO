#!/usr/bin/env python3
"""Run VICO with configurable data fields, editor, Planner, and answer metric."""
import argparse
import json
from pathlib import Path

from vico.backends import SimplePlanner, TableBackend, load_factory
from vico.data import load_samples, resolve_image, save_json
from vico.pipeline import evolve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--field-map", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--editor", default="table", help="table or module:factory")
    ap.add_argument("--editor-config", type=json.loads, default={})
    ap.add_argument("--planner", default="simple", help="simple, api, local, or module:factory")
    ap.add_argument("--planner-config", type=json.loads, default={})
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-rollouts", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--target-lo", type=float, default=0.35)
    ap.add_argument("--target-hi", type=float, default=0.65)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--metric", choices=["exact", "relaxed"], default="relaxed")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--prompt-style", choices=["answer_tag", "boxed", "direct"], default="answer_tag")
    ap.add_argument("--system-prompt-file", type=Path)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.target_lo <= args.target_hi <= 1 or args.max_turns < 1:
        ap.error("Invalid target band or max-turns")
    if args.n_rollouts < 1 or args.max_tokens < 1 or args.tolerance < 0:
        ap.error("Invalid rollout count, token limit, or tolerance")
    field_map = json.loads(args.field_map.read_text()) if args.field_map else None
    samples = load_samples(args.data, field_map)
    if args.n > 0:
        samples = samples[:args.n]
    for sample in samples:
        sample["image"] = str(resolve_image(sample, args.image_root))
        if "table" in sample:
            sample["table"] = str((args.data.parent / sample["table"]).resolve())
    output = args.output_dir / "trajectories.json"
    if output.exists() and not args.resume:
        ap.error("Output already exists; use --resume or another output directory")
    trajectories = json.loads(output.read_text()) if args.resume and output.exists() else []
    done = {t["sample_id"] for t in trajectories}
    todo = [s for s in samples if s["id"] not in done]
    if not todo:
        print("No samples remaining")
        return
    editor = TableBackend(**args.editor_config) if args.editor == "table" else load_factory(args.editor, args.editor_config)
    if args.planner == "simple":
        planner = SimplePlanner()
    elif args.planner == "api":
        from api_planner import APIPlanner
        planner = APIPlanner(**args.planner_config)
    elif args.planner == "local":
        from local_planner import LocalPlanner
        planner = LocalPlanner(**args.planner_config)
    else:
        planner = load_factory(args.planner, args.planner_config)
    from vico.solver import VLLMSolver
    solver = VLLMSolver(args.model, n_rollouts=args.n_rollouts, tp=args.tp,
                        max_tokens=args.max_tokens, metric=args.metric,
                        tolerance=args.tolerance, enforce_eager=args.enforce_eager,
                        temperature=args.temperature, gpu_memory=args.gpu_memory_utilization,
                        max_model_len=args.max_model_len, prompt_style=args.prompt_style,
                        system_prompt=args.system_prompt_file.read_text() if args.system_prompt_file else None)
    for sample in todo:
        trajectory = evolve(sample, planner, editor, solver, args.output_dir / "images",
                            args.max_turns, args.target_lo, args.target_hi)
        trajectories.append(trajectory)
        save_json(output, trajectories)
        print(f"{sample['id']}: {trajectory['status']}, p={trajectory['final_phat']}")


if __name__ == "__main__":
    main()
