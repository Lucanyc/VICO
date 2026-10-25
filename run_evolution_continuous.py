#!/usr/bin/env python3
"""
Planner-Executor-Solver Loop — VICO GenEnv, version E.

Drop-in alternative to run_evolution.py that replaces
the planner's prompt with continuous-control feedback, without modifying any
existing file.

Differences vs. the original loop:
  1. Uses a local subclass APIPlannerE:
       - SYSTEM_PROMPT rewritten to frame the task as continuous step-size
         control instead of discrete p̂-bucket templates.
       - _build_user_prompt shows FULL previous edit_instruction (no [:80]
         truncation) plus per-step quantitative deltas (Δp̂, Δrows,
         answer_changed) and a trajectory-level diagnosis
         (overshoot / stuck / oscillating).
  2. History entries now carry n_rows_before / n_rows_after so the planner
     can see the row-count trajectory across turns.

Utility classes and helpers (SolverEvaluator, atomic_json_save, verify_image,
TableEditor) are imported from the existing modules unchanged.

CLI is the same as the original loop. Recommended planner_mode is gpt5_e.

Example:
    CUDA_VISIBLE_DEVICES=6,7 python3 run_evolution_continuous.py \
        --data_json <...> --tables_dir <...> --png_dir <...> \
        --start 0 --n 30 --model_path <...> \
        --n_rollouts 8 --max_turns 5 \
        --planner_mode gpt5_e --planner_model openai/openai/gpt-5.4 \
        --output_dir ./vico_genenv_e/images \
        --output_json ./vico_genenv_e/trajectories_0_30.json \
        --resume
"""
import os, sys, json, re, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse utilities from the existing loop (unchanged).
from run_evolution import (
    atomic_json_save,
    verify_image,
    SolverEvaluator,
)
from table_editor import TableEditor
from api_planner import APIPlanner


# =============================================================================
# Planner E — continuous control + full quantitative feedback
# =============================================================================
class APIPlannerE(APIPlanner):
    """
    Subclass of APIPlanner. Overrides only SYSTEM_PROMPT and
    _build_user_prompt. Same .get_plan() signature and wire format.

    Expects history entries of the form:
      {"tool": "chart_rerender",
       "params": {"edit_instruction": ...},
       "p_hat": float|None,
       "new_answer": <value>|None,
       "n_rows_before": int|None,
       "n_rows_after": int|None,
       "error": str  (only on failure)}
    """

    SYSTEM_PROMPT = """You are a chart-editing planner controlling a Solver's pass rate p̂ using continuous step-size control.

PIPELINE (each turn): you output ONE chart_rerender instruction. The generator takes the ORIGINAL CSV + your instruction → produces a new CSV → the answer is programmatically recomputed from the cell at answer_cell → a chart is rendered → the Solver is evaluated → you receive p̂.

GOAL: bring p̂ into the band [0.35, 0.65] within the turn budget. Converge — do not explore.

CORE PRINCIPLE — CONTINUOUS CONTROL
Treat your edit as having a continuous intensity (how different the new table is from the original). Intensity is NOT a category; it is a scalar. Every turn, your only job is to pick the next intensity based on what the last one produced.

STEP-SIZE RULES (grounded in the numbers provided in the user message)
- If the last step OVERSHOT (p̂ crossed the band, |Δp̂| > 0.5): your next intensity must lie BETWEEN the last two attempts, not pick a third extreme. In practice, roughly halve the change you applied last time.
- If the last step BARELY MOVED p̂ (|Δp̂| < 0.1) and p̂ is still out of band: roughly DOUBLE the change you applied last time.
- If p̂ is close to the band (|p̂ − 0.5| < 0.2 but outside): apply a SMALL incremental nudge in the obvious direction.

INTENSITY PROXIES (use these to reason about size and to describe your edit concretely)
- Number of rows added / removed / changed. A strong, measurable proxy.
- Whether the answer value changes. This is a BINARY, HIGH-VARIANCE action: changing the answer typically produces a large and poorly controlled p̂ swing. Prefer answer-preserving edits for small adjustments. Only change the answer when you are deliberately resetting.
- Numeric distance between the answer row and distractor rows (closer = harder).

OUTPUT
Write edit_instruction concretely (specific row names and numeric values). In reasoning, state explicitly:
  (a) your estimate of the LAST step's intensity,
  (b) the intensity you are choosing NOW,
  (c) why that size given the observed Δp̂.

Output ONLY a JSON object (no markdown, no backticks):
{"tool": "chart_rerender", "params": {"edit_instruction": "<concrete edit>"}, "reasoning": "<intensity rationale>"}
"""

    @staticmethod
    def _fmt_rows(n_before, n_after):
        if n_before is None or n_after is None:
            return "?"
        return f"{n_before}→{n_after}"

    @staticmethod
    def _classify_delta(p_prev, p_cur, band_lo=0.35, band_hi=0.65):
        in_band = band_lo <= p_cur <= band_hi
        if in_band:
            return "entered band ✓"
        dp = p_cur - p_prev
        crossed = (p_prev - 0.5) * (p_cur - 0.5) < 0
        if abs(dp) > 0.5 and crossed:
            return f"OVERSHOT (crossed band, |Δp̂|={abs(dp):.2f})"
        if abs(dp) < 0.1:
            return "stuck (|Δp̂|<0.1)"
        same_side_closer = (
            (p_prev - 0.5) * (p_cur - 0.5) > 0
            and abs(p_cur - 0.5) < abs(p_prev - 0.5)
        )
        if same_side_closer:
            return "moving toward band"
        return "moved but off"

    def _build_user_prompt(self, question, answer, p_hat, history):
        lines = [
            f"Question: {question}",
            f"Current target answer value: {answer}",
            f"Current Solver p̂: {p_hat:.3f}   (goal band: [0.35, 0.65])",
        ]

        if not history:
            lines += [
                "",
                "This is the FIRST attempt. No prior feedback available.",
                "Propose a moderate-intensity edit appropriate to the current p̂.",
                "",
                "Output JSON only.",
            ]
            return "\n".join(lines)

        lines += ["", "=== Previous attempts (full detail, not truncated) ==="]

        prev_p = None
        prev_n_after = None
        prev_answer = None

        for i, h in enumerate(history, 1):
            instr = h.get("params", {}).get("edit_instruction", "") or ""
            p_val = h.get("p_hat")
            new_a = h.get("new_answer")
            n_before = h.get("n_rows_before")
            n_after = h.get("n_rows_after")
            err = h.get("error")

            p_str = f"{p_val:.3f}" if isinstance(p_val, (int, float)) else "n/a"

            lines.append("")
            lines.append(f"Attempt {i}:")
            lines.append(f"  instruction: {instr}")
            if err:
                lines.append(f"  result: FAILED — {err}")
            else:
                lines.append(
                    f"  result: p̂={p_str}, answer→{new_a}, "
                    f"rows {self._fmt_rows(n_before, n_after)}"
                )

            if i >= 2 and isinstance(p_val, (int, float)) and isinstance(prev_p, (int, float)):
                dp = p_val - prev_p
                if isinstance(n_after, int) and isinstance(prev_n_after, int):
                    dn_str = f"{n_after - prev_n_after:+d}"
                else:
                    dn_str = "?"
                ans_changed = (new_a != prev_answer)
                tag = self._classify_delta(prev_p, p_val)
                lines.append(
                    f"  Δ vs attempt {i-1}: Δp̂={dp:+.3f}, Δrows={dn_str}, "
                    f"answer_changed={'YES' if ans_changed else 'no'}  [{tag}]"
                )

            prev_p = p_val
            prev_n_after = n_after
            prev_answer = new_a

        # --- Trajectory-level diagnosis ---
        phats = [h.get("p_hat") for h in history if isinstance(h.get("p_hat"), (int, float))]
        if phats:
            lines += ["", "=== Trajectory diagnosis ==="]
            closest = min(phats, key=lambda p: abs(p - 0.5))
            lines.append(f"Closest p̂ to 0.5 so far: {closest:.3f}")

            if len(phats) >= 2:
                deltas = [phats[i] - phats[i - 1] for i in range(1, len(phats))]
                abs_big = sum(1 for d in deltas if abs(d) > 0.5)
                sign_flips = sum(
                    1 for i in range(1, len(deltas)) if deltas[i] * deltas[i - 1] < 0
                )

                if abs_big >= 1 and sign_flips >= 1:
                    lines.append(
                        f"⚠ OSCILLATING — {abs_big} big swing(s), {sign_flips} direction flip(s). "
                        f"Your intensity choices are too large. Next step MUST be smaller than "
                        f"the midpoint of the last two attempts."
                    )
                elif abs_big >= 1:
                    lines.append(
                        "⚠ Overshoot detected. Next intensity should be roughly HALF of the last change."
                    )
                elif all(abs(d) < 0.1 for d in deltas):
                    lines.append(
                        "⚠ Stuck — p̂ barely moving. DOUBLE your intensity change for next step."
                    )
                else:
                    lines.append("Moving. Continue in the same direction with a smaller step.")

            answers = [h.get("new_answer") for h in history]
            ans_changes = sum(
                1
                for i in range(1, len(answers))
                if answers[i] is not None
                and answers[i - 1] is not None
                and answers[i] != answers[i - 1]
            )
            if ans_changes >= 1:
                lines.append(
                    f"Answer value changed {ans_changes} time(s) across attempts — this is "
                    f"likely contributing to large p̂ swings. For the next step, prefer an edit "
                    f"that KEEPS the answer value unchanged (add/remove non-answer rows only)."
                )

        lines += [
            "",
            "Decide your next edit. In `reasoning`, state: estimated last intensity, "
            "chosen next intensity, and why.",
            "Output JSON only.",
        ]
        return "\n".join(lines)


# =============================================================================
# Run loop — identical to original except history entries carry row counts
# =============================================================================
def run_loop_e(rewriter_planner, generator, solver, sample,
               tables_dir, png_dir, output_dir,
               max_turns=5, target_lo=0.35, target_hi=0.65):

    sample_id = sample.get("_id", "unknown")
    imgname = sample["imgname"]
    question = sample["query"]
    original_answer = str(sample["label"])
    tier = sample.get("tier", 2)
    answer_cell = sample.get("answer_cell")

    basename = os.path.splitext(imgname)[0]
    csv_path = os.path.join(tables_dir, f"{basename}.csv")
    img_path = os.path.join(png_dir, imgname)

    traj = {
        "sample_id": sample_id, "imgname": imgname,
        "question": question[:100], "original_answer": original_answer,
        "tier": tier, "answer_cell": answer_cell,
        "turns": [], "final_phat": None, "final_image_path": None,
        "final_answer": None, "status": "pending", "total_turns": 0,
    }

    if not os.path.exists(img_path):
        traj["status"] = "failed"
        traj["error"] = f"Image not found: {img_path}"
        return traj

    if not os.path.exists(csv_path):
        traj["status"] = "failed"
        traj["error"] = f"CSV not found: {csv_path}"
        return traj

    print(f"  [Original] Evaluating...")
    p0 = solver.evaluate(img_path, question, original_answer)
    traj["p_hat_original"] = p0

    if target_lo <= p0 <= target_hi:
        traj.update({"status": "already_good", "final_phat": p0,
                     "final_image_path": img_path, "final_answer": original_answer})
        return traj

    history = []
    cur_p = p0
    current_answer = original_answer

    for turn in range(1, max_turns + 1):
        print(f"  [Turn {turn}] Planner (p̂={cur_p:.3f})...")

        plan = rewriter_planner.get_plan(
            image_path=img_path, question=question, answer=current_answer,
            p_hat_original=cur_p, scene_objects=[], history=history,
        )

        edit_instruction = plan.get("params", {}).get("edit_instruction", "Modify chart data")
        reasoning = plan.get("reasoning", "")
        print(f"  [Turn {turn}] Direction: {edit_instruction[:60]}...")

        mod_path = os.path.join(output_dir, f"{sample_id}_turn{turn}.png")
        gen_result = generator.generate(
            csv_path=csv_path, question=question,
            original_answer=current_answer,
            answer_cell=answer_cell, tier=tier,
            edit_instruction=edit_instruction,
            output_path=mod_path, p_hat=cur_p,
        )

        if not gen_result["success"]:
            print(f"  [Turn {turn}] Failed: {gen_result['error']}")
            traj["turns"].append({
                "turn": turn, "plan": plan, "exec_success": False,
                "exec_error": gen_result["error"],
                "new_answer": None, "p_hat": None,
            })
            history.append({
                "tool": "chart_rerender",
                "params": plan.get("params", {}),
                "p_hat": None,
                "new_answer": None,
                "n_rows_before": None,
                "n_rows_after": None,
                "error": gen_result["error"],
            })
            continue

        if not verify_image(mod_path):
            print(f"  [Turn {turn}] Image verification failed")
            continue

        new_answer = gen_result["new_answer"]
        n_before = gen_result["n_rows_before"]
        n_after = gen_result["n_rows_after"]
        print(f"  [Turn {turn}] Rows: {n_before}→{n_after}, "
              f"answer: {current_answer}→{new_answer}")

        print(f"  [Turn {turn}] Evaluating...")
        ph = solver.evaluate(mod_path, question, str(new_answer))
        print(f"  [Turn {turn}] → p̂={ph:.3f}", end="")

        traj["turns"].append({
            "turn": turn, "plan": plan, "exec_success": True,
            "modified_image": mod_path,
            "new_answer": new_answer,
            "n_rows_before": n_before, "n_rows_after": n_after,
            "p_hat": ph,
        })
        history.append({
            "tool": "chart_rerender",
            "params": plan.get("params", {}),
            "p_hat": ph,
            "new_answer": new_answer,
            "n_rows_before": n_before,
            "n_rows_after": n_after,
        })

        if target_lo <= ph <= target_hi:
            print(f" ✓ TARGET HIT!")
            traj.update({"status": "success", "final_phat": ph,
                         "final_image_path": mod_path,
                         "final_answer": new_answer, "total_turns": turn})
            return traj
        print(f" ({'too_easy' if ph > target_hi else 'too_hard'})")

        cur_p = ph
        current_answer = new_answer

    valid = [t for t in traj["turns"] if isinstance(t.get("p_hat"), float)]
    if valid:
        best = min(valid, key=lambda t: abs(t["p_hat"] - 0.5))
        traj.update({"status": "partial", "final_phat": best["p_hat"],
                     "final_image_path": best.get("modified_image"),
                     "final_answer": best.get("new_answer"),
                     "total_turns": max_turns})
    else:
        traj.update({"status": "failed", "total_turns": max_turns})
    return traj


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--tables_dir", required=True)
    parser.add_argument("--png_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_rollouts", type=int, default=8)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--target_lo", type=float, default=0.35)
    parser.add_argument("--target_hi", type=float, default=0.65)
    parser.add_argument("--planner_mode", default="gpt5_e",
                        choices=["gpt5_e", "gpt5"],
                        help="gpt5_e uses the E (continuous-control) planner; "
                             "gpt5 falls back to the original APIPlanner.")
    parser.add_argument("--planner_model", default="gpt-5")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)

    with open(args.data_json) as f:
        all_data = json.load(f)
    samples = all_data[args.start:args.start + args.n]
    for i, s in enumerate(samples):
        s["_id"] = f"chart_{args.start + i}"
    print(f"Samples: {len(samples)} (index {args.start}-{args.start + args.n - 1})")

    trajs = []
    done = set()
    if args.resume and os.path.exists(args.output_json):
        with open(args.output_json) as f:
            trajs = json.load(f)
        done = set(t["sample_id"] for t in trajs)
        print(f"Resume: {len(trajs)} done")

    todo = [(s["_id"], s) for s in samples if s["_id"] not in done]
    print(f"To process: {len(todo)}")

    if todo:
        if args.planner_mode == "gpt5_e":
            planner = APIPlannerE(model=args.planner_model)
            print(f"Using APIPlannerE (continuous-control, model={args.planner_model})")
        else:
            planner = APIPlanner(model=args.planner_model)
            print(f"Using APIPlanner (baseline, model={args.planner_model})")

        generator = TableEditor()
        solver = SolverEvaluator(args.model_path, args.n_rollouts)

        for idx, (sid, s) in enumerate(todo):
            print(f"\n{'─'*60}\n[{idx+1}/{len(todo)}] {sid}")
            print(f"  Q: {s['query'][:70]}...\n  A: {s['label']}, tier={s.get('tier', '?')}")
            tr = run_loop_e(planner, generator, solver, s,
                            args.tables_dir, args.png_dir, args.output_dir,
                            args.max_turns, args.target_lo, args.target_hi)
            trajs.append(tr)

            fp = tr.get("final_phat")
            fa = tr.get("final_answer", "?")
            fp_str = f"{fp:.3f}" if isinstance(fp, float) else "?"
            print(f"  → {tr['status']} | p̂: {tr.get('p_hat_original','?')} → {fp_str} "
                  f"| answer: {s['label']} → {fa}")

            atomic_json_save(trajs, args.output_json)
            print(f"  💾 {len(trajs)} saved")

    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    if not trajs:
        print("  No trajectories.")
        return

    sc = Counter(t["status"] for t in trajs)
    fps = [t["final_phat"] for t in trajs if isinstance(t.get("final_phat"), float)]
    sweet = sum(1 for p in fps if .35 <= p <= .65) if fps else 0
    print(f"  Total: {len(trajs)}, Status: {dict(sc)}")
    if fps:
        print(f"  Sweet zone: {sweet}/{len(fps)} ({sweet/len(fps)*100:.1f}%)")
    print(f"  Saved: {args.output_json}")


if __name__ == "__main__":
    main()
