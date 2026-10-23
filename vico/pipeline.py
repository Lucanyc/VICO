"""Planner → editor → Solver feedback, independent of dataset and renderer.

An editor implements edit(sample, plan, output_path, p_hat) and returns a new
sample (including its valid answer). A Solver implements evaluate(image,
question, answer) -> pass rate. A Planner implements get_plan(**context).
"""
import hashlib
import math
from pathlib import Path


def evolve(sample, planner, editor, solver, output_dir, max_turns=5,
           target_lo=0.35, target_hi=0.65):
    if not 0 <= target_lo <= target_hi <= 1 or max_turns < 1:
        raise ValueError("Invalid target band or max_turns")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha256(sample["id"].encode()).hexdigest()[:16]
    current = dict(sample)
    history = []

    def score(s):
        p = float(solver.evaluate(s["image"], s["question"], s["answer"]))
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"Invalid Solver pass rate: {p}")
        return p

    p = score(current)
    trajectory = {"sample_id": sample["id"], "imgname": sample["image"],
                  "question": sample["question"], "original_answer": sample["answer"],
                  "p_hat_original": p, "turns": [], "status": "pending"}
    if target_lo <= p <= target_hi:
        trajectory.update(status="already_good", final_phat=p,
                          final_image_path=current["image"],
                          final_answer=current["answer"], total_turns=0)
        return trajectory
    midpoint = (target_lo + target_hi) / 2
    for turn in range(1, max_turns + 1):
        plan = planner.get_plan(image_path=current["image"], question=current["question"],
                                answer=current["answer"], p_hat_original=p,
                                scene_objects=[], history=history)
        record = {"turn": turn, "plan": plan, "input_image": current["image"],
                  "input_answer": current["answer"], "input_question": current["question"],
                  "input_phat": p, "exec_success": False}
        try:
            edited = editor.edit(dict(current), plan, output_dir / f"{stem}_{turn}.png", p)
            if any(k not in edited for k in ("image", "question", "answer")):
                raise ValueError("Editor must return image, question, answer")
            if not Path(edited["image"]).is_file():
                raise ValueError("Editor did not create an image")
        except Exception as error:
            record.update(exec_error=str(error), new_answer=None, p_hat=None)
            trajectory["turns"].append(record)
            history.append({**plan, "p_hat": None, "error": str(error)})
            continue
        new_p = score(edited)
        record.update(exec_success=True, modified_image=edited["image"],
                      new_answer=str(edited["answer"]), question=edited["question"], p_hat=new_p)
        trajectory["turns"].append(record)
        history.append({**plan, "p_hat": new_p})
        current, p = edited, new_p
        if target_lo <= p <= target_hi:
            trajectory["status"] = "success"
            break
    valid = [t for t in trajectory["turns"] if t["exec_success"]]
    best = min(valid, key=lambda t: abs(t["p_hat"] - midpoint)) if valid else None
    trajectory.update(final_phat=best["p_hat"] if best else None,
                      final_image_path=best["modified_image"] if best else None,
                      final_answer=best["new_answer"] if best else None,
                      final_question=best["question"] if best else None,
                      total_turns=len(trajectory["turns"]))
    if trajectory["status"] == "pending":
        trajectory["status"] = "partial" if best else "failed"
    return trajectory
