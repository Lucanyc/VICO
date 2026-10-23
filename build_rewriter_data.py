#!/usr/bin/env python3
"""
Build Rewriter RWR Training Data from VICO GenEnv Trajectories

Same logic as build_rewriter_rwr_genenv_data.py (visual reasoning), adapted for VICO:
  - imgname → full path via --png_dir
  - answer changes per turn (GenEnv mode)
  - RWR weight: exp(-β(p̂ - 0.5)²)

Usage:
    python3 build_rewriter_data.py \
        --traj_files vico_genenv_test/trajectories_*.json \
        --png_dir /path/to/vico/train/png \
        --output_dir vico_rewriter_rwr \
        --beta 4.0 --max_repeats 5
"""

import os, json, argparse, random, math, glob, re
from collections import Counter
from PIL import Image


from local_planner import LocalPlanner
SYSTEM_PROMPT = LocalPlanner.SYSTEM_PROMPT


def compute_rwr_weight(p_hat, alpha=0.5, beta=4.0):
    return math.exp(-beta * (p_hat - alpha) ** 2)


def weight_to_repeats(weight, max_repeats=5):
    scaled = weight * max_repeats
    base = int(scaled)
    frac = scaled - base
    if random.random() < frac:
        base += 1
    return min(base, max_repeats)


def is_valid_image(path):
    if not path or not os.path.exists(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False


def clean_markdown(text):
    if not text:
        return text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__([^_]*)__', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    return text.strip()


def build_user_message(question, answer, p_hat, history):
    msg = f"Question: {question}\nAnswer: {answer}\n"
    msg += f"Current Solver pass rate (p̂): {p_hat:.3f}\n"

    if history:
        msg += "\nEdit history (cumulative):\n"
        for i, h in enumerate(history, 1):
            tool = h.get("tool", "edit")
            instr = h.get("params", {}).get("edit_instruction", "")[:60]
            p_val = h.get("p_hat", "?")
            p_str = f"{p_val:.3f}" if isinstance(p_val, (int, float)) else str(p_val)
            err = h.get("error", "")
            err_str = f" [FAILED: {err[:40]}]" if err else ""
            msg += f'  Step {i}: {tool}("{instr}...") → p̂={p_str}{err_str}\n'
    else:
        msg += "\nNo edits applied yet. This is the original image.\n"

    msg += "\nDecide the next edit."
    return msg


def build_assistant_message(plan):
    reasoning = plan.get("reasoning", "").strip()
    reasoning = clean_markdown(reasoning)
    if not reasoning:
        reasoning = "Selecting the best modification based on current p̂."

    tool = plan.get("tool", "edit")
    params = plan.get("params", {})
    if "edit_instruction" in params:
        params = dict(params)
        params["edit_instruction"] = clean_markdown(params["edit_instruction"])

    tool_json = {"tool": tool, "params": params}
    return (f"<think>\n{reasoning}\n</think>\n"
            f"<tool_call>\n{json.dumps(tool_json, ensure_ascii=False)}\n</tool_call>")


def get_image_for_turn(traj, turn_idx, png_dir):
    """Get the image the Rewriter would have seen at this turn."""
    if turn_idx == 0:
        # Original chart image
        imgname = traj.get("imgname", "")
        full_path = os.path.join(png_dir, imgname) if png_dir else imgname
        if os.path.exists(full_path):
            return os.path.abspath(full_path)
        return None
    # Previous turn's modified image
    for j in range(turn_idx - 1, -1, -1):
        t = traj["turns"][j]
        if t.get("exec_success") and t.get("modified_image") and os.path.exists(t["modified_image"]):
            return os.path.abspath(t["modified_image"])
    # Fallback to original
    imgname = traj.get("imgname", "")
    full_path = os.path.join(png_dir, imgname) if png_dir else imgname
    if os.path.exists(full_path):
        return os.path.abspath(full_path)
    return None


def get_answer_for_turn(traj, turn_idx):
    """Get the answer the Rewriter would have seen at this turn."""
    if turn_idx == 0:
        return str(traj.get("original_answer", ""))
    for j in range(turn_idx - 1, -1, -1):
        t = traj["turns"][j]
        if t.get("exec_success") and t.get("new_answer") is not None:
            return str(t["new_answer"])
    return str(traj.get("original_answer", ""))


def get_phat_for_turn(traj, turn_idx):
    """Get the p̂ the Rewriter would have seen at this turn."""
    if turn_idx == 0:
        return traj.get("p_hat_original", 0.5)
    for j in range(turn_idx - 1, -1, -1):
        t = traj["turns"][j]
        if isinstance(t.get("p_hat"), float):
            return t["p_hat"]
    return traj.get("p_hat_original", 0.5)


def build_samples(trajs, beta, max_repeats, png_dir, system_prompt=SYSTEM_PROMPT):
    samples = []
    stats = Counter()

    for traj in trajs:
        question = traj.get("question", "")
        status = traj.get("status", "")

        if status in ("already_good", "failed"):
            stats[f"skip_{status}"] += 1
            continue

        turns = traj.get("turns", [])
        if not turns:
            stats["skip_no_turns"] += 1
            continue

        history = []

        for turn_idx, turn in enumerate(turns):
            if not turn.get("exec_success"):
                stats["skip_exec_failed"] += 1
                history.append({
                    "tool": turn.get("plan", {}).get("tool", "edit"),
                    "params": turn.get("plan", {}).get("params", {}),
                    "p_hat": None,
                    "error": turn.get("exec_error", "failed"),
                })
                continue

            out_p = turn.get("p_hat")
            if not isinstance(out_p, float):
                stats["skip_no_phat"] += 1
                continue

            weight = compute_rwr_weight(out_p, alpha=0.5, beta=beta)
            repeats = weight_to_repeats(weight, max_repeats=max_repeats)

            if repeats == 0:
                stats["zero_weight_dropped"] += 1
                history.append({
                    "tool": turn.get("plan", {}).get("tool", "edit"),
                    "params": turn.get("plan", {}).get("params", {}),
                    "p_hat": out_p,
                })
                continue

            image_path = turn.get("input_image") or get_image_for_turn(traj, turn_idx, png_dir)
            if not is_valid_image(image_path):
                stats["skip_bad_image"] += 1
                continue

            input_answer = turn.get("input_answer", get_answer_for_turn(traj, turn_idx))
            input_phat = turn.get("input_phat", get_phat_for_turn(traj, turn_idx))

            user_text = build_user_message(turn.get("input_question", question), input_answer, input_phat, history)
            assistant_text = build_assistant_message(turn["plan"])

            sample = {
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": f"<image>\n{user_text}"},
                    {"from": "gpt", "value": assistant_text},
                ],
                "images": [image_path],
                "meta": {
                    "sample_id": traj.get("sample_id", ""),
                    "source_image": traj.get("imgname", traj.get("sample_id", "")),
                    "turn": turn.get("turn", turn_idx + 1),
                    "input_phat": input_phat,
                    "output_phat": out_p,
                    "input_answer": input_answer,
                    "output_answer": str(turn.get("new_answer", "")),
                    "weight": weight,
                    "repeats": repeats,
                    "source": "vico_genenv",
                },
            }

            for _ in range(repeats):
                samples.append(sample)
            stats["kept"] += 1

            if 0.35 <= out_p <= 0.65:
                stats["sweet_zone"] += 1

            history.append({
                "tool": turn.get("plan", {}).get("tool", "edit"),
                "params": turn.get("plan", {}).get("params", {}),
                "p_hat": out_p,
            })

    return samples, stats


def main():
    parser = argparse.ArgumentParser(description="Build VICO Rewriter RWR data")
    parser.add_argument("--traj_files", nargs="+", required=True)
    parser.add_argument("--png_dir", default="",
                        help="Root for relative legacy trajectory image paths")
    parser.add_argument("--system-prompt-file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--max_repeats", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Expand glob
    traj_files = []
    for pattern in args.traj_files:
        if '*' in pattern:
            traj_files.extend(sorted(glob.glob(pattern)))
        else:
            traj_files.append(pattern)

    # Load (dedup by sample_id)
    print(f"Loading trajectories...")
    seen = {}
    for f in traj_files:
        try:
            data = json.load(open(f))
            print(f"  {os.path.basename(f)}: {len(data)} trajectories")
            for t in data:
                sid = t.get("sample_id", "")
                if sid not in seen:
                    seen[sid] = t
        except Exception as e:
            print(f"  CORRUPTED: {f} ({e})")

    trajs = list(seen.values())
    print(f"  Total unique: {len(trajs)}")

    # Build samples
    print(f"\nBuilding RWR samples (beta={args.beta}, max_repeats={args.max_repeats})...")
    system_prompt = open(args.system_prompt_file).read() if args.system_prompt_file else SYSTEM_PROMPT
    samples, stats = build_samples(trajs, args.beta, args.max_repeats, args.png_dir, system_prompt)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"\n  Total samples (with oversampling): {len(samples)}")
    print(f"  Unique turns: {stats.get('kept', 0)}")
    print(f"  Sweet zone: {stats.get('sweet_zone', 0)}")

    if not samples:
        print("  ERROR: No samples!")
        return

    # Weight distribution
    weights = [s["meta"]["weight"] for s in samples]
    high = sum(1 for w in weights if w > 0.8)
    med = sum(1 for w in weights if 0.5 < w <= 0.8)
    low = sum(1 for w in weights if 0.2 < w <= 0.5)
    vlow = sum(1 for w in weights if w <= 0.2)
    print(f"\n  Weight distribution:")
    print(f"    high (>0.8): {high}")
    print(f"    med (0.5-0.8): {med}")
    print(f"    low (0.2-0.5): {low}")
    print(f"    very low (<=0.2): {vlow}")

    # Save
    print(f"\n{'='*60}")
    print("Saving")
    print(f"{'='*60}")

    # Split by source image before assigning oversampled turns.
    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    groups = sorted({s["meta"]["source_image"] for s in samples})
    if len(groups) < 2:
        raise ValueError("At least two source images are required for a disjoint split")
    random.shuffle(groups)
    cut = max(1, min(len(groups) - 1, int(len(groups) * args.train_ratio)))
    train_groups = set(groups[:cut])
    train = [s for s in samples if s["meta"]["source_image"] in train_groups]
    val = [s for s in samples if s["meta"]["source_image"] not in train_groups]
    random.shuffle(train)
    random.shuffle(val)

    for split, data in [("train", train), ("val", val)]:
        llama = [{"conversations": s["conversations"], "images": s["images"]} for s in data]
        path = os.path.join(args.output_dir, f"rewriter_rwr_{split}_llama.json")
        with open(path, "w") as f:
            json.dump(llama, f, ensure_ascii=False, indent=2)
        print(f"  {split}: {path} ({len(llama)} samples)")

    full_path = os.path.join(args.output_dir, "rewriter_rwr_train_full.json")
    with open(full_path, "w") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    print(f"  Full: {full_path}")


if __name__ == "__main__":
    main()
