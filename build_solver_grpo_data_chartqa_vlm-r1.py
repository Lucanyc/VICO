#!/usr/bin/env python3
"""
Build Solver GRPO Training Data for VLM-R1 framework — ChartQA + GenEnv mix.

Same trajectory parsing as build_solver_grpo_data_chartqa.py, but outputs
VLM-R1's JSONL format instead of verl's parquet:

  {"id": ..., "image": "<abs path>",
   "conversations": [{"from":"human","value":"<image>question"},
                     {"from":"gpt","value":"answer"}],
   "accu_reward_method": "default"}

VLM-R1 wraps the question with its own question_template that injects
<think>/<answer> instructions, and wraps the answer with <answer>...</answer>
at load time, so jsonl carries bare question + bare answer.

Usage:
    python3 build_solver_grpo_data_chartqa_vlm-r1.py \\
        --traj_files \\
            chartqa_genenv_gpt5/trajectories_*.json \\
            chartqa_genenv_round2/trajectories_*.json \\
            chartqa_genenv_round3/newprompt_trajectories_*.json \\
            chartqa_genenv_round4/trajectories_*.json \\
        --original_data /path/to/dataset/chartqa/train/train_augmented.json \\
        --png_dir /path/to/dataset/chartqa/train/png \\
        --n_original 2200 \\
        --output chartqa_solver_grpo_vlmr1.jsonl
"""

import os, json, argparse, random, glob
from collections import Counter
from PIL import Image


def verify_image(path):
    if not path or not os.path.exists(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def extract_from_genenv_trajectories(traj_files):
    """Same logic as build_solver_grpo_data_chartqa.py."""
    samples = []
    stats = Counter()

    for traj_file in traj_files:
        print(f"  Loading: {os.path.basename(traj_file)}")
        try:
            with open(traj_file) as f:
                trajs = json.load(f)
        except Exception as e:
            print(f"    CORRUPTED: {e}")
            stats["file_corrupted"] += 1
            continue

        print(f"    {len(trajs)} trajectories")

        for traj in trajs:
            status = traj.get("status", "")
            question = traj.get("question", "")
            original_answer = traj.get("original_answer", traj.get("label", ""))
            sample_id = traj.get("sample_id", "")

            if not question:
                stats["skip_no_question"] += 1
                continue

            if status == "already_good":
                img_path = traj.get("final_image_path", "")
                if img_path and verify_image(img_path):
                    samples.append({
                        "image_path": os.path.abspath(img_path),
                        "question": question,
                        "answer": str(original_answer),
                        "p_hat": traj.get("p_hat_original"),
                        "sample_id": sample_id,
                        "source": "genenv_already_good",
                    })
                    stats["already_good"] += 1
                continue

            if status == "failed":
                stats["skip_failed"] += 1
                continue

            for turn in traj.get("turns", []):
                if not turn.get("exec_success"):
                    stats["skip_exec_failed"] += 1
                    continue
                turn_image = turn.get("modified_image")
                turn_answer = turn.get("new_answer")
                turn_p = turn.get("p_hat")
                if not turn_image or not verify_image(turn_image):
                    stats["skip_bad_image"] += 1
                    continue
                if turn_answer is None:
                    stats["skip_no_answer"] += 1
                    continue
                if not isinstance(turn_p, (int, float)):
                    stats["skip_no_phat"] += 1
                    continue

                samples.append({
                    "image_path": os.path.abspath(turn_image),
                    "question": question,
                    "answer": str(turn_answer),
                    "p_hat": turn_p,
                    "sample_id": f"{sample_id}_t{turn.get('turn', 0)}",
                    "source": "genenv",
                })
                stats["kept"] += 1

                if 0.35 <= turn_p <= 0.65:
                    stats["sweet_zone"] += 1
                elif turn_p > 0.8:
                    stats["too_easy"] += 1
                elif turn_p < 0.2:
                    stats["too_hard"] += 1

    return samples, stats


def extract_original_samples(original_data_path, png_dir, n_samples, seed=42):
    with open(original_data_path) as f:
        data = json.load(f)

    random.seed(seed)
    valid = []
    for s in data:
        img_path = os.path.join(png_dir, s["imgname"])
        if os.path.exists(img_path):
            valid.append((s, os.path.abspath(img_path)))
    random.shuffle(valid)
    selected = valid[:n_samples]

    samples = []
    for s, img_path in selected:
        samples.append({
            "image_path": img_path,
            "question": s["query"],
            "answer": str(s["label"]),
            "p_hat": -1.0,
            "sample_id": str(s.get("imgname", "")),
            "source": "original",
        })
    return samples


def deduplicate(samples):
    """Same dedup as the original script — keep all original; dedup GenEnv by (img,q)."""
    original = [s for s in samples if s.get("source") == "original"]
    genenv = [s for s in samples if s.get("source") != "original"]
    best = {}
    for s in genenv:
        key = s["image_path"] + "|||" + s["question"]
        if key not in best:
            best[key] = s
        else:
            old_p = best[key].get("p_hat")
            new_p = s.get("p_hat")
            if isinstance(new_p, (int, float)) and isinstance(old_p, (int, float)):
                if new_p >= 0 and (old_p < 0 or abs(new_p - 0.5) < abs(old_p - 0.5)):
                    best[key] = s
    return original + list(best.values())


def to_vlmr1_jsonl(samples, output_path, val_ratio=0.05, seed=42, accu_reward_method="default"):
    """Write VLM-R1 JSONL format. Train + val files."""
    random.seed(seed)
    random.shuffle(samples)

    rows = []
    for i, s in enumerate(samples):
        rows.append({
            "id": s.get("sample_id") or f"sample_{i}",
            "image": s["image_path"],
            "conversations": [
                {"from": "human", "value": "<image>" + s["question"]},
                {"from": "gpt", "value": str(s["answer"])},
            ],
            "accu_reward_method": accu_reward_method,
            # extras (VLM-R1 ignores unknown keys, but useful for debugging)
            "_source": s.get("source", ""),
            "_p_hat": s.get("p_hat", -1.0),
        })

    n_val = max(int(len(rows) * val_ratio), 10)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Train: {output_path} ({len(train_rows)} rows)")

    val_path = output_path.replace(".jsonl", "_val.jsonl")
    with open(val_path, "w") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Val:   {val_path} ({len(val_rows)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Build ChartQA GRPO data for VLM-R1 (JSONL format)")
    parser.add_argument("--traj_files", nargs="+", required=True)
    parser.add_argument("--original_data", required=True,
                        help="Original ChartQA train JSON (train_augmented.json)")
    parser.add_argument("--png_dir", required=True,
                        help="Directory with original chart PNGs")
    parser.add_argument("--n_original", type=int, default=2200)
    parser.add_argument("--output", required=True,
                        help="Output JSONL path (e.g. chartqa_solver_grpo_vlmr1.jsonl)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accu_reward_method", default="default",
                        choices=["default", "mcq", "yes_no", "llm", "math"],
                        help="VLM-R1 reward method tag for each sample")
    args = parser.parse_args()

    # Expand glob
    traj_files = []
    for pattern in args.traj_files:
        if '*' in pattern:
            traj_files.extend(sorted(glob.glob(pattern)))
        else:
            traj_files.append(pattern)

    print(f"{'='*60}")
    print(f"Building ChartQA GRPO data for VLM-R1 (JSONL)")
    print(f"{'='*60}")
    print(f"  trajectory files: {len(traj_files)}")
    for tf in traj_files[:5]:
        print(f"    {tf}")
    if len(traj_files) > 5:
        print(f"    ... and {len(traj_files)-5} more")

    # (a) GenEnv trajectory samples
    print(f"\n(a) GenEnv trajectories:")
    genenv_samples, genenv_stats = extract_from_genenv_trajectories(traj_files)
    print(f"\n  Stats:")
    for k, v in sorted(genenv_stats.items()):
        print(f"    {k}: {v}")

    # (b) Original ChartQA samples
    print(f"\n(b) Original ChartQA ({args.n_original} samples):")
    original_samples = extract_original_samples(
        args.original_data, args.png_dir, args.n_original, args.seed)
    print(f"  Extracted: {len(original_samples)}")

    # Combine + dedup
    all_samples = genenv_samples + original_samples
    all_samples = deduplicate(all_samples)
    random.seed(args.seed)
    random.shuffle(all_samples)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    sources = Counter(s["source"] for s in all_samples)
    print(f"\n  Total: {len(all_samples)}")
    for src, cnt in sources.most_common():
        print(f"    {src}: {cnt} ({cnt/len(all_samples)*100:.0f}%)")

    genenv_only = [s for s in all_samples if s["source"] == "genenv"]
    if genenv_only:
        phats = [s["p_hat"] for s in genenv_only
                 if isinstance(s.get("p_hat"), (int, float)) and s["p_hat"] >= 0]
        if phats:
            sweet = sum(1 for p in phats if 0.35 <= p <= 0.65)
            print(f"\n  GenEnv p_hat: sweet={sweet}/{len(phats)} "
                  f"({sweet/len(phats)*100:.0f}%), mean={sum(phats)/len(phats):.3f}")

    # Save
    print(f"\n{'='*60}")
    print(f"Saving (VLM-R1 JSONL format)")
    print(f"{'='*60}")
    to_vlmr1_jsonl(all_samples, args.output,
                   accu_reward_method=args.accu_reward_method,
                   seed=args.seed)


if __name__ == "__main__":
    main()
