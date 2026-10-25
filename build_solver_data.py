#!/usr/bin/env python3
"""
Build Solver GRPO Training Data from VICO GenEnv Trajectories

Same approach as build_solver_grpo_data_genenv.py (SCENE), adapted for VICO:
  - Uses chart images (modified_image from trajectories)
  - Answer from new_answer (programmatically computed)
  - System prompt for chart QA

Usage:
    python3 build_solver_data.py \
        --traj_files vico_genenv_gpt5/trajectories_*.json \
        --original_data /path/to/vico/train/train_augmented.json \
        --png_dir /path/to/vico/train/png \
        --n_original 300 \
        --output vico_solver_grpo.parquet
"""

import os, json, re, argparse, random, glob
from collections import Counter
from typing import List
from PIL import Image
from datasets import Dataset


SOLVER_SYSTEM_PROMPT = (
    "Please answer the question about the given image. "
    "The image may contain charts, graphs, geometric figures, or mathematical diagrams. "
    "You FIRST think about the reasoning process as an internal monologue "
    "with your own reasoning, and then provide the final answer. "
    "Think step by step in <think> tags, then give your final answer in <answer> tags."
)


def verify_image(path):
    if not path or not os.path.exists(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False


def build_prompt(question, system_prompt):
    """Build prompt in verl format. content must be STRING."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "<image>" + question},
    ]


def extract_from_genenv_trajectories(traj_files):
    """
    Extract samples from VICO GenEnv trajectories.
    Each successful turn becomes one training sample.
    Uses ALL samples (following GenEnv paper).
    """
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

            # For already_good: use original image + original answer
            if status == "already_good":
                # Find original image path
                imgname = traj.get("imgname", "")
                img_path = traj.get("final_image_path", "")
                if img_path and verify_image(img_path):
                    samples.append({
                        "image_path": img_path,
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

            # Extract from each turn
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
                    "image_path": turn_image,
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
    """Extract random samples from original VICO training data."""
    with open(original_data_path) as f:
        data = json.load(f)

    random.seed(seed)
    valid = []
    for s in data:
        img_path = os.path.join(png_dir, s["imgname"])
        if os.path.exists(img_path):
            valid.append((s, img_path))

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
    """Deduplicate GenEnv samples by (image_path, question). Keep original data untouched."""
    # Keep ALL original samples, only dedup GenEnv samples among themselves
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


def samples_to_parquet(samples, output_path):
    """Convert samples to verl GRPO parquet format."""
    rows = []
    for s in samples:
        prompt = build_prompt(s["question"], SOLVER_SYSTEM_PROMPT)
        row = {
            "prompt": prompt,
            "images": [{"image": s["image_path"]}],
            "reward_model": {"ground_truth": str(s["answer"])},
            "extra_info": {
                "sample_id": str(s.get("sample_id", "")),
                "source": str(s.get("source", "")),
                "p_hat": float(s["p_hat"]) if isinstance(s.get("p_hat"), (int, float)) else -1.0,
                "question": str(s["question"]),
                "answer": str(s["answer"]),
            },
        }
        rows.append(row)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ds = Dataset.from_list(rows)
    ds.to_parquet(output_path)
    print(f"  Saved: {output_path} ({len(ds)} rows)")

    # Also save val split
    val_n = max(int(len(rows) * 0.05), 10)
    random.shuffle(rows)
    val_rows = rows[:val_n]
    val_path = output_path.replace(".parquet", "_val.parquet")
    ds_val = Dataset.from_list(val_rows)
    ds_val.to_parquet(val_path)
    print(f"  Val: {val_path} ({len(ds_val)} rows)")

    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Build VICO GRPO data from GenEnv trajectories")
    parser.add_argument("--traj_files", nargs="+", required=True)
    parser.add_argument("--original_data", required=True,
                        help="Original VICO train JSON (train_augmented.json)")
    parser.add_argument("--png_dir", required=True,
                        help="Directory with original chart PNGs")
    parser.add_argument("--n_original", type=int, default=300)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Expand glob
    traj_files = []
    for pattern in args.traj_files:
        if '*' in pattern:
            traj_files.extend(sorted(glob.glob(pattern)))
        else:
            traj_files.append(pattern)

    print(f"{'='*60}")
    print(f"Building VICO GRPO data (GenEnv mode)")
    print(f"{'='*60}")

    # (a) GenEnv trajectory samples
    print(f"\n(a) GenEnv trajectories:")
    genenv_samples, genenv_stats = extract_from_genenv_trajectories(traj_files)
    print(f"\n  Stats:")
    for k, v in sorted(genenv_stats.items()):
        print(f"    {k}: {v}")

    # (b) Original VICO samples
    print(f"\n(b) Original VICO ({args.n_original} samples):")
    original_samples = extract_original_samples(
        args.original_data, args.png_dir, args.n_original, args.seed)
    print(f"  Extracted: {len(original_samples)}")

    # Combine
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
        phats = [s["p_hat"] for s in genenv_only if isinstance(s.get("p_hat"), (int, float)) and s["p_hat"] >= 0]
        sweet = sum(1 for p in phats if 0.35 <= p <= 0.65)
        print(f"\n  GenEnv p_hat: sweet={sweet}/{len(phats)} ({sweet/len(phats)*100:.0f}%), mean={sum(phats)/len(phats):.3f}")

    # Save
    print(f"\n{'='*60}")
    print(f"Saving")
    print(f"{'='*60}")
    samples_to_parquet(all_samples, args.output)


if __name__ == "__main__":
    main()
