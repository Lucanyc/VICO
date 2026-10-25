#!/usr/bin/env python3
"""
Build VICO Solver SFT Data from Qwen3-VL-235B-Thinking generated reasoning chains.

Adapted from build_solver_sft_data_qwen3vl235bthinking.py (SCENE version).
System prompt aligned with VICO GRPO training and reward function.

Usage:
    python3 build_solver_sft_thinking_data.py \
        --raw_files vico_solver_sft_235b/solver_sft_raw_*.json \
        --output_dir vico_solver_sft_235b \
        --min_think_words 100
"""

import os, json, argparse, random, glob
from collections import Counter


# Must match VICO GRPO (build_solver_data.py)
SOLVER_SYSTEM_PROMPT = (
    "Please answer the question about the chart image. "
    "Think step by step in <think> tags, then give your final answer in <answer> tags."
)


def build_sft_sample(item):
    question = item.get("question", "")
    image_path = item.get("image_path", "")
    think = item.get("think", "")
    answer = str(item.get("answer", item.get("label", "")))

    assistant = f"<think>\n{think.strip()}\n</think>\n<answer>{answer}</answer>"

    sample = {
        "conversations": [
            {"from": "system", "value": SOLVER_SYSTEM_PROMPT},
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": assistant},
        ],
        "images": [os.path.abspath(image_path)],
    }

    meta = {
        "id": item.get("id", ""),
        "question": question,
        "answer": answer,
        "think_word_count": item.get("think_word_count", len(think.split())),
        "source": "qwen3vl235b_thinking_vico",
    }

    return sample, meta


def main():
    parser = argparse.ArgumentParser(
        description="Build VICO Solver SFT data from Qwen3-VL-235B reasoning chains")
    parser.add_argument("--raw_files", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_think_words", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    print("=" * 60)
    print("Loading raw files")
    print("=" * 60)

    all_items = []
    for pattern in args.raw_files:
        for f in sorted(glob.glob(pattern)):
            data = json.load(open(f))
            print(f"  {os.path.basename(f)}: {len(data)} entries")
            all_items.extend(data)

    # Dedup
    seen = {}
    for item in all_items:
        item_id = item.get("id", "")
        if item_id not in seen:
            seen[item_id] = item
    all_items = list(seen.values())
    print(f"\nTotal unique: {len(all_items)}")

    # Filter
    print(f"\n{'=' * 60}")
    print("Filtering")
    print("=" * 60)

    stats = Counter()
    qualified = []

    for item in all_items:
        stats["total"] += 1

        if not item.get("correct"):
            stats["filtered_incorrect"] += 1
            continue

        if item.get("refusal"):
            stats["filtered_refusal"] += 1
            continue

        think = item.get("think", "")
        if not think:
            stats["filtered_no_think"] += 1
            continue

        twc = item.get("think_word_count", len(think.split()))
        if twc < args.min_think_words:
            stats["filtered_short_think"] += 1
            continue

        img = item.get("image_path", "")
        if not img or not os.path.exists(img):
            stats["filtered_no_image"] += 1
            continue

        stats["qualified"] += 1
        qualified.append(item)

    print(f"  Filtering stats:")
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v}")

    if not qualified:
        print("\nERROR: No qualified samples!")
        return

    # Build SFT samples
    print(f"\n{'=' * 60}")
    print("Building SFT samples")
    print("=" * 60)

    samples = []
    metas = []
    for item in qualified:
        sample, meta = build_sft_sample(item)
        samples.append(sample)
        metas.append(meta)

    print(f"  Total SFT samples: {len(samples)}")

    twcs = [m["think_word_count"] for m in metas]
    import numpy as np
    print(f"  Think word count: mean={np.mean(twcs):.0f}, "
          f"median={np.median(twcs):.0f}, min={min(twcs)}, max={max(twcs)}")

    # Save
    print(f"\n{'=' * 60}")
    print("Saving")
    print("=" * 60)

    random.shuffle(samples)

    train_path = os.path.join(args.output_dir, "solver_sft_train.json")
    with open(train_path, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"  Train: {len(samples)} samples")
    print(f"  Saved: {train_path}")

    # Verify format
    print(f"\n{'=' * 60}")
    print("Format verification")
    print("=" * 60)
    s = samples[0]
    sys_prompt = s["conversations"][0]["value"]
    response = s["conversations"][2]["value"]
    has_think = "<think>" in response and "</think>" in response
    has_answer = "<answer>" in response and "</answer>" in response
    print(f"  System prompt matches GRPO: {sys_prompt == SOLVER_SYSTEM_PROMPT}")
    print(f"  Response has <think>+<answer>: {has_think and has_answer}")

    # Examples
    print(f"\n{'=' * 60}")
    print("Examples")
    print("=" * 60)
    for i, s in enumerate(samples[:2]):
        print(f"\n  --- Example {i+1} ---")
        print(f"  Image: ...{s['images'][0][-50:]}")
        print(f"  Q: {s['conversations'][1]['value'][:80]}")
        asst = s['conversations'][2]['value']
        print(f"  Response ({len(asst)} chars): {asst[:200]}...")


if __name__ == "__main__":
    main()
