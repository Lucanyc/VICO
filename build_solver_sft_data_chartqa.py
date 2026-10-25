#!/usr/bin/env python3
"""
Build ChartQA Solver SFT Data from Qwen3-VL-235B-Thinking generated reasoning chains.

Reads raw files from generate_solver_sft_data_chartqa_qwen3vl235b_api.py, filters for
correct + long-enough reasoning, and outputs LlamaFactory-compatible format.

Response format: <think>...</think>\n<answer>X</answer>

Usage:
    python3 build_solver_sft_data_chartqa.py \
        --raw_files chartqa_solver_sft_235b/solver_sft_raw_*.json \
        --output_dir chartqa_solver_sft_235b \
        --min_think_words 200
"""

import os, json, argparse, random, glob
from collections import Counter


SOLVER_SYSTEM_PROMPT = (
    "Please answer the question about the given image. "
    "The image may contain charts, graphs, geometric figures, or mathematical diagrams. "
    "You FIRST think about the reasoning process as an internal monologue "
    "with your own reasoning, and then provide the final answer. "
    "Think step by step in <think> tags, then give your final answer in <answer> tags."
)


def build_sft_sample(item):
    """Convert a raw item into a LlamaFactory SFT sample."""
    question = item.get("question", "")
    image_path = item.get("image_path", "")
    think = item.get("think", "")

    # Use GT answer (item already filtered for correctness, so predicted == GT)
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
        "word_count": item.get("word_count", 0),
        "source": "qwen3vl235b_thinking_chartqa",
    }

    return sample, meta


def main():
    parser = argparse.ArgumentParser(
        description="Build ChartQA Solver SFT data from Qwen3-VL-235B-Thinking reasoning chains")
    parser.add_argument("--raw_files", nargs="+", required=True,
                        help="Raw JSON files from generate_solver_sft_data_chartqa_qwen3vl235b_api.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_think_words", type=int, default=200,
                        help="Minimum words in <think> to accept")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load raw files ──
    print("=" * 60)
    print("Loading raw files")
    print("=" * 60)

    all_items = []
    for pattern in args.raw_files:
        for f in sorted(glob.glob(pattern)):
            data = json.load(open(f))
            fname = os.path.basename(f)
            if not data:
                print(f"  {fname}: empty")
                continue
            id_range = f"{data[0]['id']}-{data[-1]['id']}"
            print(f"  {fname}: {len(data)} entries (id {id_range})")
            all_items.extend(data)

    # Deduplicate by id
    seen = {}
    for item in all_items:
        item_id = item.get("id", "")
        if item_id not in seen:
            seen[item_id] = item
    all_items = list(seen.values())
    print(f"\nTotal unique entries: {len(all_items)}")

    # ── Filter ──
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

    # ── Build SFT samples ──
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
          f"median={np.median(twcs):.0f}, "
          f"min={min(twcs)}, max={max(twcs)}")

    # ── Save ──
    print(f"\n{'=' * 60}")
    print("Saving")
    print("=" * 60)

    random.shuffle(samples)

    train_path = os.path.join(args.output_dir, "solver_sft_train.json")
    with open(train_path, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"  Train: {len(samples)} (all data, no val split)")
    print(f"  Saved: {train_path}")

    # ── Verify format ──
    print(f"\n{'=' * 60}")
    print("Format verification")
    print("=" * 60)
    s = samples[0]
    response = s["conversations"][2]["value"]
    has_think = "<think>" in response and "</think>" in response
    has_answer = "<answer>" in response and "</answer>" in response
    print(f"  System prompt: {s['conversations'][0]['value'][:80]}...")
    print(f"  Response has <think>...</think>: {has_think}")
    print(f"  Response has <answer>...</answer>: {has_answer}")

    # ── Examples ──
    print(f"\n{'=' * 60}")
    print("Examples")
    print("=" * 60)

    for i, s in enumerate(samples[:3]):
        print(f"\n  --- Example {i+1} ---")
        print(f"  Image: ...{s['images'][0][-60:]}")
        print(f"  Question: {s['conversations'][1]['value'][:100]}")
        asst = s['conversations'][2]['value']
        print(f"  Response ({len(asst)} chars): {asst[:300]}...")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
