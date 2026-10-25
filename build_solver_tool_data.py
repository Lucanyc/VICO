#!/usr/bin/env python3
"""
Build GRPO training data for tool-agent Solver on VICO.

Samples 2000-3000 from VICO train_augmented (excluding rewriter 2200),
adds tool-agent system prompt, outputs parquet for verl GRPO.

Usage:
    python build_solver_tool_data.py \
        --n 2500 --output vico_tool_agent_grpo.parquet
"""

import os, json, argparse, random
from collections import Counter
from datasets import Dataset

TOOL_AGENT_SYSTEM_PROMPT = """You are a tool-augmented VLM. For every query you MUST:

1. Think about which tool to use inside <think></think>.
2. Call exactly one tool.
3. Select the tool based on what the image contains and what the question asks:
   - Need to extract structured data from a chart, table, or plot → chartmoe
   - Need to detect, locate, or count objects in a scene → sam3_detect
   - Need to read text, numbers, or labels from a document or figure → easyocr_tool
   - Need to compute a math expression → calculator
   - Need to formalize a geometry diagram → diagramformalizer

Call format:
   <tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>
   <tool_call>{"tool": "sam3_detect", "task": "detect_objects"}</tool_call>
   <tool_call>{"tool": "easyocr_tool", "task": "read_text"}</tool_call>
   <tool_call>{"tool": "calculator", "expression": "python_expr"}</tool_call>
   <tool_call>{"tool": "diagramformalizer", "task": "formalize"}</tool_call>

4. After the tool result returns in <observation></observation>, use it as a structured starting point. Combine the tool output with your own visual understanding of the image to reason step by step inside <think></think>. The tool output may be partial or imperfect — use your visual understanding to fill in gaps and verify details. Then give the final answer inside <answer></answer>."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="/path/to/dataset/vico/train/train_augmented.json")
    parser.add_argument("--rewriter_json", default="/path/to/dataset/vico/train/vico_rewriter_2200.json")
    parser.add_argument("--png_dir", default="/path/to/dataset/vico/train/png")
    parser.add_argument("--n", type=int, default=2500)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all data
    with open(args.train_json) as f:
        all_data = json.load(f)
    print(f"Total train_augmented: {len(all_data)}")

    # Load rewriter data to exclude
    exclude_keys = set()
    if os.path.exists(args.rewriter_json):
        with open(args.rewriter_json) as f:
            rewriter_data = json.load(f)
        for d in rewriter_data:
            exclude_keys.add(f"{d['imgname']}_{d['query']}")
        print(f"Excluding {len(exclude_keys)} rewriter samples")

    # Filter: exists in png_dir + not in rewriter
    pool = []
    for d in all_data:
        key = f"{d['imgname']}_{d['query']}"
        if key in exclude_keys:
            continue
        img_path = os.path.join(args.png_dir, d["imgname"])
        if os.path.exists(img_path):
            pool.append(d)

    print(f"Available pool: {len(pool)}")

    # Sample
    random.shuffle(pool)
    samples = pool[:args.n]
    print(f"Sampled: {len(samples)}")

    # Build parquet rows — use native Python objects, NOT json.dumps()
    # datasets.Dataset.from_list() preserves nested structures correctly
    rows = []
    for s in samples:
        img_path = os.path.join(args.png_dir, s["imgname"])
        question = s["query"]
        answer = str(s["label"])

        row = {
            "prompt": [
                {"role": "system", "content": TOOL_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>{question}"},
            ],
            "images": [{"image": img_path}],
            "reward_model": {"ground_truth": answer},
            "extra_info": {
                "question": question,
                "answer": answer,
                "source": "vico_tool_agent",
                "sample_id": f"vico_{s['imgname']}",
            },
        }
        rows.append(row)

    # Split train/val
    random.shuffle(rows)
    n_val = max(10, len(rows) // 20)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    train_path = args.output
    val_path = args.output.replace(".parquet", "_val.parquet")

    os.makedirs(os.path.dirname(train_path) or ".", exist_ok=True)
    Dataset.from_list(train_rows).to_parquet(train_path)
    Dataset.from_list(val_rows).to_parquet(val_path)

    # Stats
    print(f"\nOutput:")
    print(f"  Train: {train_path} ({len(train_rows)} samples)")
    print(f"  Val:   {val_path} ({len(val_rows)} samples)")

    # Verify format matches working parquets
    import pandas as pd
    check = pd.read_parquet(train_path)
    print(f"\nVerification:")
    print(f"  Columns: {list(check.columns)}")
    p = check['prompt'].iloc[0]
    print(f"  prompt type: {type(p)}")
    if hasattr(p, '__len__') and len(p) > 0:
        print(f"  prompt[0] type: {type(p[0])}")
        if isinstance(p[0], dict):
            print(f"  prompt[0] keys: {list(p[0].keys())}")
            print(f"  System prompt starts with: {p[0]['content'][:80]}...")
            print(f"  Has tool instructions: {'tool_call' in p[0]['content']}")
        else:
            print(f"  ERROR: prompt[0] is not a dict! Got: {type(p[0])}")
    else:
        print(f"  ERROR: prompt is empty or not iterable")


if __name__ == "__main__":
    main()
