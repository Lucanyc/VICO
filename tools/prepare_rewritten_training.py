#!/usr/bin/env python3
"""Convert the rewritten-only release to portable JSONL or local verl Parquet.

Train/validation groups are connected components of original chart IDs and
generated image IDs, so edits derived from the same chart cannot cross splits.
"""
import argparse
import hashlib
import json
from pathlib import Path

SYSTEM_PROMPT = (
    "Please answer the question about the given image. "
    "The image may contain charts, graphs, geometric figures, or mathematical diagrams. "
    "You FIRST think about the reasoning process as an internal monologue "
    "with your own reasoning, and then provide the final answer. "
    "Think step by step in <think> tags, then give your final answer in <answer> tags."
)


def split_samples(samples, val_ratio, seed):
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    sample_keys = {}
    for sample in samples:
        keys = ["image:" + sample["image"]]
        keys.extend("original:" + p["original_image_id"] for p in sample.get("provenance", [])
                    if p.get("original_image_id"))
        keys = sorted(set(keys))
        for key in keys[1:]:
            left, right = find(keys[0]), find(key)
            parents[max(left, right)] = min(left, right)
        sample_keys[sample["id"]] = keys[0]
    train, val = [], []
    for sample in samples:
        group = find(sample_keys[sample["id"]])
        fraction = int(hashlib.sha256((str(seed) + ":" + group).encode()).hexdigest(), 16) / 2**256
        (val if fraction < val_ratio else train).append(sample)
    return train, val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--subset", default="all", help="Subset name, or all")
    ap.add_argument("--data-source", default="vico_rewritten")
    ap.add_argument("--system-prompt-file", type=Path)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--format", choices=["jsonl", "parquet"], default="parquet")
    args = ap.parse_args()
    if not 0 <= args.val_ratio < 1:
        ap.error("--val-ratio must be in [0, 1)")
    root = args.dataset_root.resolve()
    selected = None
    if args.subset != "all":
        selected = set((root / "subsets" / (args.subset + ".txt")).read_text().splitlines())
    samples = []
    with (root / "samples.jsonl").open() as f:
        for line in f:
            sample = json.loads(line)
            if selected is None or sample["id"] in selected:
                image = (root / sample["image"]).resolve()
                if root not in image.parents or not image.is_file():
                    raise ValueError("Missing or unsafe generated image: " + sample["image"])
                samples.append(sample)
    if not samples:
        raise ValueError("No samples selected")
    train, val = split_samples(samples, args.val_ratio, args.seed)
    if args.val_ratio and (not train or not val):
        raise ValueError("Empty grouped split; choose another seed or val ratio")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    system_prompt = args.system_prompt_file.read_text() if args.system_prompt_file else SYSTEM_PROMPT
    for name, split in [("train", train), ("val", val)]:
        if args.format == "jsonl":
            with (args.output_dir / (name + ".jsonl")).open("w") as f:
                for sample in split:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            continue
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = [{
            "data_source": args.data_source,
            "prompt": [{"role": "system", "content": system_prompt},
                       {"role": "user", "content": "<image>" + s["question"]}],
            "images": [{"image": str((root / s["image"]).resolve())}],
            "reward_model": {"ground_truth": s["answer"], "style": "rule"},
            "extra_info": {"sample_id": s["id"], "source": "rewritten_only",
                           "question": s["question"], "answer": s["answer"]},
        } for s in split]
        pq.write_table(pa.Table.from_pylist(rows), args.output_dir / (name + ".parquet"))
    print(json.dumps({"train": len(train), "val": len(val), "original_only_rows": 0,
                      "subset": args.subset, "format": args.format}))


if __name__ == "__main__":
    main()
