#!/usr/bin/env python3
"""
Build the round4 VICO GenEnv input batch.

What it does:
  1. Loads train_augmented.json (20,901 vico samples)
  2. Excludes any imgname already processed in round2 / round3 trajectory files
  3. For each remaining sample, derives (tier, answer_cell) heuristically by
     matching the gold label against cells in the associated CSV:
       - Numeric: match with 1% relative tolerance
       - Textual: case-insensitive, stripped
       - Tier 1: exactly one (row, col) matches  -> answer_cell set, chart_rerender usable
       - Tier 2: no match or multiple matches     -> answer_cell=None, answer kept as-is
  4. Keeps only Tier 1 samples (Tier 2 are harder to edit reliably),
     writes them in the same schema as vico_rewriter_2200.json.

Output schema per sample:
  {"imgname": str, "query": str, "label": str, "tier": 1, "answer_cell": {"row": int, "col": str}}

Usage:
    python3 build_seed_batch.py \
        --train_augmented /path/to/dataset/vico/train/train_augmented.json \
        --tables_dir      /path/to/dataset/vico/train/tables \
        --png_dir         /path/to/dataset/vico/train/png \
        --used_glob       "vico_genenv_round2/newprompt_trajectories_*.json" \
        --used_glob       "vico_genenv_round3/newprompt_trajectories_*.json" \
        --n_target        2500 \
        --output          /path/to/dataset/vico/train/vico_rewriter_round4_2500.json
"""

import argparse
import glob
import json
import os
import random
import sys

import pandas as pd


def load_used_imgnames(glob_patterns):
    used = set()
    for pat in glob_patterns:
        for f in sorted(glob.glob(pat)):
            try:
                with open(f) as fh:
                    data = json.load(fh)
            except Exception as e:
                print(f"  [warn] skipping {f}: {e}")
                continue
            if not isinstance(data, list):
                continue
            for traj in data:
                img = traj.get("imgname")
                if img:
                    used.add(img)
    return used


def parse_numeric(s):
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").strip().rstrip("%").strip())
    except (ValueError, TypeError):
        return None


def find_answer_cell(df: pd.DataFrame, label: str):
    """
    Locate (row, col) in df matching label.
    Skip the first column (entity/label column) — answers live in value columns.
    Returns list of (row_idx, col_name) matches.
    """
    matches = []
    label_num = parse_numeric(label)
    label_txt = str(label).strip().lower()

    value_cols = list(df.columns)[1:]
    for col in value_cols:
        series = df[col]
        for row_idx in range(len(series)):
            cell = series.iloc[row_idx]
            # Numeric match (1% relative tolerance)
            if label_num is not None:
                cell_num = parse_numeric(cell)
                if cell_num is not None:
                    denom = max(abs(label_num), abs(cell_num), 1e-9)
                    if abs(cell_num - label_num) / denom <= 0.01:
                        matches.append((row_idx, col))
                        continue
            # Textual match
            if str(cell).strip().lower() == label_txt:
                matches.append((row_idx, col))
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_augmented", required=True,
                        help="Path to train_augmented.json")
    parser.add_argument("--tables_dir", required=True,
                        help="Directory with CSV tables")
    parser.add_argument("--png_dir", required=True,
                        help="Directory with PNG charts (for existence check)")
    parser.add_argument("--used_glob", action="append", default=[],
                        help="Glob(s) of prior trajectory JSONs to exclude; can pass multiple")
    parser.add_argument("--extra_exclude", default=None,
                        help="Optional JSON with [{imgname: ...}, ...] to also exclude")
    parser.add_argument("--n_target", type=int, default=2500,
                        help="Desired number of Tier-1 samples in the output batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True,
                        help="Output JSON path (round4 rewriter input)")
    args = parser.parse_args()

    print(f"[1/4] Loading {args.train_augmented} ...")
    with open(args.train_augmented) as f:
        all_samples = json.load(f)
    print(f"     train_augmented size: {len(all_samples)}")

    # Collect used imgnames
    print(f"[2/4] Scanning used imgnames from {len(args.used_glob)} globs ...")
    used = load_used_imgnames(args.used_glob)
    if args.extra_exclude and os.path.exists(args.extra_exclude):
        with open(args.extra_exclude) as f:
            for s in json.load(f):
                if "imgname" in s:
                    used.add(s["imgname"])
    print(f"     excluded unique imgs: {len(used)}")

    # Filter unused + shuffle deterministically
    candidates = [s for s in all_samples if s["imgname"] not in used]
    print(f"     candidates remaining: {len(candidates)} / {len(all_samples)}")
    random.seed(args.seed)
    random.shuffle(candidates)

    # Derive tier/answer_cell
    print(f"[3/4] Deriving tier/answer_cell (target={args.n_target} Tier-1 samples) ...")
    output_samples = []
    stats = {
        "scanned": 0,
        "missing_csv": 0,
        "missing_png": 0,
        "csv_load_fail": 0,
        "tier1_unique_match": 0,
        "tier2_no_match": 0,
        "tier2_multi_match": 0,
    }

    for s in candidates:
        if len(output_samples) >= args.n_target:
            break
        stats["scanned"] += 1

        imgname = s["imgname"]
        basename = os.path.splitext(imgname)[0]
        csv_path = os.path.join(args.tables_dir, f"{basename}.csv")
        png_path = os.path.join(args.png_dir, imgname)

        if not os.path.exists(csv_path):
            stats["missing_csv"] += 1
            continue
        if not os.path.exists(png_path):
            stats["missing_png"] += 1
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            stats["csv_load_fail"] += 1
            continue
        if df.empty or df.shape[1] < 2:
            stats["csv_load_fail"] += 1
            continue

        matches = find_answer_cell(df, s["label"])
        if len(matches) == 0:
            stats["tier2_no_match"] += 1
            continue
        if len(matches) > 1:
            stats["tier2_multi_match"] += 1
            continue

        row_idx, col_name = matches[0]
        output_samples.append({
            "imgname": imgname,
            "query": s["query"],
            "label": str(s["label"]),
            "tier": 1,
            "answer_cell": {"row": int(row_idx), "col": col_name},
        })
        stats["tier1_unique_match"] += 1

        if stats["scanned"] % 500 == 0:
            print(f"     scanned={stats['scanned']:>5} "
                  f"tier1={stats['tier1_unique_match']:>4} "
                  f"no_match={stats['tier2_no_match']:>4} "
                  f"multi={stats['tier2_multi_match']:>4}")

    # Write output
    print(f"[4/4] Writing {len(output_samples)} samples to {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_samples, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  output: {args.output}")
    print(f"  output_size: {len(output_samples)}")

    if len(output_samples) < args.n_target:
        print(f"\n[warn] Only produced {len(output_samples)} Tier-1 samples "
              f"(target {args.n_target}). Candidate pool likely exhausted.")


if __name__ == "__main__":
    main()
