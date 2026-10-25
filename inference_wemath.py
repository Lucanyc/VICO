#!/usr/bin/env python3
"""
We-Math testmini Inference for a SCENE-trained Solver (vLLM).

Target model (example):
    /path/to/VICO/round_3/solver_genenv_merged_step210

That model was GRPO-trained with a SCENE-specific SYSTEM prompt. We-Math
images are math problem diagrams (geometric figures, number lines, tables,
etc.) — NOT SCENE scenes.

Adaptation mirrors the MathVerse one (inference_mathverse_
scene_trained_model.py):
  • Keep the sentence register + the <think>/<answer> output contract
    (that's what the solver's RL training locked in).
  • Swap the image-description clause to match We-Math content.
  • We-Math is entirely multiple-choice with options A–E (E is often
    'No correct answer'), so we only run the multi-choice extractor.

Answer extraction priority:
  1. <answer>...</answer>
  2. \\boxed{...}
  3. "the answer is X" / "option X" / "letter X" in post-</think> region
  4. Last letter A–E in post-</think> region
  5. Last letter A–E in the full response

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 inference_wemath.py \\
        --model /path/to/VICO/round_3/solver_genenv_merged_step210 \\
        --test_file /path/to/dataset/We-Math/testmini_output/testmini.json \\
        --output eval_scene_genenv_step210_on_wemath.json \\
        --batch_size 64 --temperature 0
"""

import argparse
import json
import os
import re
import sys
import time


# =============================================================================
# System prompt — We-Math-adapted (keeps <think>/<answer> output contract)
# =============================================================================

WEMATH_SOLVER_SYSTEM_PROMPT = (
    "Please answer the question about the given image. "
    "The image shows a mathematical diagram such as a geometric figure, "
    "number line, measurement diagram, or labeled schematic. "
    "Think step by step in <think> tags, then give your final answer "
    "in <answer> tags. The question is multiple-choice; the answer "
    "must be a single option letter (A, B, C, D, or E)."
)


# =============================================================================
# Answer parsing (multi-choice only)
# =============================================================================

def _norm_letter(s):
    if s is None:
        return None
    s = str(s).strip().strip("()[]{}.,;:").strip()
    if not s:
        return None
    m = re.search(r"\b([A-Za-z])\b", s)
    return m.group(1).upper() if m else None


def _visible_region(text):
    """Region after </think>; the solver emits reasoning inside <think>...</think>."""
    return text.split("</think>", 1)[1] if "</think>" in text else text


def extract_answer_multichoice(text, valid_letters=("A", "B", "C", "D", "E")):
    if not text:
        return None
    valid = set(v.upper() for v in valid_letters)

    # 1) <answer>...</answer>
    answers = re.findall(
        r"<answer>\s*([^<]+?)\s*</answer>",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    for raw in reversed(answers):
        L = _norm_letter(raw)
        if L and L in valid:
            return L

    # 2) \\boxed{...}
    m = re.search(r"\\?boxed\s*\{\s*([^}]+?)\s*\}", text)
    if m:
        L = _norm_letter(m.group(1))
        if L and L in valid:
            return L

    visible = _visible_region(text)

    # 3) phrasing heuristics in the visible region
    for pat in [
        r"(?:the\s+answer\s+is|answer\s*:|final\s+answer\s*(?:is)?\s*:?)\s*"
        r"[\(\[]?([A-Ea-e])[\)\]\.\s]",
        r"(?:option|letter|choice)\s*[\(\[]?([A-Ea-e])[\)\]\.\s]",
    ]:
        m = re.search(pat, visible)
        if m:
            L = m.group(1).upper()
            if L in valid:
                return L

    # 4) last A–E letter in the visible region
    letters = [L.upper() for L in re.findall(r"\b([A-Ea-e])\b", visible)
               if L.upper() in valid]
    if letters:
        return letters[-1]

    # 5) last A–E letter anywhere
    letters = [L.upper() for L in re.findall(r"\b([A-Ea-e])\b", text)
               if L.upper() in valid]
    return letters[-1] if letters else None


# =============================================================================
# Data loading + prompt building
# =============================================================================

def load_test_data(test_file):
    try:
        with open(test_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    out = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def resolve_image_path(sample, images_root):
    img = sample.get("image") or sample.get("image_path") or ""
    if isinstance(img, dict):
        img = img.get("path") or img.get("url") or ""
    if not img:
        return None
    if os.path.isabs(img) and os.path.exists(img):
        return img
    cand = os.path.join(images_root, img)
    if os.path.exists(cand):
        return cand
    alt = os.path.join(os.path.dirname(images_root.rstrip("/")), img)
    if os.path.exists(alt):
        return alt
    return None


def format_options(option_raw):
    """'A. 300, 400;B. 400, 300;C. ...' -> bullet list."""
    if not option_raw:
        return ""
    parts = [p.strip() for p in str(option_raw).split(";") if p.strip()]
    return "\n".join(["Choices:"] + parts)


def build_user_text(sample):
    question = (sample.get("question") or "").strip()
    choices = format_options(sample.get("option") or "")
    parts = [f"Question: {question}"]
    if choices:
        parts.append(choices)
    return "\n".join(parts)


# =============================================================================
# Stats
# =============================================================================

def print_table(stats, name_col="Group", min_rows=2, top=None):
    if len(stats) < min_rows:
        return
    rows = sorted(stats.items(), key=lambda x: -x[1]["total"])
    if top is not None:
        rows = rows[:top]
    print(f"    {name_col:<60s} {'Acc':>7s}  {'Correct':>9s} / {'Total':>5s}")
    print(f"    {'-' * 90}")
    for t, s in rows:
        t_acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"    {str(t)[:60]:<60s} {t_acc:>6.1f}%  "
              f"{s['correct']:>9d} / {s['total']:>5d}")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="We-Math testmini evaluation for SCENE-trained solvers (vLLM)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--images_root", type=str, default=None,
                    help="defaults to the dir of --test_file")
    ap.add_argument("--output", default="eval_scene_genenv_step210_on_wemath.json")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--allowed_local_media_path", type=str, default=None)
    ap.add_argument("--show_examples", type=int, default=3)
    ap.add_argument("--save_full_response", action="store_true")
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--max_image_pixels", type=int, default=1280 * 28 * 28)
    args = ap.parse_args()

    if args.images_root is None:
        args.images_root = os.path.dirname(os.path.abspath(args.test_file))
    if args.allowed_local_media_path is None:
        default_root = "/path/to"
        try:
            args.allowed_local_media_path = os.path.realpath(default_root)
        except Exception:
            args.allowed_local_media_path = default_root

    print("=" * 60)
    print("We-Math testmini Evaluation (SCENE-trained solver, vLLM)")
    print(f"  Model:              {args.model}")
    print(f"  Test:               {args.test_file}")
    print(f"  Images root:        {args.images_root}")
    print(f"  Allowed media path: {args.allowed_local_media_path}")
    print(f"  Temperature:        {args.temperature}")
    print(f"  Max model len:      {args.max_model_len}")
    print(f"  Max image pixels:   {args.max_image_pixels}")
    print(f"  SYSTEM prompt:      We-Math-adapted "
          f"('math diagram ... multiple choice A–E')")
    print("=" * 60)

    from vllm import LLM, SamplingParams

    print("\nLoading vLLM engine...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=args.allowed_local_media_path,
        mm_processor_kwargs={
            "min_pixels": 4 * 28 * 28,
            "max_pixels": args.max_image_pixels,
        },
    )
    print("  Engine ready.")

    sp = (SamplingParams(max_tokens=args.max_new_tokens, temperature=0)
          if args.temperature == 0
          else SamplingParams(max_tokens=args.max_new_tokens,
                              temperature=args.temperature, top_p=0.95))

    samples = load_test_data(args.test_file)
    print(f"  Loaded {len(samples)} samples")
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    valid = []
    skipped_image = 0
    skipped_other = 0
    for s in samples:
        if not (s.get("question") or "").strip():
            skipped_other += 1
            continue
        img_path = resolve_image_path(s, args.images_root)
        if not img_path:
            skipped_image += 1
            continue
        s["_img_path"] = img_path
        s["_user_text"] = build_user_text(s)
        # stable id field
        s["_id"] = s.get("id") or s.get("ID") or s.get("key") or ""
        valid.append(s)

    print(f"  Valid: {len(valid)} "
          f"(skipped: {skipped_image} no-image, {skipped_other} no-question)")
    if not valid:
        sys.exit(1)

    t0 = time.time()
    results = []
    correct = 0
    total = 0
    parse_fail = 0
    split_stats = {}      # 2steps / 3steps
    kc_stats = {}         # knowledge concept

    for bs in range(0, len(valid), args.batch_size):
        batch = valid[bs:bs + args.batch_size]
        messages_list = []
        for s in batch:
            messages_list.append([
                {"role": "system", "content": WEMATH_SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"file://{s['_img_path']}"}},
                    {"type": "text", "text": s["_user_text"]},
                ]},
            ])

        outputs = llm.chat(messages_list, sampling_params=sp)

        batch_correct = 0
        for i, (s, out) in enumerate(zip(batch, outputs)):
            response = out.outputs[0].text
            gt = str(s.get("answer", "")).strip().upper()
            pred = extract_answer_multichoice(response)
            is_correct = (pred is not None) and (pred == gt)

            if pred is None:
                parse_fail += 1
            if is_correct:
                correct += 1
                batch_correct += 1
            total += 1

            split_tag = (s.get("image") or s.get("image_path") or "").split("/", 1)[0]
            kc = s.get("knowledge concept") or "unknown"
            for bucket, key in [(split_stats, split_tag), (kc_stats, kc)]:
                d = bucket.setdefault(key, {"correct": 0, "total": 0})
                d["total"] += 1
                if is_correct:
                    d["correct"] += 1

            results.append({
                "id": s["_id"],
                "knowledge_concept": kc,
                "split_tag": split_tag,
                "answer": gt,
                "prediction": pred,
                "correct": is_correct,
                "response": response if args.save_full_response else response[:800],
            })
            if i < args.show_examples:
                icon = "OK" if is_correct else "XX"
                qsnip = s["_user_text"].split("\n")[0][:80]
                print(f"    [{icon}] {qsnip}")
                print(f"         Pred: {pred}  |  GT: {gt}")
                snippet = response.replace("\n", " ")[:220]
                print(f"         Response: {snippet}...")

        elapsed = time.time() - t0
        speed = total / elapsed if elapsed > 0 else 0
        acc = correct / total * 100 if total else 0
        batch_acc = batch_correct / len(batch) * 100 if batch else 0
        print(f"\n  [{total}/{len(valid)}] "
              f"batch_acc={batch_acc:.1f}%  "
              f"running_acc={acc:.1f}% ({correct}/{total})  "
              f"parse_fail={parse_fail}  speed={speed:.1f} s/s")
        print("  Per split:")
        print_table(split_stats, "split", min_rows=1)

    elapsed = time.time() - t0
    accuracy = correct / total * 100 if total else 0

    print(f"{'=' * 60}\nFinal Results\n{'=' * 60}")
    print(f"  Model:      {args.model}")
    print(f"  Total:      {total}")
    print(f"  Correct:    {correct}")
    print(f"  Accuracy:   {accuracy:.2f}%")
    print(f"  Parse fail: {parse_fail}")
    print(f"  Skipped:    {skipped_image} no-image, {skipped_other} no-question")
    print(f"  Time:       {elapsed:.1f}s ({total/max(elapsed,1e-6):.1f} s/s)")
    print(f"\n  Per split (2steps/3steps):")
    print_table(split_stats, "split", min_rows=1)
    print(f"  Per knowledge concept (top 20):")
    print_table(kc_stats, "knowledge concept", min_rows=1, top=20)

    output_data = {
        "model": args.model,
        "test_file": args.test_file,
        "system_prompt": WEMATH_SOLVER_SYSTEM_PROMPT,
        "prompt_style": "wemath-adapted",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "parse_failures": parse_fail,
        "skipped_image": skipped_image,
        "skipped_other": skipped_other,
        "elapsed_seconds": elapsed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "max_image_pixels": args.max_image_pixels,
        "split_stats": split_stats,
        "knowledge_concept_stats": kc_stats,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
