#!/usr/bin/env python3
"""
MathVerse testmini Inference for ChartQA-trained Solver (vLLM).

Alignment strategy:
  • SYSTEM prompt = ChartQA GRPO training prompt (tells model to wrap
    reasoning in <think> and final answer in <answer>).
  • USER message  = MathVerse's `query_cot` verbatim (already contains
    "conduct reasoning... provide option letter at the end" for multi-
    choice, or "conduct reasoning... answer the question" for free-form).
  • Expected output:
        <think>...reasoning...</think>
        <answer>B</answer>           (multi-choice)
    or  <answer>42.5</answer>         (free-form)

Answer parsing priority:
  1. <answer>...</answer>   ← training format
  2. "the answer is ..." / "option X" / "letter X"
  3. Last number / letter in the visible (post-</think>) region
  4. Last number / letter anywhere

Per-type scoring:
  • multi-choice: compare uppercased single letter (A-E)
  • free-form:    numeric equality with 0.5% tolerance, else string match

Known MathVerse testmini: 3940 items (2180 multi-choice, 1760 free-form).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 Trained_solver_inference_mathverse.py \\
        --model /path/to/VICO/chartqa_round3_solver_merged_step110 \\
        --test_file /path/to/dataset/MathVerse_testmini/testmini_output/testmini.json \\
        --images_root /path/to/dataset/MathVerse_testmini/testmini_output \\
        --output eval_chartqa_r3_on_mathverse.json
"""

import argparse
import json
import os
import re
import sys
import time


# =============================================================================
# System prompts — must match TRAINING exactly
# =============================================================================

# Math-tuned CoT prompt for MathVerse (replaces the chartqa training prompt).
# Keeps <think>/<answer> structure (matches RL training format) but adds
# math-specific guidance: examine figure -> identify relations -> apply theorems.
MATHVERSE_SYSTEM_PROMPT = (
    "You are solving a math problem about a geometric figure or mathematical diagram. "
    "Carefully examine the image and identify: "
    "(1) all given values (angles, side lengths, coordinates, area/volume labels), "
    "(2) geometric relationships present in the figure (parallel, perpendicular, "
    "similar, congruent, tangent, inscribed), "
    "(3) the specific quantity being asked. "
    "Then reason step by step, applying relevant principles "
    "(Pythagorean theorem, angle-sum, similar triangles, area/volume formulas, "
    "coordinate geometry, etc.). "
    "Show your reasoning inside <think>...</think> tags. "
    "Put the final answer inside <answer>...</answer> tags — "
    "either a single number / expression (e.g., 30, 5.5, 1/2, sqrt(3), pi/4) "
    "or an option letter (A, B, C, D, E)."
)


# =============================================================================
# Answer parsing
# =============================================================================

WORD_TO_NUM = {
    "zero": "0", "no": "0", "none": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}


def _strip_units(s):
    """Remove common unit / decoration suffixes."""
    s = str(s).strip()
    s = re.sub(r"[°%$,]", "", s)
    s = re.sub(
        r"\b(million|billion|trillion|thousand|percent|approximately|about|around|"
        r"degrees?|kg|km|miles?|meters?|cm|mm|inches?|feet|ft|rad|radians?|"
        r"dollars?|euros?|pounds?|yen|yuan|rupees?|usd|eur|gbp|cad|aud)\b",
        "", s, flags=re.IGNORECASE,
    )
    return s.strip()


def _norm_free_form(s):
    """Normalize a free-form answer for comparison. Returns canonical string."""
    if s is None:
        return None
    s = _strip_units(s).rstrip(".,;:").strip()
    if not s:
        return s
    low = s.lower()
    if low in WORD_TO_NUM:
        return WORD_TO_NUM[low]
    # Prefer the last numeric token if there are multiple
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if nums:
        try:
            val = float(nums[-1])
            if val == int(val):
                return str(int(val))
            return f"{val:g}"
        except Exception:
            return nums[-1]
    return s.lower().strip()


def _norm_letter(s):
    """Normalize to a single uppercase letter A-Z, or None."""
    if s is None:
        return None
    s = str(s).strip().strip("()[]{}.,;:").strip()
    if not s:
        return None
    # Find the first standalone letter token (A-E most common, support A-Z)
    m = re.search(r"\b([A-Za-z])\b", s)
    if m:
        return m.group(1).upper()
    return None


def _visible_region(text):
    """Text after the closing </think>, used for fallback searches."""
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


def extract_answer_multichoice(text, options=None):
    """Return a single uppercase letter, or None."""
    if not text:
        return None
    valid = set([o.upper() for o in (options or [])]) or set("ABCDE")

    # 1. <answer>...</answer> — take LAST occurrence, extract letter
    answers = re.findall(
        r"<answer>\s*([^<]+?)\s*</answer>",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    for raw in reversed(answers):
        letter = _norm_letter(raw)
        if letter and letter in valid:
            return letter

    # 2. \boxed{...}
    m = re.search(r"\\?boxed\s*\{\s*([^}]+?)\s*\}", text)
    if m:
        letter = _norm_letter(m.group(1))
        if letter and letter in valid:
            return letter

    # 3. Phrases: "the answer is X", "option X", "letter X", "choice X"
    visible = _visible_region(text)
    for pat in [
        r"(?:the\s+answer\s+is|answer\s*:|final\s+answer\s*(?:is)?\s*:?)\s*"
        r"[\(\[]?([A-Ea-e])[\)\]\.\s]",
        r"(?:option|letter|choice)\s*[\(\[]?([A-Ea-e])[\)\]\.\s]",
    ]:
        m = re.search(pat, visible)
        if m:
            letter = m.group(1).upper()
            if letter in valid:
                return letter

    # 4. Any standalone letter in the visible region — take the last one
    letters = re.findall(r"\b([A-Ea-e])\b", visible)
    letters = [L.upper() for L in letters if L.upper() in valid]
    if letters:
        return letters[-1]

    # 5. Any standalone letter anywhere
    letters = re.findall(r"\b([A-Ea-e])\b", text)
    letters = [L.upper() for L in letters if L.upper() in valid]
    if letters:
        return letters[-1]
    return None


def extract_answer_freeform(text):
    """Return the free-form answer string (normalized), or None."""
    if not text:
        return None

    # 1. <answer>...</answer>
    answers = re.findall(
        r"<answer>\s*([^<]+?)\s*</answer>",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    if answers:
        return _norm_free_form(answers[-1])

    # 2. \boxed{...}
    m = re.search(r"\\?boxed\s*\{\s*([^}]+?)\s*\}", text)
    if m:
        return _norm_free_form(m.group(1))

    visible = _visible_region(text)

    # 3. "the answer is ..." / "final answer: ..."
    m = re.search(
        r"(?:the\s+answer\s+is|final\s+answer\s*(?:is)?\s*:?|answer\s*:)\s*([^\n.]+)",
        visible, flags=re.IGNORECASE,
    )
    if m:
        return _norm_free_form(m.group(1))

    # 4. Last number in visible region
    nums = re.findall(r"-?\d+(?:\.\d+)?", visible)
    if nums:
        return _norm_free_form(nums[-1])

    # 5. Last number anywhere
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return _norm_free_form(nums[-1])
    return None


def freeform_is_correct(pred, gt):
    """Compare two normalized free-form answers with numeric tolerance."""
    if pred is None or gt is None:
        return False
    p = _norm_free_form(pred)
    g = _norm_free_form(gt)
    if p == g:
        return True
    try:
        pn = float(p)
        gn = float(g)
        if gn != 0 and abs(pn - gn) / abs(gn) <= 0.005:
            return True
        # Scale-equivalent (e.g., model output in different unit)
        for sc in (1e3, 1e6, 1e9):
            if gn != 0 and abs(pn - gn * sc) / abs(gn * sc) <= 0.005:
                return True
            if pn != 0 and abs(gn - pn * sc) / abs(pn * sc) <= 0.005:
                return True
    except Exception:
        pass
    return False


# =============================================================================
# Data loading
# =============================================================================

def load_test_data(test_file):
    """MathVerse ships as a JSON array; fall back to JSONL if needed."""
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
    """Resolve the sample's image path to an absolute, existing path (or None)."""
    img = sample.get("image", "")
    if isinstance(img, dict):
        img = img.get("path") or img.get("url") or ""
    if not img:
        return None
    if os.path.isabs(img) and os.path.exists(img):
        return img
    cand = os.path.join(images_root, img)
    if os.path.exists(cand):
        return cand
    # Some MathVerse dumps have image under a sibling images/ dir
    alt = os.path.join(os.path.dirname(images_root.rstrip("/")), img)
    if os.path.exists(alt):
        return alt
    return None


# =============================================================================
# Stats
# =============================================================================

def print_table(stats, name_col="Group", min_rows=2):
    if len(stats) < min_rows:
        return
    print(f"    {name_col:<30s} {'Acc':>7s}  {'Correct':>9s} / {'Total':>5s}")
    print(f"    {'-' * 60}")
    for t, s in sorted(stats.items(), key=lambda x: -x[1]["total"]):
        t_acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"    {str(t):<30s} {t_acc:>6.1f}%  {s['correct']:>9d} / {s['total']:>5d}")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MathVerse testmini evaluation for ChartQA-trained solvers (vLLM)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--images_root", type=str, default=None,
                        help=("Root directory for relative image paths. Defaults to "
                              "the directory containing --test_file."))
    parser.add_argument("--output", default="mathverse_eval_results.json")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--allowed_local_media_path", type=str, default=None)
    parser.add_argument("--max_model_len", type=int, default=16384,
                        help="Max model context length. MathVerse geometry images "
                             "can produce 7k+ image tokens, so keep this well above 4096.")
    parser.add_argument("--max_image_pixels", type=int, default=1280 * 28 * 28,
                        help="Upper bound on image tokens (pixels/patch). Default "
                             "1280*28*28 caps images to ≈1024 image tokens.")
    parser.add_argument("--show_examples", type=int, default=3)
    parser.add_argument("--filter_question_type", type=str, default="all",
                        choices=["all", "multi-choice", "free-form"])
    parser.add_argument("--save_full_response", action="store_true")
    args = parser.parse_args()

    if args.images_root is None:
        args.images_root = os.path.dirname(os.path.abspath(args.test_file))
    if args.allowed_local_media_path is None:
        default_root = "/path/to"
        try:
            args.allowed_local_media_path = os.path.realpath(default_root)
        except Exception:
            args.allowed_local_media_path = default_root

    print("=" * 60)
    print("MathVerse testmini Evaluation (vLLM)")
    print(f"  Model:              {args.model}")
    print(f"  Test:               {args.test_file}")
    print(f"  Images root:        {args.images_root}")
    print(f"  Allowed media path: {args.allowed_local_media_path}")
    print(f"  Temperature:        {args.temperature}")
    print(f"  Type filter:        {args.filter_question_type}")
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

    # Filter & resolve image paths
    valid = []
    skipped_image = 0
    skipped_type = 0
    skipped_other = 0
    for s in samples:
        qt = s.get("question_type", "unknown")
        if args.filter_question_type != "all" and qt != args.filter_question_type:
            skipped_type += 1
            continue
        query = s.get("query_cot") or s.get("question_for_eval") or s.get("question")
        if not query:
            skipped_other += 1
            continue
        img_path = resolve_image_path(s, args.images_root)
        if not img_path:
            skipped_image += 1
            continue
        s["_img_path"] = img_path
        s["_query"] = query
        valid.append(s)

    print(f"  Valid: {len(valid)} "
          f"(skipped: {skipped_image} no-image, "
          f"{skipped_type} filtered-type, {skipped_other} no-query)")
    if not valid:
        sys.exit(1)

    # Inference
    t0 = time.time()
    results = []
    correct = 0
    total = 0
    parse_fail = 0
    type_stats = {}
    subj_stats = {}
    subfield_stats = {}
    version_stats = {}

    for bs in range(0, len(valid), args.batch_size):
        batch = valid[bs:bs + args.batch_size]
        messages_list = []
        for s in batch:
            messages_list.append([
                {"role": "system", "content": MATHVERSE_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"file://{s['_img_path']}"}},
                    {"type": "text", "text": s["_query"]},
                ]},
            ])

        outputs = llm.chat(messages_list, sampling_params=sp)

        batch_correct = 0
        for i, (s, out) in enumerate(zip(batch, outputs)):
            response = out.outputs[0].text
            qt = s.get("question_type", "unknown")
            gt = str(s.get("answer", "")).strip()

            if qt == "multi-choice":
                opts = s.get("options") or ["A", "B", "C", "D"]
                pred = extract_answer_multichoice(response, options=opts)
                gt_norm = _norm_letter(gt) or gt.upper()
                is_correct = (pred is not None) and (pred == gt_norm)
            else:  # free-form (or unknown: try free-form)
                pred = extract_answer_freeform(response)
                gt_norm = _norm_free_form(gt)
                is_correct = freeform_is_correct(pred, gt_norm)

            if pred is None:
                parse_fail += 1
            if is_correct:
                correct += 1
                batch_correct += 1
            total += 1

            # Per-slice stats
            for bucket, key in [
                (type_stats, qt),
                (subj_stats, s.get("subject", "unknown")),
                (subfield_stats, s.get("subfield", "unknown")),
                (version_stats, s.get("problem_version", "unknown")),
            ]:
                d = bucket.setdefault(key, {"correct": 0, "total": 0})
                d["total"] += 1
                if is_correct:
                    d["correct"] += 1

            results.append({
                "id": s.get("id", ""),
                "question_type": qt,
                "subject": s.get("subject"),
                "subfield": s.get("subfield"),
                "problem_version": s.get("problem_version"),
                "answer": gt_norm,
                "prediction": pred,
                "correct": is_correct,
                "response": response if args.save_full_response else response[:800],
            })
            if i < args.show_examples:
                icon = "✅" if is_correct else "❌"
                qsnip = s["_query"].split("\n")[0][:80]
                print(f"    {icon} [{qt}] {qsnip}")
                print(f"       Pred: {pred}  |  GT: {gt_norm}")
                snippet = response.replace("\n", " ")[:220]
                print(f"       Response: {snippet}...")

        elapsed = time.time() - t0
        speed = total / elapsed if elapsed > 0 else 0
        acc = correct / total * 100 if total else 0
        batch_acc = batch_correct / len(batch) * 100 if batch else 0
        print(f"\n  [{total}/{len(valid)}] "
              f"batch_acc={batch_acc:.1f}%  "
              f"running_acc={acc:.1f}% ({correct}/{total})  "
              f"parse_fail={parse_fail}  speed={speed:.1f} s/s")
        print("  Per question_type:")
        print_table(type_stats, "question_type", min_rows=1)

    elapsed = time.time() - t0
    accuracy = correct / total * 100 if total else 0

    print(f"{'=' * 60}\nFinal Results\n{'=' * 60}")
    print(f"  Model:      {args.model}")
    print(f"  Total:      {total}")
    print(f"  Correct:    {correct}")
    print(f"  Accuracy:   {accuracy:.2f}%")
    print(f"  Parse fail: {parse_fail}")
    print(f"  Skipped:    {skipped_image} no-image, "
          f"{skipped_type} filtered-type, {skipped_other} no-query")
    print(f"  Time:       {elapsed:.1f}s ({total/max(elapsed,1e-6):.1f} s/s)")
    print(f"\n  Per question_type:")
    print_table(type_stats, "question_type", min_rows=1)
    print(f"  Per subject:")
    print_table(subj_stats, "subject")
    print(f"  Per subfield:")
    print_table(subfield_stats, "subfield")
    print(f"  Per problem_version:")
    print_table(version_stats, "problem_version")

    output_data = {
        "model": args.model,
        "test_file": args.test_file,
        "system_prompt": MATHVERSE_SYSTEM_PROMPT,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "parse_failures": parse_fail,
        "skipped_image": skipped_image,
        "skipped_type": skipped_type,
        "skipped_other": skipped_other,
        "elapsed_seconds": elapsed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "type_stats": type_stats,
        "subject_stats": subj_stats,
        "subfield_stats": subfield_stats,
        "version_stats": version_stats,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
