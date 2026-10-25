#!/usr/bin/env python3
"""
Evaluate on LogicVista (logical reasoning over images).

Dataset: /path/to/dataset/LogicVista_json/
  - test.json  (448 samples)
  - images/test/<id>.png

Answer format: almost entirely single MCQ letter (A-I), with 6 multi-select
samples like "A, C" where two letters are both correct. Extraction returns a
SET of letters and compares against the GT letter set.

Prompt-style presets (`rethinker` reproduces the best-performing setup from
the ViGaL / Vision-Zero papers: CoT + <answer>LETTER</answer> tag):
  - boxed     : base-model style, extract from \\boxed{}
  - answer_tag: FT-model style with <think>/<answer> system prompt
  - rethinker : explicit "reason step by step, final answer in <answer>" hint

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python3 eval_logicvista.py \\
        --model_path /path/to/models/Qwen3-VL-8B-Instruct \\
        --tp 2 --prompt_style rethinker \\
        --output eval_logicvista_qwen3vl_base_results.json
"""

import os, json, re, argparse, time
from vllm import LLM, SamplingParams
from tqdm import tqdm
from collections import defaultdict

from eval_mathvista import parse_mcq, extract_boxed, extract_answer_tag


DEFAULT_DATA_ROOT = "/path/to/dataset/LogicVista_json"
DEFAULT_TEST_JSON = os.path.join(DEFAULT_DATA_ROOT, "test.json")


# ══════════════════════════════════════════════════════════════════════
#  Letter-set extraction (handles both single "C" and multi "A, C")
# ══════════════════════════════════════════════════════════════════════

def extract_letter_set(text):
    """Return a set of uppercase A-I letters found in `text`, or empty set."""
    if not text:
        return set()
    # Pull any standalone letter token (word boundary) plus parenthesised "(A)"
    raw = re.findall(r'\b([A-Ia-i])\b|\(([A-Ia-i])\)', text)
    letters = {m[0] or m[1] for m in raw}
    return {c.upper() for c in letters if c}


def extract_answer(text, prompt_style="rethinker"):
    """
    Pull the answer container (<answer> / \\boxed{}) first, then reduce to a
    letter-set. `answer_tag` and `rethinker` prompts ask the model to emit
    <answer>...</answer>, so we try that first under those styles.
    """
    if prompt_style in ("answer_tag", "rethinker"):
        container = extract_answer_tag(text) or extract_boxed(text)
    else:
        container = extract_boxed(text) or extract_answer_tag(text)

    payload = container if container else text

    # Primary path: parse_mcq returns a single letter if confidently identifiable.
    letter = parse_mcq(payload) if container else ""

    # Secondary path: collect ALL A-I letters in the container (handles "A, C").
    letters = extract_letter_set(payload)

    # Prefer multi-select if the container clearly lists multiple letters,
    # else fall back to parse_mcq's single pick.
    if len(letters) >= 2:
        return letters
    if letter:
        return {letter}
    if letters:
        return letters
    return set()


def normalize_gt(ans):
    """GT like 'C' or 'A, C' -> set of uppercase letters."""
    return extract_letter_set(str(ans))


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def atomic_save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_json", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image_root", default=DEFAULT_DATA_ROOT,
                        help="Root dir; image_path in JSON is joined under here.")
    parser.add_argument("--output", default="eval_logicvista_results.json")
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--prompt_style", default="rethinker",
                        choices=["boxed", "answer_tag", "rethinker"])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Resume from partial --output if it exists (default: True).")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Atomic-save the running summary every N batches.")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    with open(args.test_json) as f:
        data = json.load(f)
    if args.n > 0:
        data = data[:args.n]
    print(f"Loaded {len(data)} samples from {args.test_json}")

    # Resolve + verify image paths
    samples = []
    for s in data:
        img_path = os.path.join(args.image_root, s["image_path"])
        if not os.path.exists(img_path):
            continue
        samples.append({
            "id": s["id"],
            "question": s["question"],
            "answer": s["answer"],
            "skill": s.get("skill", []),
            "broad_capability": s.get("broad_capability", []),
            "specific_capability": s.get("specific_capability", []),
            "image_path": img_path,
        })
    print(f"Valid samples: {len(samples)}")

    # ── Resume ───────────────────────────────────────────────────────────
    prior_results, done_ids = [], set()
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                prev = json.load(f)
            prior_results = prev.get("results", [])
            done_ids = {r["id"] for r in prior_results if r.get("id") is not None}
            print(f"[resume] Loaded {len(prior_results)} existing results from {args.output}")
        except Exception as e:
            print(f"[resume] Could not parse existing output ({e}); starting fresh")

    remaining = [s for s in samples if s["id"] not in done_ids]
    print(f"To process: {len(remaining)}/{len(samples)} ({len(done_ids)} already done)")

    if not remaining:
        print("[resume] All samples already evaluated; regenerating summary only.")
        _write_summary(args, prior_results, elapsed=0.0)
        return

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model: {args.model_path}")
    allowed = os.path.realpath(args.image_root)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=allowed,
    )
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)

    # Pre-resize oversized images (mirrors eval_mathvista policy).
    from PIL import Image as PILImage
    MAX_PIXELS = 1536 * 1536
    resized_dir = os.path.join(args.image_root, "_resized_tmp")
    os.makedirs(resized_dir, exist_ok=True)
    resized = 0
    for s in remaining:
        try:
            with PILImage.open(s["image_path"]) as img:
                w, h = img.size
                if w * h > MAX_PIXELS:
                    scale = (MAX_PIXELS / (w * h)) ** 0.5
                    new_p = os.path.join(resized_dir, os.path.basename(s["image_path"]))
                    img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS).save(new_p)
                    s["image_path"] = new_p
                    resized += 1
        except Exception:
            pass
    if resized:
        print(f"Resized {resized} oversize images (> {MAX_PIXELS} pixels)")

    # ── Inference ────────────────────────────────────────────────────────
    results = list(prior_results)
    correct = sum(int(r.get("correct", False)) for r in results)
    total = len(results)
    t0 = time.time()

    for bi, i in enumerate(tqdm(range(0, len(remaining), args.batch_size), desc="Evaluating")):
        batch = remaining[i:i + args.batch_size]
        messages_batch = []
        for s in batch:
            query = s["question"]

            if args.prompt_style == "rethinker":
                # ViGaL / Vision-Zero-style CoT prompt — best observed on FT models.
                query = (query +
                         "\n\nPlease reason step by step, and put your final answer "
                         "(the option letter, e.g., A, B, C, D; or 'A, C' if two are correct) "
                         "within <answer> </answer> tags.")
                sys_prompt = None
            elif args.prompt_style == "answer_tag":
                sys_prompt = (
                    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
                    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. "
                    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
                )
            else:  # boxed
                query = (query +
                         "\n\nPlease answer the question and provide the correct option letter "
                         "(e.g., A, B, C, D) within \\boxed{}.")
                sys_prompt = None

            content = [
                {"type": "image_url",
                 "image_url": {"url": f"file://{os.path.realpath(s['image_path'])}"}},
                {"type": "text", "text": query},
            ]
            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": content})
            messages_batch.append(messages)

        outputs = llm.chat(messages_batch, sampling_params=sampling_params)

        for s, out in zip(batch, outputs):
            resp = out.outputs[0].text
            pred_set = extract_answer(resp, args.prompt_style)
            gt_set = normalize_gt(s["answer"])
            match = (len(pred_set) > 0) and (pred_set == gt_set)
            correct += int(match)
            total += 1
            results.append({
                "id": s["id"],
                "question": s["question"],
                "skill": s.get("skill", []),
                "broad_capability": s.get("broad_capability", []),
                "gt": str(s["answer"]),
                "gt_set": sorted(gt_set),
                "pred_set": sorted(pred_set),
                "correct": match,
                "response": resp,
            })
        tqdm.write(f"  [{total}/{len(samples)}] acc={correct/total*100:.1f}%")

        if ((bi + 1) % args.save_every) == 0:
            _write_summary(args, results, elapsed=time.time() - t0, verbose=False)

    elapsed = time.time() - t0
    _write_summary(args, results, elapsed=elapsed, verbose=True)


def _compute_metrics(results):
    total = len(results)
    correct = sum(int(r.get("correct", False)) for r in results)
    skill_c = defaultdict(int)
    skill_t = defaultdict(int)
    cap_c = defaultdict(int)
    cap_t = defaultdict(int)
    for r in results:
        for sk in r.get("skill") or ["<none>"]:
            skill_t[sk] += 1
            skill_c[sk] += int(r.get("correct", False))
        for bc in r.get("broad_capability") or ["<none>"]:
            cap_t[bc] += 1
            cap_c[bc] += int(r.get("correct", False))
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total else 0.0,
        "per_skill": {s: {"correct": skill_c[s], "total": skill_t[s],
                          "accuracy": round(skill_c[s] / skill_t[s] * 100, 2)}
                      for s in sorted(skill_t)},
        "per_broad_capability": {b: {"correct": cap_c[b], "total": cap_t[b],
                                     "accuracy": round(cap_c[b] / cap_t[b] * 100, 2)}
                                 for b in sorted(cap_t)},
        "extraction_failures": sum(1 for r in results if not r.get("pred_set")),
    }


def _write_summary(args, results, elapsed, verbose=True):
    m = _compute_metrics(results)
    summary = {
        "model": args.model_path,
        "prompt_style": args.prompt_style,
        "total": m["total"],
        "correct": m["correct"],
        "accuracy": m["accuracy"],
        "per_skill": m["per_skill"],
        "per_broad_capability": m["per_broad_capability"],
        "time_s": round(elapsed, 1),
        "results": results,
    }
    atomic_save_json(args.output, summary)
    if verbose:
        print(f"\n{'='*60}")
        print(f"LogicVista Evaluation Results")
        print(f"  Model: {args.model_path}")
        print(f"  Prompt style: {args.prompt_style}")
        print(f"  Overall: {m['correct']}/{m['total']} = {m['accuracy']:.1f}%")
        print(f"  Per skill:")
        for s in sorted(m["per_skill"]):
            v = m["per_skill"][s]
            print(f"    {s}: {v['correct']}/{v['total']} = {v['accuracy']:.1f}%")
        print(f"  Extraction failures: {m['extraction_failures']}/{m['total']}")
        print(f"  Time: {elapsed:.0f}s")
        print(f"{'='*60}")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()