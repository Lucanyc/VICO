#!/usr/bin/env python3
"""
Evaluate on MMVP (Multimodal Visual Patterns).

Loads MMVP/MMVP from HuggingFace. Computes both per-question accuracy and
paired accuracy (a pair counts only if BOTH questions are correct) — pair
accuracy is the official MMVP metric.

Usage (multi-GPU):
    CUDA_VISIBLE_DEVICES=0,1 python3 eval_mmvp.py \
        --model_path /path/to/models/Qwen3-VL-8B-Instruct \
        --tp 2 --prompt_style boxed \
        --output eval_mmvp_qwen3vl8b_base_results.json

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python3 eval_mmvp.py \
        --model_path /path/to/models/Qwen3-VL-8B-Instruct \
        --tp 1 --prompt_style boxed \
        --output eval_mmvp_qwen3vl8b_base_results.json
"""

import os, json, re, argparse, time
from vllm import LLM, SamplingParams
from tqdm import tqdm
from collections import defaultdict

# Reuse the robust MCQ parser from the MathVista eval — identical extraction
# logic keeps the comparison across scripts internally consistent.
from eval_mathvista import parse_mcq, extract_boxed, extract_answer_tag


DEFAULT_DATA_ROOT = "/path/to/dataset/MMVP"
DEFAULT_DATA_JSON = os.path.join(DEFAULT_DATA_ROOT, "mmvp_test.json")


def extract_answer(text, prompt_style="boxed"):
    """Extract answer from model response — MMVP is always MCQ, so we route
    everything through parse_mcq after trying <answer>/\\boxed{} containers.

    For `answer_tag` and `rethinker` styles the prompt asks the model to emit
    `<answer>...</answer>`, so that container is tried first.
    """
    if prompt_style in ("answer_tag", "rethinker"):
        container = extract_answer_tag(text) or extract_boxed(text)
    else:
        container = extract_boxed(text) or extract_answer_tag(text)

    payload = container if container else text
    letter = parse_mcq(payload)
    return letter if letter else None


def normalize_gt(ans):
    """GT in MMVP looks like '(a)' / '(b)' — strip to uppercase letter."""
    if ans is None:
        return None
    m = re.search(r'[A-Za-z]', str(ans))
    return m.group(0).upper() if m else None


def build_query(sample):
    """Build prompt: question + options + MCQ hint.

    Sample comes from mmvp_test.json with fields: question, choices (list of strings).
    """
    q = sample["question"].strip()
    lines = ["Choices:"]
    for i, c in enumerate(sample["choices"]):
        lines.append(f"({chr(ord('A') + i)}) {c}")
    opts = "\n".join(lines)
    hint = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, at the end."
    return f"Question: {q}\n{opts}\n{hint}"


def atomic_save_json(path, data):
    """Write JSON atomically via tmp + rename so an interrupted run never
    leaves a corrupt output file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", default="eval_mmvp_results.json")
    parser.add_argument("--data_json", default=DEFAULT_DATA_JSON,
                        help="Path to mmvp_test.json (built from Questions.csv)")
    parser.add_argument("--image_root", default=DEFAULT_DATA_ROOT,
                        help="Root dir; sample.image_path is resolved relative to this")
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="vLLM GPU memory fraction (lower this if OOM on a single GPU)")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--prompt_style", default="boxed",
                        choices=["boxed", "answer_tag", "rethinker"])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Resume from partial --output if it exists (default: True). "
                             "Use --no-resume to start fresh.")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Atomic-save the running summary every N batches (default: 1)")
    args = parser.parse_args()

    # ── Load data from JSON ──────────────────────────────────────────────
    print(f"Loading MMVP from {args.data_json}")
    with open(args.data_json) as f:
        data = json.load(f)
    print(f"  {len(data)} rows")
    if args.n > 0:
        data = data[:args.n]

    samples = []
    for s in data:
        img_path = os.path.join(args.image_root, s["image_path"])
        if not os.path.exists(img_path):
            print(f"  [warn] missing image: {img_path}")
            continue
        samples.append({
            "Index": int(s["id"]),
            "question": s["question"],
            "choices": s["choices"],
            "answer": s["answer"],          # already a letter "A"/"B"/...
            "image_path": img_path,
        })
    print(f"  Valid samples: {len(samples)}")

    # ── Resume: load any prior results and filter remaining ──────────────
    prior_results = []
    done_indices = set()
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                prev = json.load(f)
            prior_results = prev.get("results", [])
            done_indices = {r["index"] for r in prior_results if r.get("index") is not None}
            print(f"[resume] Loaded {len(prior_results)} existing results from {args.output}")
        except Exception as e:
            print(f"[resume] Could not parse existing output ({e}); starting fresh")
            prior_results, done_indices = [], set()

    remaining = [s for s in samples if s["Index"] not in done_indices]
    print(f"To process: {len(remaining)}/{len(samples)} ({len(done_indices)} already done)")

    # Nothing to do — just recompute metrics and exit
    if not remaining:
        print("[resume] All samples already evaluated; regenerating summary only.")
        _write_summary(args, prior_results, elapsed=0.0)
        return

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=os.path.realpath(args.image_root),
    )
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)

    # ── Inference ────────────────────────────────────────────────────────
    results = list(prior_results)  # carry over resumed rows
    correct = sum(int(r.get("correct", False)) for r in results)
    total = len(results)
    t0 = time.time()

    for bi, i in enumerate(tqdm(range(0, len(remaining), args.batch_size), desc="Evaluating")):
        batch = remaining[i:i + args.batch_size]
        messages_batch = []
        for s in batch:
            query = build_query(s)
            if args.prompt_style == "answer_tag":
                sys_prompt = (
                    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
                    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. "
                    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
                )
            else:
                sys_prompt = None
            if args.prompt_style == "rethinker":
                query = query + "\n\nPlease reason step by step, and put your final answer (the option letter, e.g., A or B) within <answer> </answer> tags."

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
            pred = extract_answer(resp, args.prompt_style)
            gt = normalize_gt(s["answer"])
            match = (pred is not None) and (pred == gt)
            correct += int(match)
            total += 1
            results.append({
                "index": s["Index"],
                "question": s["question"],
                "choices": s["choices"],
                "gt": gt,
                "pred": pred,
                "correct": match,
                "response": resp,
            })
        tqdm.write(f"  [{total}/{len(samples)}] acc={correct/total*100:.1f}%")

        # Periodic atomic save — cheap insurance against crashes / preemption
        if ((bi + 1) % args.save_every) == 0:
            _write_summary(args, results, elapsed=time.time() - t0, verbose=False)

    elapsed = time.time() - t0
    _write_summary(args, results, elapsed=elapsed, verbose=True)


def _compute_metrics(results):
    """Compute per-question + pair metrics from a list of result rows."""
    total = len(results)
    correct = sum(int(r.get("correct", False)) for r in results)
    by_index = {r["index"]: r for r in results}
    pair_correct, pair_total = 0, 0
    pair_details = []
    for k in range(1, 151):
        a = by_index.get(2 * k - 1)
        b = by_index.get(2 * k)
        if a is None or b is None:
            continue
        pair_total += 1
        both = bool(a["correct"]) and bool(b["correct"])
        pair_correct += int(both)
        pair_details.append({"pair": k, "q1_idx": a["index"], "q2_idx": b["index"],
                             "q1_correct": a["correct"], "q2_correct": b["correct"],
                             "pair_correct": both})
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total else 0.0,
        "pair_total": pair_total,
        "pair_correct": pair_correct,
        "pair_accuracy": round(pair_correct / pair_total * 100, 2) if pair_total else 0.0,
        "extraction_failures": sum(1 for r in results if r.get("pred") is None),
        "pair_details": pair_details,
    }


def _write_summary(args, results, elapsed, verbose=True):
    m = _compute_metrics(results)
    summary = {
        "model": args.model_path,
        "prompt_style": args.prompt_style,
        "total": m["total"],
        "correct": m["correct"],
        "accuracy": m["accuracy"],
        "pair_total": m["pair_total"],
        "pair_correct": m["pair_correct"],
        "pair_accuracy": m["pair_accuracy"],
        "time_s": round(elapsed, 1),
        "pair_details": m["pair_details"],
        "results": results,
    }
    atomic_save_json(args.output, summary)
    if verbose:
        print(f"\n{'='*60}")
        print(f"MMVP Evaluation Results")
        print(f"  Model: {args.model_path}")
        print(f"  Prompt style: {args.prompt_style}")
        print(f"  Per-question: {m['correct']}/{m['total']} = {m['accuracy']:.1f}%")
        print(f"  Pair accuracy (official): {m['pair_correct']}/{m['pair_total']} = {m['pair_accuracy']:.1f}%")
        print(f"  Extraction failures: {m['extraction_failures']}/{m['total']}")
        print(f"  Time: {elapsed:.0f}s")
        print(f"{'='*60}")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
