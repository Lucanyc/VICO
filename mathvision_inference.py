#!/usr/bin/env python3
"""
MathVision Test Inference — vLLM accelerated

Usage:
    python mathvision_inference.py \
        --model /path/to/merged_model \
        --test_file /path/to/dataset/MathVision/test_output/test.json \
        --output mathvision_eval.json \
        --tp 8
"""

import os, json, re, argparse, time
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "Please answer the question about the given image. "
    "The image may contain charts, graphs, geometric figures, or mathematical diagrams. "
    "You FIRST think about the reasoning process as an internal monologue "
    "with your own reasoning, and then provide the final answer. "
    "Think step by step in <think> tags, then give your final answer in <answer> tags."
)


def extract_answer(text):
    """Extract answer from <answer>...</answer> tags, fallback to boxed/last token."""
    # Try <answer>...</answer>
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try \boxed{...}
    for pat in [r'\\boxed\{', r'\\boxed \{']:
        m = re.search(pat, text)
        if m:
            s, d, i = m.end(), 1, m.end()
            while i < len(text) and d > 0:
                if text[i] == '{': d += 1
                elif text[i] == '}': d -= 1
                i += 1
            if d == 0:
                ans = text[s:i-1].strip()
                return re.sub(r'[\\%$]', '', ans).strip()

    return ""


def check_answer(pred, gt, options=None):
    """Check if prediction matches ground truth."""
    if not pred:
        return False
    pred = str(pred).strip()
    gt = str(gt).strip()

    # Exact match (case-insensitive)
    if pred.lower() == gt.lower():
        return True

    # For MCQ: check if pred contains the option letter
    if options and len(gt) == 1 and gt.isalpha():
        # pred might be "A" or "A. something" or "(A)"
        pred_letter = re.match(r'^[(\s]*([A-E])[)\s.:,]?', pred.upper())
        if pred_letter and pred_letter.group(1) == gt.upper():
            return True

    # Numeric match
    try:
        pn = float(pred.replace(',', ''))
        gn = float(gt.replace(',', ''))
        if gn == 0:
            return abs(pn) < 1e-6
        return abs(gn - pn) / abs(gn) <= 0.05
    except (ValueError, TypeError):
        pass

    return False


def main():
    parser = argparse.ArgumentParser(description="MathVision Inference (vLLM)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--image_base_dir", default=None,
                        help="Base dir for image paths (default: same dir as test_file)")
    parser.add_argument("--output", default="mathvision_eval.json")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--show_examples", type=int, default=3)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skipping already-processed samples")
    args = parser.parse_args()

    if args.image_base_dir is None:
        args.image_base_dir = os.path.dirname(os.path.abspath(args.test_file))

    print("=" * 60)
    print("MathVision Inference")
    print(f"  Model: {args.model}")
    print(f"  Test:  {args.test_file}")
    print(f"  Image base: {args.image_base_dir}")
    print(f"  TP:    {args.tp}")
    print(f"  Temp:  {args.temperature}")
    print("=" * 60)

    # Load vLLM
    allowed_media_path = os.path.realpath(args.image_base_dir)
    print("\nLoading vLLM engine...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=allowed_media_path,
    )
    print("  Engine ready.")

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=0.95,
    )

    # Load data
    with open(args.test_file) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples")

    if args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"  Using first {len(data)} samples")

    # Filter valid images
    valid_samples = []
    for s in data:
        img = os.path.join(args.image_base_dir, s['image'])
        if os.path.exists(img):
            s['_image_path'] = os.path.realpath(img)
            valid_samples.append(s)

    print(f"  Valid samples: {len(valid_samples)}")

    # Resume: load existing results and skip already-processed samples
    results = []
    correct = 0
    total = 0
    subject_stats = {}
    format_ok_count = 0

    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            prev = json.load(f)
        prev_results = prev.get("results", [])
        done_ids = {r["id"] for r in prev_results}
        results = prev_results
        for r in prev_results:
            total += 1
            if r["correct"]:
                correct += 1
            if r.get("format_ok"):
                format_ok_count += 1
            subj = r.get("subject", "unknown")
            if subj not in subject_stats:
                subject_stats[subj] = {"correct": 0, "total": 0}
            subject_stats[subj]["total"] += 1
            if r["correct"]:
                subject_stats[subj]["correct"] += 1
        valid_samples = [s for s in valid_samples if s["id"] not in done_ids]
        print(f"  Resumed: {len(done_ids)} already done, {len(valid_samples)} remaining")

    # Inference
    t0 = time.time()

    for batch_start in range(0, len(valid_samples), args.batch_size):
        batch = valid_samples[batch_start:batch_start + args.batch_size]

        batch_messages = []
        for s in batch:
            # Build question with options if MCQ
            question = s['question']
            if s.get('options'):
                opts_str = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(s['options']))
                question = question + "\n\nOptions:\n" + opts_str

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"file://{s['_image_path']}"}},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            batch_messages.append(messages)

        outputs = llm.chat(batch_messages, sampling_params=sampling_params)

        batch_correct = 0
        for i, (s, output) in enumerate(zip(batch, outputs)):
            response = output.outputs[0].text
            gt = str(s['answer'])
            pred = extract_answer(response)
            options = s.get('options', [])
            is_correct = check_answer(pred, gt, options)
            subject = s.get('subject', 'unknown')

            has_format = "<think>" in response and "</think>" in response and \
                         "<answer>" in response and "</answer>" in response
            if has_format:
                format_ok_count += 1

            if is_correct:
                correct += 1
                batch_correct += 1
            total += 1

            if subject not in subject_stats:
                subject_stats[subject] = {"correct": 0, "total": 0}
            subject_stats[subject]["total"] += 1
            if is_correct:
                subject_stats[subject]["correct"] += 1

            results.append({
                "id": s['id'],
                "question": s['question'][:200],
                "answer": gt,
                "prediction": pred,
                "correct": is_correct,
                "format_ok": has_format,
                "response": response,
                "response_len": len(response),
                "subject": subject,
                "level": s.get('level', ''),
                "has_options": bool(options),
            })

            if i < args.show_examples:
                icon = "O" if is_correct else "X"
                fmt = "fmt_ok" if has_format else "NO_FMT"
                print(f"    [{icon}] [{fmt}] [{subject}] Q: {s['question'][:70]}...")
                print(f"       Pred: {pred}  |  GT: {gt}  |  resp_len: {len(response)}")

        done = total
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        acc = correct / total * 100
        batch_acc = batch_correct / len(batch) * 100
        fmt_rate = format_ok_count / total * 100
        remaining = len(valid_samples) - done
        print(f"\n  [{done}/{len(valid_samples)}] "
              f"batch_acc={batch_acc:.1f}%  "
              f"running_acc={acc:.1f}% ({correct}/{total})  "
              f"format_ok={fmt_rate:.1f}%  "
              f"speed={speed:.1f} samples/s"
              f"  (remaining={remaining})")

        # Incremental save after each batch
        _save = {
            "model": args.model, "test_file": args.test_file,
            "accuracy_all": acc, "correct": correct, "total": total,
            "format_ok_rate": fmt_rate, "subject_stats": subject_stats,
            "results": results, "_partial": True,
        }
        with open(args.output, "w") as f:
            json.dump(_save, f, indent=2, ensure_ascii=False)

    # Final
    elapsed = time.time() - t0
    accuracy = correct / total * 100 if total > 0 else 0
    fmt_rate = format_ok_count / total * 100 if total > 0 else 0

    # Accuracy excluding empty predictions (no answer extracted)
    valid_results = [r for r in results if r['prediction']]
    invalid_count = total - len(valid_results)
    valid_correct = sum(1 for r in valid_results if r['correct'])
    valid_acc = valid_correct / len(valid_results) * 100 if valid_results else 0

    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    print(f"  Total: {total}, Correct: {correct}, Accuracy (all): {accuracy:.2f}%")
    print(f"  No answer extracted: {invalid_count}/{total} ({invalid_count/total*100:.1f}%)")
    print(f"  Valid predictions: {len(valid_results)}, Correct: {valid_correct}, Accuracy (valid only): {valid_acc:.2f}%")
    print(f"  Format OK: {format_ok_count}/{total} ({fmt_rate:.1f}%)")
    print(f"  Time: {elapsed:.1f}s ({total/elapsed:.1f} samples/s)")

    # Per-subject (valid only)
    valid_subject_stats = {}
    for r in valid_results:
        s = r['subject']
        if s not in valid_subject_stats:
            valid_subject_stats[s] = {"correct": 0, "total": 0}
        valid_subject_stats[s]["total"] += 1
        if r['correct']:
            valid_subject_stats[s]["correct"] += 1

    print(f"\n  Per-subject accuracy (valid predictions only):")
    print(f"    {'Subject':<35s} {'Acc':>7s}  {'Correct':>7s} / {'Total':>5s}")
    print(f"    {'-'*60}")
    for s, st in sorted(valid_subject_stats.items(), key=lambda x: -x[1]["total"]):
        s_acc = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0
        print(f"    {s:<35s} {s_acc:>6.1f}%  {st['correct']:>7d} / {st['total']:>5d}")

    # MCQ vs free-form (valid only)
    mcq = [r for r in valid_results if r['has_options']]
    free = [r for r in valid_results if not r['has_options']]
    if mcq:
        mcq_acc = sum(1 for r in mcq if r['correct']) / len(mcq) * 100
        print(f"\n  MCQ accuracy (valid): {mcq_acc:.1f}% ({len(mcq)} samples)")
    if free:
        free_acc = sum(1 for r in free if r['correct']) / len(free) * 100
        print(f"  Free-form accuracy (valid): {free_acc:.1f}% ({len(free)} samples)")

    # Save
    output_data = {
        "model": args.model,
        "test_file": args.test_file,
        "accuracy_all": accuracy,
        "accuracy_valid": valid_acc,
        "format_ok_rate": fmt_rate,
        "correct": correct,
        "total": total,
        "no_answer_count": invalid_count,
        "valid_total": len(valid_results),
        "valid_correct": valid_correct,
        "elapsed_seconds": elapsed,
        "subject_stats": subject_stats,
        "valid_subject_stats": valid_subject_stats,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
