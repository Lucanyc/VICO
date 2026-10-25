#!/usr/bin/env python3
"""
Evaluate Qwen3-VL on VICO test set.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 evaluate_answer_tag.py \
        --model_path /path/to/models/Qwen3-VL-8B-Instruct \
        --data_json /path/to/dataset/vico/converted_test/vico_vlmgym_format.json \
        --image_root /path/to/dataset/vico \
        --output evaluate_answer_tag_results.json
"""

import os, json, re, argparse, time
from vllm import LLM, SamplingParams
from tqdm import tqdm


SYSTEM_PROMPT = (
    "Please answer the question about the given image. "
    "The image may contain charts, graphs, geometric figures, or mathematical diagrams. "
    "You FIRST think about the reasoning process as an internal monologue "
    "with your own reasoning, and then provide the final answer. "
    "Think step by step in <think> tags, then give your final answer in <answer> tags."
)


def extract_answer(text):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def relaxed_match(pred, gt):
    """VICO relaxed accuracy: exact match or within 5% for numbers."""
    if pred is None:
        return False
    pred, gt = pred.strip().lower(), gt.strip().lower()
    if pred == gt:
        return True
    try:
        p, g = float(pred), float(gt)
        return abs(p - g) <= abs(g) * 0.05 + 1e-9
    except (ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--image_root", required=True,
                        help="Root dir for image_path (e.g. /path/to/vico)")
    parser.add_argument("--output", default="eval_vico_results.json")
    parser.add_argument("--n", type=int, default=-1, help="Number of samples (-1 = all)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    # Load data
    with open(args.data_json) as f:
        data = json.load(f)
    if args.n > 0:
        data = data[:args.n]
    print(f"Loaded {len(data)} samples")

    # Filter samples with valid images
    valid = []
    for s in data:
        img_path = os.path.join(args.image_root, s["image_path"])
        if os.path.exists(img_path):
            s["_full_image_path"] = img_path
            valid.append(s)
    print(f"Valid samples (image exists): {len(valid)}")

    # Resume: load existing results and skip done samples
    done_ids = set()
    prev_results = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            prev = json.load(f)
        prev_results = prev.get("results", [])
        done_ids = {r["id"] for r in prev_results}
        print(f"Resume: {len(done_ids)} already done")
        valid = [s for s in valid if s["id"] not in done_ids]
        print(f"Remaining: {len(valid)}")

    # Load model
    print(f"Loading model: {args.model_path}")
    allowed_path = os.path.realpath(args.image_root)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=allowed_path,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Init results from resume
    results = list(prev_results)
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    t0 = time.time()

    for i in tqdm(range(0, len(valid), args.batch_size), desc="Evaluating"):
        batch = valid[i:i + args.batch_size]
        messages_batch = []
        for s in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"file://{os.path.realpath(s['_full_image_path'])}"}},
                    {"type": "text", "text": s["question"]},
                ]},
            ]
            messages_batch.append(messages)

        outputs = llm.chat(messages_batch, sampling_params=sampling_params)

        for s, out in zip(batch, outputs):
            resp = out.outputs[0].text
            pred = extract_answer(resp)
            gt = str(s["answer"])
            match = relaxed_match(pred, gt)
            correct += int(match)
            total += 1

            results.append({
                "id": s["id"],
                "question": s["question"],
                "gt": gt,
                "pred": pred,
                "correct": match,
                "response": resp,
            })

        # Print running accuracy and incremental save
        acc = correct / total * 100
        tqdm.write(f"  [{total}/{total + len(valid) - i - len(batch)}] acc={acc:.1f}%")

        # Save after each batch (so resume works if crash)
        with open(args.output, "w") as f:
            json.dump({"model": args.model_path, "total": total, "correct": correct,
                        "accuracy": round(acc, 2), "results": results}, f, ensure_ascii=False, indent=2)

    # Summary
    elapsed = time.time() - t0
    acc = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"VICO Evaluation Results")
    print(f"  Model: {args.model_path}")
    print(f"  Samples: {total}")
    print(f"  Relaxed Accuracy: {correct}/{total} = {acc:.1f}%")
    print(f"  Time: {elapsed:.0f}s ({elapsed/total:.1f}s/sample)")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")

    # Save
    summary = {
        "model": args.model_path,
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 2),
        "time_s": round(elapsed, 1),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
