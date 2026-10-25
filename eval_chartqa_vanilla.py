#!/usr/bin/env python3
"""
Evaluate Qwen3-VL on ChartQA test set — vanilla variant.

Same pipeline as eval_chartqa_base.py, but uses a minimal single-line prompt
that asks the model to put its answer directly inside \\boxed{}. Answer
extraction is correspondingly switched to pull the last \\boxed{...}.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 eval_chartqa_vanilla.py \
        --model_path /path/to/models/Qwen3-VL-8B-Instruct \
        --data_json /path/to/dataset/chartqa/converted_test/chartqa_vlmgym_format.json \
        --image_root /path/to/dataset/chartqa \
        --output eval_chartqa_vanilla_results.json
"""

import os, json, re, argparse, time
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm


# -------- edit this prompt to your liking ----------------------------------
SYSTEM_PROMPT = (
    "Please answer the question about the given chart/table/image with step-by-step reasoning"
    "and put your answer into \\boxed{}"
)


# Model shortcuts. Extend as you add more local checkpoints.
MODEL_ALIASES = {
    "qwen3vl8b":       "/path/to/models/Qwen3-VL-8B-Instruct",
    "qwen2.5vl7b":     "/path/to/models/Qwen2.5-VL-7B-Instruct",
    "internvl3-8b":    "/path/to/models/InternVL3-8B-Instruct",
    "openvlthinker7b": "/path/to/models/OpenVLThinker-7B",
}


def resolve_model(arg: str) -> tuple[str, str]:
    """Return (model_path, tag). `arg` is an alias or a filesystem path."""
    if arg in MODEL_ALIASES:
        return MODEL_ALIASES[arg], arg
    if Path(arg).exists():
        for tag, p in MODEL_ALIASES.items():
            if os.path.realpath(p) == os.path.realpath(arg):
                return arg, tag
        return arg, Path(arg).name
    raise SystemExit(
        f"unknown model: {arg}\n"
        f"  known aliases: {', '.join(MODEL_ALIASES)}\n"
        f"  or pass a local path that exists."
    )


def mm_kwargs_for(model_path: str) -> dict:
    """Qwen-VL uses min/max_pixels; InternVL's processor rejects those
    kwargs — let it read defaults from the model config instead.

    Detects Qwen-VL family by either dir-name ('qwen*vl*') OR by reading
    config.json's `model_type` (catches Qwen2.5-VL fine-tunes whose dir
    name doesn't contain 'qwen', e.g. OpenVLThinker-7B).
    """
    name = Path(model_path).name.lower()
    is_qwen_vl = "qwen" in name and "vl" in name
    if not is_qwen_vl:
        cfg = Path(model_path) / "config.json"
        if cfg.exists():
            try:
                model_type = json.load(open(cfg)).get("model_type", "")
                is_qwen_vl = model_type.startswith("qwen") and "vl" in model_type
            except Exception:
                pass
    if is_qwen_vl:
        return {
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }
    return {}


# Brace-balanced \boxed{...} extractor (handles nested braces like \frac{a}{b}).
_BOXED_START = re.compile(r"\\boxed\s*{")


def extract_answer(text):
    """Return the content of the last \\boxed{...}, or None."""
    if not text:
        return None
    last = None
    for m in _BOXED_START.finditer(text):
        i = m.end()
        depth, buf = 1, []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
                buf.append(c)
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
                buf.append(c)
            else:
                buf.append(c)
            i += 1
        if depth == 0:
            last = "".join(buf).strip()
    return last


def relaxed_match(pred, gt):
    """ChartQA relaxed accuracy: exact match or within 5% for numbers."""
    if pred is None:
        return False
    pred, gt = pred.strip().lower(), gt.strip().lower()
    # strip common LaTeX wrappers so '\text{42}' == '42'
    pred = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", pred).strip()
    pred = pred.strip("$").strip()
    if pred == gt:
        return True
    try:
        p, g = float(pred), float(gt)
        return abs(p - g) <= abs(g) * 0.05 + 1e-9
    except (ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help=f"alias ({' | '.join(MODEL_ALIASES)}) or local model path")
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--image_root", required=True,
                        help="Root dir for image_path (e.g. /path/to/chartqa)")
    parser.add_argument("--output", default=None,
                        help="default: eval_chartqa_vanilla_<model-tag>.json")
    parser.add_argument("--n", type=int, default=-1, help="Number of samples (-1 = all)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    # Resolve model + derive default output name from the tag.
    model_path, model_tag = resolve_model(args.model_path)
    args.model_path = model_path
    if args.output is None:
        args.output = f"eval_chartqa_vanilla_{model_tag}.json"
    print(f"model: {model_path}  (tag={model_tag})")
    print(f"out  : {args.output}")

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
    mm_kwargs = mm_kwargs_for(args.model_path)
    print(f"mm_processor_kwargs: {mm_kwargs}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=allowed_path,
        mm_processor_kwargs=mm_kwargs,
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
                        "accuracy": round(acc, 2), "results": results},
                      f, ensure_ascii=False, indent=2)

    # Summary
    elapsed = time.time() - t0
    acc = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"ChartQA Evaluation Results (vanilla \\boxed{{}} prompt)")
    print(f"  Model: {args.model_path}")
    print(f"  Samples: {total}")
    print(f"  Relaxed Accuracy: {correct}/{total} = {acc:.1f}%")
    print(f"  Time: {elapsed:.0f}s ({elapsed/total:.1f}s/sample)")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")

    # Save
    summary = {
        "model": args.model_path,
        "prompt": SYSTEM_PROMPT,
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
