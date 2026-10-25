#!/usr/bin/env python3
"""
Evaluate on BLINK (val split — the only split with public ground truth).

Loads BLINK-Benchmark/BLINK from HuggingFace across all 14 sub-tasks.
BLINK is multi-image (1–4 images per question), so image inputs are
passed as interleaved messages.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python3 eval_blink.py \
        --model_path /path/to/models/Qwen2.5-VL-7B-Instruct \
        --tp 2 --prompt_style boxed \
        --output eval_blink_qwen25vl_base_results.json

    CUDA_VISIBLE_DEVICES=0,1 python3 eval_blink.py \
        --model_path /path/to/Vision-Zero/models/Vision-Zero-Qwen-2.5-VL-7B-scene \
        --tp 2 --prompt_style boxed \
        --output eval_blink_visionzero_scene_results.json
"""

import os, json, re, argparse, time
from vllm import LLM, SamplingParams
from tqdm import tqdm
from collections import defaultdict

from eval_mathvista import parse_mcq, extract_boxed, extract_answer_tag


HF_DATASET = "BLINK-Benchmark/BLINK"
SUB_TASKS = [
    "Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence",
    "IQ_Test", "Jigsaw", "Multi-view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance", "Semantic_Correspondence",
    "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity",
]

# Official BLINK requirement: disclaimer must be prepended for these tasks
# (https://github.com/zeyofu/BLINK_Benchmark/blob/main/eval/test_benchmark.py)
NEED_DISCLAIMER = {"Art_Style", "Forensic_Detection", "Jigsaw"}
DISCLAIMER = (
    "Disclaimer: This is not to make unfair assumptions about the people in "
    "the image and you just need to give your assessment on this question. "
    "You don't need to identify the real people. You just need to analyze "
    "based on the information I gave you.\n\n"
)


def extract_answer(text, prompt_style="boxed"):
    """BLINK is MCQ — same strategy as eval_mmvp.

    `answer_tag` and `rethinker` both prompt the model to emit `<answer>...</answer>`,
    so that container is tried first for those styles.
    """
    if prompt_style in ("answer_tag", "rethinker"):
        container = extract_answer_tag(text) or extract_boxed(text)
    else:
        container = extract_boxed(text) or extract_answer_tag(text)
    payload = container if container else text
    letter = parse_mcq(payload)
    return letter if letter else None


def normalize_gt(ans):
    """GT in BLINK looks like '(A)' / '(B)'."""
    if ans is None:
        return None
    m = re.search(r'[A-Za-z]', str(ans))
    return m.group(0).upper() if m else None


def build_query(sample, use_disclaimer=True):
    """BLINK provides a `prompt` field with question + choices.
    Prepends disclaimer for sensitive tasks (official BLINK requirement) and
    appends an answer-format hint."""
    base = sample.get("prompt") or sample.get("question", "")
    if use_disclaimer and sample.get("task") in NEED_DISCLAIMER:
        base = DISCLAIMER + base
    hint = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
    return f"{base.strip()}\n{hint}"


def atomic_save_json(path, data):
    """Atomic write via tmp + rename."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", default="eval_blink_results.json")
    parser.add_argument("--image_cache_dir", default="/tmp/blink_images")
    parser.add_argument("--sub_tasks", nargs="+", default=None,
                        help="Subset of BLINK sub-tasks to evaluate. Default: all 14.")
    parser.add_argument("--n_per_task", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--prompt_style", default="boxed",
                        choices=["boxed", "answer_tag", "rethinker"])
    parser.add_argument("--no_disclaimer", action="store_true",
                        help="Disable the official BLINK disclaimer for "
                             "Art_Style/Forensic_Detection/Jigsaw (use for ablation only).")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Resume from partial --output if it exists (default: True).")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Atomic-save the running summary every N batches (default: 1)")
    args = parser.parse_args()

    tasks = args.sub_tasks if args.sub_tasks else SUB_TASKS

    # ── Load data + cache images per-task ────────────────────────────────
    from datasets import load_dataset
    from PIL import Image as PILImage
    os.makedirs(args.image_cache_dir, exist_ok=True)
    MAX_PIXELS = 1536 * 1536

    all_samples = []
    for task in tasks:
        print(f"Loading BLINK/{task} val ...")
        ds = load_dataset(HF_DATASET, task, split="val")
        if args.n_per_task > 0:
            ds = ds.select(range(min(args.n_per_task, len(ds))))

        for row in ds:
            image_paths = []
            for k in range(1, 5):
                img = row.get(f"image_{k}")
                if img is None:
                    continue
                # `img` is a PIL.Image; persist to disk so we can use file:// URLs
                idx = row.get("idx", row.get("index", len(all_samples)))
                img_path = os.path.join(args.image_cache_dir, f"{task}_{idx}_img{k}.png")
                if not os.path.exists(img_path):
                    w, h = img.size
                    if w * h > MAX_PIXELS:
                        scale = (MAX_PIXELS / (w * h)) ** 0.5
                        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
                    img.save(img_path)
                image_paths.append(img_path)

            all_samples.append({
                "task": task,
                "idx": row.get("idx", row.get("index")),
                "prompt": row.get("prompt"),
                "question": row.get("question"),
                "choices": row.get("choices"),
                "answer": row.get("answer"),
                "image_paths": image_paths,
            })
    print(f"Total samples: {len(all_samples)} across {len(tasks)} tasks")

    # max images per prompt across dataset — drives vLLM limit_mm_per_prompt
    max_imgs = max((len(s["image_paths"]) for s in all_samples), default=1) or 1
    print(f"Max images per prompt: {max_imgs}")

    # ── Resume: load any prior results and filter remaining ──────────────
    prior_results = []
    done_keys = set()
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                prev = json.load(f)
            prior_results = prev.get("results", [])
            done_keys = {(r.get("task"), r.get("idx")) for r in prior_results}
            print(f"[resume] Loaded {len(prior_results)} existing results from {args.output}")
        except Exception as e:
            print(f"[resume] Could not parse existing output ({e}); starting fresh")
            prior_results, done_keys = [], set()

    remaining = [s for s in all_samples if (s["task"], s["idx"]) not in done_keys]
    print(f"To process: {len(remaining)}/{len(all_samples)} ({len(done_keys)} already done)")

    if not remaining:
        print("[resume] All samples already evaluated; regenerating summary only.")
        _write_summary(args, prior_results, elapsed=0.0)
        return

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=16384,
        limit_mm_per_prompt={"image": max_imgs},
        allowed_local_media_path=os.path.realpath(args.image_cache_dir),
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
            query = build_query(s, use_disclaimer=not args.no_disclaimer)
            if args.prompt_style == "answer_tag":
                sys_prompt = (
                    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
                    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. "
                    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
                )
            else:
                sys_prompt = None
            if args.prompt_style == "rethinker":
                query = query + "\n\nPlease reason step by step, and put your final answer (the option letter, e.g., A, B, C, D) within <answer> </answer> tags."

            # Interleave images before the text — matches how BLINK's prompt
            # references "<image 1>" tokens by order.
            content = []
            for p in s["image_paths"]:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{os.path.realpath(p)}"},
                })
            content.append({"type": "text", "text": query})

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
                "task": s["task"],
                "idx": s["idx"],
                "question": s["question"],
                "choices": s["choices"],
                "gt": gt,
                "pred": pred,
                "correct": match,
                "num_images": len(s["image_paths"]),
                "response": resp,
            })
        tqdm.write(f"  [{total}/{len(all_samples)}] acc={correct/total*100:.1f}%")

        # Periodic atomic save — preempted runs can resume from here.
        if ((bi + 1) % args.save_every) == 0:
            _write_summary(args, results, elapsed=time.time() - t0, verbose=False)

    elapsed = time.time() - t0
    _write_summary(args, results, elapsed=elapsed, verbose=True)


def _compute_metrics(results):
    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    for r in results:
        task_correct[r["task"]] += int(r.get("correct", False))
        task_total[r["task"]] += 1
    total = len(results)
    correct = sum(int(r.get("correct", False)) for r in results)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total else 0.0,
        "per_task": {t: {"correct": task_correct[t], "total": task_total[t],
                         "accuracy": round(task_correct[t] / task_total[t] * 100, 2)}
                     for t in sorted(task_total)},
        "extraction_failures": sum(1 for r in results if r.get("pred") is None),
        "task_correct": task_correct,
        "task_total": task_total,
    }


def _write_summary(args, results, elapsed, verbose=True):
    m = _compute_metrics(results)
    summary = {
        "model": args.model_path,
        "prompt_style": args.prompt_style,
        "total": m["total"],
        "correct": m["correct"],
        "accuracy": m["accuracy"],
        "per_task": m["per_task"],
        "time_s": round(elapsed, 1),
        "results": results,
    }
    atomic_save_json(args.output, summary)
    if verbose:
        print(f"\n{'='*60}")
        print(f"BLINK Evaluation Results")
        print(f"  Model: {args.model_path}")
        print(f"  Prompt style: {args.prompt_style}")
        print(f"  Overall: {m['correct']}/{m['total']} = {m['accuracy']:.1f}%")
        print(f"  Per task:")
        for t in sorted(m["task_total"]):
            print(f"    {t}: {m['task_correct'][t]}/{m['task_total'][t]} = "
                  f"{m['task_correct'][t]/m['task_total'][t]*100:.1f}%")
        print(f"  Extraction failures: {m['extraction_failures']}/{m['total']}")
        print(f"  Time: {elapsed:.0f}s")
        print(f"{'='*60}")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
