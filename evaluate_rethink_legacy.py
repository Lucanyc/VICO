#!/usr/bin/env python3
"""Evaluate Qwen3-VL on VICO test with selectable prompt style.

Mirrors the convention of eval_mathvista.py — supports three prompt styles:
  * rethinker  : VL-Rethinker style ("Please reason step by step, and put
                 your final answer within \\boxed{}.") — no system prompt
  * answer_tag : <think>/<answer> system prompt — for models GRPO-trained
                 with that output contract
  * boxed      : minimal — no system prompt, no append; for base models that
                 already follow boxed convention

Extraction is UNIVERSAL: tries <answer>...</answer>, \\boxed{...}, "the
answer is...", and a last-number fallback. So CoT + <think>/<answer>
output contract is preserved across all styles — the prompt only steers
the model's behavior.

Match: VICO relaxed_correctness (5% relative tolerance for numbers,
case-insensitive lower-strip for text). Strips LaTeX wrappers (\\text{},
\\mathrm{}, $...$) and outer punctuation before comparing.

Usage examples:
    # Trained ckpt with answer_tag (your default story)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 evaluate_rethink_legacy.py \\
        --model_path /path/to/vico-code/vico_round3_solver_merged_step110 \\
        --data_json /path/to/vico/converted_test/test_full_vico.json \\
        --image_root /path/to/vico \\
        --tp 4 --prompt_style answer_tag \\
        --output eval_vico_r3_step110_full_answer_tag.json

    # Same ckpt under VL-Rethinker prompt
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 evaluate_rethink_legacy.py \\
        --model_path /path/to/vico-code/vico_round3_solver_merged_step110 \\
        --data_json /path/to/vico/converted_test/test_full_vico.json \\
        --image_root /path/to/vico \\
        --tp 4 --prompt_style rethinker \\
        --output eval_vico_r3_step110_full_rethinker.json

    # Base Qwen3-VL-8B for comparison
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 evaluate_rethink_legacy.py \\
        --model_path /path/to/models/Qwen3-VL-8B-Instruct \\
        --data_json /path/to/vico/converted_test/test_full_vico.json \\
        --image_root /path/to/vico \\
        --tp 4 --prompt_style rethinker \\
        --output evaluate_answer_tag_legacy_full_rethinker.json
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams


# ══════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════

ANSWER_TAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response "
    "must include two parts: first, the reasoning process enclosed in "
    "<think>...</think> tags, then the final answer enclosed in "
    "<answer>...</answer> tags. Please provide a clear, concise response within "
    "<answer>...</answer> tags that directly addresses the question."
)

RETHINKER_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def build_messages(question: str, image_path: str, prompt_style: str):
    """Build chat messages for one sample given a prompt style."""
    sys_prompt = None
    user_text = question
    if prompt_style == "answer_tag":
        sys_prompt = ANSWER_TAG_SYSTEM_PROMPT
    elif prompt_style == "rethinker":
        user_text = question + RETHINKER_SUFFIX
    # boxed: no system prompt, no suffix — minimal

    content = [
        {"type": "image_url", "image_url": {
            "url": f"file://{os.path.realpath(image_path)}"}},
        {"type": "text", "text": user_text},
    ]
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": content})
    return messages


# ══════════════════════════════════════════════════════════════════════
# Answer extraction — universal across styles
# ══════════════════════════════════════════════════════════════════════

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>",
                            flags=re.IGNORECASE | re.DOTALL)
_BOXED_START = re.compile(r"\\boxed\s*{")


def extract_boxed(text: str):
    """Return contents of the LAST \\boxed{...} (handles nested braces)."""
    last = None
    for m in _BOXED_START.finditer(text or ""):
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


def extract_answer_tag(text: str):
    matches = _ANSWER_TAG_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def extract_answer(text: str, prompt_style: str = "rethinker"):
    """Universal extractor.

    Order is style-aware: prefer the format the prompt asked for first,
    then fall back through the other formats.
    """
    if not text:
        return None

    if prompt_style == "answer_tag":
        ans = extract_answer_tag(text)
        if ans:
            return ans
        ans = extract_boxed(text)
        if ans:
            return ans
    else:
        # rethinker / boxed: prefer \boxed{} first
        ans = extract_boxed(text)
        if ans:
            return ans
        ans = extract_answer_tag(text)
        if ans:
            return ans

    # "the answer is X"
    for pat in [
        r"(?:the |my |final )?answer is[:\s]*([^\n.;]+)",
        r"(?:答案|结果)[是为：:\s]*([^\n.;。]+)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # text after </think>
    if "</think>" in text:
        tail = text.split("</think>", 1)[1].strip()
        if tail:
            return tail

    # last-number fallback
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]

    return text.strip() or None


# ══════════════════════════════════════════════════════════════════════
# VICO relaxed match
# ══════════════════════════════════════════════════════════════════════

def _strip_wrappers(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").strip()
    s = s.strip(",.;:()[]{} ").strip()
    # drop trailing % so "30.33%" matches GT "30.33"
    if s.endswith("%"):
        s = s[:-1].strip()
    return s


def relaxed_match(pred, gt) -> bool:
    """Exact (case-insensitive) match OR within 5% relative for numbers."""
    if pred is None:
        return False
    p = _strip_wrappers(pred).lower()
    g = _strip_wrappers(str(gt)).lower()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) <= abs(float(g)) * 0.05 + 1e-9
    except (ValueError, TypeError):
        pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", p)
    if nums:
        try:
            return abs(float(nums[-1]) - float(g)) <= abs(float(g)) * 0.05 + 1e-9
        except (ValueError, TypeError):
            pass
    return False


# ══════════════════════════════════════════════════════════════════════
# vLLM helpers
# ══════════════════════════════════════════════════════════════════════

def mm_kwargs_for(model_path: str) -> dict:
    """Qwen-VL accepts min/max_pixels; InternVL doesn't."""
    name = Path(model_path).name.lower()
    if "qwen" in name and "vl" in name:
        return {
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }
    return {}


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", default=None,
                        help="default: eval_vico_<model_basename>_<style>.json")
    parser.add_argument("--n", type=int, default=-1, help="-1 = all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 for reproducible greedy")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--prompt_style", default="rethinker",
                        choices=["rethinker", "answer_tag", "boxed"],
                        help="rethinker (default): VL-Rethinker '\\boxed{}' instruction; "
                             "answer_tag: <think>/<answer> system prompt; "
                             "boxed: bare prompt for base models")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        base = Path(args.model_path).name or "model"
        args.output = f"eval_vico_{base}_{args.prompt_style}.json"
    print(f"model        : {args.model_path}")
    print(f"prompt_style : {args.prompt_style}")
    print(f"data_json    : {args.data_json}")
    print(f"output       : {args.output}")

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

    # Resume
    results = []
    correct, total = 0, 0
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                prev = json.load(f)
            results = prev.get("results", [])
            done_ids = {str(r["id"]) for r in results}
            correct = sum(int(r.get("correct", 0)) for r in results)
            total = len(results)
            before = len(valid)
            valid = [s for s in valid if str(s["id"]) not in done_ids]
            print(f"[Resume] Loaded {total} done (correct={correct}); "
                  f"remaining {len(valid)}/{before}")
        except Exception as e:
            print(f"[Resume] Failed: {e}; starting fresh")
            results, correct, total = [], 0, 0

    if not valid:
        print("[Resume] All samples already complete — nothing to do.")
        return

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

    t0 = time.time()
    for i in tqdm(range(0, len(valid), args.batch_size), desc="Evaluating"):
        batch = valid[i:i + args.batch_size]
        messages_batch = [
            build_messages(s["question"], s["_full_image_path"], args.prompt_style)
            for s in batch
        ]

        outputs = llm.chat(messages_batch, sampling_params=sampling_params)

        for s, out in zip(batch, outputs):
            resp = out.outputs[0].text
            pred = extract_answer(resp, args.prompt_style)
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

        acc = correct / total * 100
        tqdm.write(
            f"  [{total}/{total + len(valid) - i - len(batch)}] "
            f"acc={acc:.2f}%"
        )

        # Incremental save
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model_path,
                "prompt_style": args.prompt_style,
                "data_json": args.data_json,
                "total": total,
                "correct": correct,
                "accuracy": round(acc, 2),
                "results": results,
            }, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    acc = correct / total * 100 if total > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"VICO Evaluation Results  (prompt_style = {args.prompt_style})")
    print(f"  Model:    {args.model_path}")
    print(f"  Samples:  {total}")
    print(f"  Relaxed accuracy: {correct}/{total} = {acc:.2f}%")
    print(f"  Time:     {elapsed:.0f}s ({elapsed/max(total,1):.2f}s/sample)")
    print(f"  Output:   {args.output}")
    print(f"{'=' * 60}")

    # Final save
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model_path,
            "prompt_style": args.prompt_style,
            "data_json": args.data_json,
            "total": total,
            "correct": correct,
            "accuracy": round(acc, 2),
            "time_s": round(elapsed, 1),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
