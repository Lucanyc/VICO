#!/usr/bin/env python3
"""
Generate VICO Solver SFT data with Qwen3-VL-235B-A22B-Thinking (API version).

Adapted from the local-inference version — replaces HF model loading with
OpenAI-compatible API calls. No GPU needed.

Usage:
    export OPENAI_API_KEY="sk-..."
    export OPENAI_BASE_URL="http://14.103.68.46/v1"

    python3 generate_solver_sft_api.py \
        --data_json /path/to/dataset/vico/train/train_augmented.json \
        --png_dir /path/to/dataset/vico/train/png \
        --start 8300 --n 200 \
        --output_dir vico_solver_sft_235b \
        --model qwen3-vl-235b-a22b-thinking \
        --resume
"""

import os, json, argparse, random, re, time, base64
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm
from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════
#  Configuration  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert visual reasoning assistant for chart images.

When answering, carefully observe the chart image and reason step by step. Think naturally as a human would — use inner dialogue like "let me think," "hmm," "wait," "I notice," "let me check again," etc. Include self-reflection or verification as you reason through the problem.

Your thought process should be output within <think>...</think> tags, and your final answer within <answer>...</answer> tags. For example: <think> ... </think><answer> ... </answer>.

Important:
- Carefully read ALL data labels, axis values, legends, and titles in the chart
- Identify the chart type (bar, line, pie, etc.) and understand the data structure
- List out every relevant data point you can see in the chart
- Break down the question into sub-tasks and solve each one
- If the question involves counting, comparison, or calculation, show ALL your work explicitly
- Go through each relevant data point one by one — do not skip or summarize
- Verify your answer using a DIFFERENT method before finalizing
- Think thoroughly and in detail — your reasoning should be at least 300 words
- Give a concise final answer (just the number or short text, not a full sentence)"""

SOLVER_SYSTEM_PROMPT = (
    "Please answer the question about the chart image. "
    "Think step by step in <think> tags, then give your final answer in <answer> tags."
)


# ═══════════════════════════════════════════════════════════════════
#  Helpers  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════

def extract_answer(text: str) -> Optional[str]:
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_think(text: str) -> Optional[str]:
    match = re.search(r'<think>\s*(.*?)\s*</think>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def check_answer(pred: str, gt: str) -> bool:
    if not pred:
        return False
    pred_s = pred.strip().lower()
    gt_s = str(gt).strip().lower()

    # Remove common units/modifiers for comparison
    for word in ["million", "billion", "trillion", "thousand", "%", "$",
                 "dollars", "euros", "pounds", "approximately", "about"]:
        pred_s = pred_s.replace(word, "").strip()
        gt_s = gt_s.replace(word, "").strip()

    if pred_s == gt_s:
        return True

    # Numeric comparison with tolerance
    try:
        gn = float(gt_s.replace(",", ""))
        pn = float(pred_s.replace(",", ""))
        if gn == 0:
            return abs(pn) < 1e-6
        # 5% tolerance for VICO (chart values are approximate)
        if abs(gn - pn) / abs(gn) <= 0.05:
            return True
        # Check scale factors (million/billion)
        for scale in [1e3, 1e6, 1e9]:
            if abs(gn * scale - pn) / abs(gn * scale) <= 0.05:
                return True
            if pn != 0 and abs(pn * scale - gn) / abs(pn * scale) <= 0.05:
                return True
    except (ValueError, TypeError):
        pass

    return False


def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in [
        "can't provide", "cannot provide", "can not provide",
        "unable to provide", "won't provide", "i'm not able",
        "i cannot answer", "not answerable"])


def atomic_json_save(data, filepath):
    tmpf = filepath + ".tmp"
    with open(tmpf, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmpf, filepath)


# ═══════════════════════════════════════════════════════════════════
#  Image helper (new, for API base64 encoding)
# ═══════════════════════════════════════════════════════════════════

def encode_image_b64(path: str) -> str:
    """Read an image from disk and return a data: URL (base64)."""
    ext = Path(path).suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


# ═══════════════════════════════════════════════════════════════════
#  Generation via API
# ═══════════════════════════════════════════════════════════════════

def generate_one(client: OpenAI, model_name: str,
                 sid, question: str, image_path: str, answer: str,
                 temperature: float, max_tokens: int) -> Dict:
    if not os.path.exists(image_path):
        return {
            "id": sid, "question": question, "answer": answer,
            "image_path": image_path,
            "status": "failed", "error": "image not found",
        }

    try:
        image_data_url = encode_image_b64(image_path)
    except Exception as e:
        return {
            "id": sid, "question": question, "answer": answer,
            "image_path": image_path,
            "status": "failed", "error": f"image encode failed: {e}",
        }

    user_content = (
        f"Look at the chart image carefully and answer the following question.\n\n"
        f"Question: {question}\n\n"
        f"Please carefully analyze the chart and reason step by step, making sure to "
        f"reference the visual content of the chart (data values, labels, axes, legends) "
        f"as part of your thought process. "
        f"Think as a human would: use natural internal dialogue such as \"let me think,\" "
        f"\"hmm,\" \"wait,\" \"oh, I see,\" \"let me check again,\" etc., and include "
        f"self-reflection or verification as you reason through the problem."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": user_content},
        ]},
    ]

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature + attempt * 0.1,
                max_tokens=max_tokens,
            )
            msg = resp.choices[0].message
            content = (msg.content or "").strip()

            # Some OpenAI-compatible servers for "thinking" models expose the
            # chain-of-thought separately in `reasoning_content`. Merge both
            # into the <think>/<answer> format that the rest of the script expects.
            reasoning = getattr(msg, "reasoning_content", None) or ""

            if reasoning:
                if "<answer>" in content:
                    full_output = f"<think>\n{reasoning}\n</think>\n{content}"
                else:
                    # content holds just the final answer
                    final_ans = content if content else ""
                    full_output = (
                        f"<think>\n{reasoning}\n</think>\n"
                        f"<answer>{final_ans}</answer>"
                    )
            else:
                # Fall back to the original parsing logic — model emitted
                # <think>...</think><answer>...</answer> inline in content
                if "<think>" not in content:
                    if "</think>" in content:
                        full_output = "<think>\n" + content
                    else:
                        full_output = content
                else:
                    full_output = content

            think = extract_think(full_output)
            answer_pred = extract_answer(full_output)
            refusal = is_refusal(full_output)

            if think and answer_pred and not refusal:
                think_wc = len(think.split())
                is_correct = check_answer(answer_pred, answer)
                return {
                    "id": sid, "question": question, "answer": answer,
                    "image_path": image_path,
                    "predicted": answer_pred, "correct": is_correct,
                    "think": think, "full_output": full_output,
                    "word_count": len(full_output.split()),
                    "think_word_count": think_wc,
                    "refusal": False,
                }
            elif refusal:
                return {
                    "id": sid, "question": question, "answer": answer,
                    "image_path": image_path,
                    "status": "refusal", "refusal": True,
                    "full_output": full_output[:500],
                }
            else:
                # Could not parse both think and answer; retry with a bit
                # more temperature unless this is the last attempt.
                if attempt < 2:
                    time.sleep(1)
                    continue
                return {
                    "id": sid, "question": question, "answer": answer,
                    "image_path": image_path,
                    "status": "parse_failed",
                    "full_output": full_output[:1000],
                }

        except Exception as e:
            if attempt == 2:
                return {
                    "id": sid, "question": question, "answer": answer,
                    "image_path": image_path,
                    "status": "failed", "error": str(e)[:200],
                }
            # Exponential backoff for transient API errors
            time.sleep(2 ** attempt)

    return {
        "id": sid, "question": question, "answer": answer,
        "image_path": image_path, "status": "failed",
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate VICO Solver SFT data with Qwen3-VL-235B-Thinking (API)")
    parser.add_argument("--data_json", required=True, help="VICO train JSON")
    parser.add_argument("--png_dir", required=True, help="Directory with chart PNGs")
    parser.add_argument("--start", type=int, default=8000,
                        help="Start index into data_json")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of samples to process")
    parser.add_argument("--output_dir", required=True)

    # Model / API options
    parser.add_argument("--model", default="qwen3-vl-235b-a22b-thinking",
                        help="API model name")
    parser.add_argument("--model_path", default=None,
                        help="(ignored in API mode; kept for backward compatibility)")
    parser.add_argument("--api_key", default=None,
                        help="Override OPENAI_API_KEY env var")
    parser.add_argument("--base_url", default=None,
                        help="Override OPENAI_BASE_URL env var")

    parser.add_argument("--max_tokens", type=int, default=8096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--min_think_words", type=int, default=300)
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save results every N samples (default 1 = save every sample)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_json}")
    with open(args.data_json) as f:
        all_data = json.load(f)

    samples = all_data[args.start:args.start + args.n]
    print(f"Samples: {len(samples)} (data index {args.start}-{args.start + len(samples) - 1})")

    # Resolve image paths
    valid_samples = []
    for s in samples:
        img_path = os.path.join(args.png_dir, s["imgname"])
        if os.path.exists(img_path):
            s["_image_path"] = img_path
            valid_samples.append(s)
    print(f"Valid images: {len(valid_samples)}/{len(samples)}")

    # Resume
    end = args.start + args.n
    output_path = os.path.join(
        args.output_dir, f"solver_sft_raw_{args.start}-{end}.json")
    results = []
    done_ids = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ids = set(r["id"] for r in results)
        print(f"Resume: {len(results)} done")

    todo = []
    for idx, s in enumerate(valid_samples):
        sid = f"vico_{args.start + idx}"
        if sid not in done_ids:
            todo.append((sid, s))
    print(f"TODO: {len(todo)} samples")
    if not todo:
        print("All done!")
        return

    # Build API client
    client_kwargs = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    print(f"Using model: {args.model}")
    print(f"Base URL: {client.base_url}")
    print(f"temperature={args.temperature}, max_tokens={args.max_tokens}")

    # Process
    t0 = time.time()
    new_count = 0

    try:
        for sid, s in tqdm(todo, desc="Generating"):
            r = generate_one(
                client, args.model,
                sid, s.get("query", ""), s["_image_path"],
                str(s.get("label", "")),
                args.temperature, args.max_tokens,
            )
            results.append(r)
            new_count += 1

            # Stats
            correct = sum(1 for r in results if r.get("correct"))
            total_done = len(results)

            if new_count % args.save_every == 0:
                atomic_json_save(results, output_path)
                elapsed = time.time() - t0
                speed = (new_count / elapsed * 60) if elapsed > 0 else 0
                tqdm.write(
                    f"  [{new_count}/{len(todo)}] "
                    f"correct={correct}/{total_done} ({correct/total_done*100:.0f}%), "
                    f"speed={speed:.1f} samples/min, "
                    f"saved to {os.path.basename(output_path)}")
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving partial results before exit...")
    finally:
        # Always save — covers normal finish, Ctrl+C, and uncaught errors.
        atomic_json_save(results, output_path)
        print(f"Saved {len(results)} results to {output_path}")

    # Summary
    correct = sum(1 for r in results if r.get("correct"))
    failed = sum(1 for r in results if r.get("status") == "failed")
    refusal = sum(1 for r in results if r.get("refusal"))
    think_wcs = [r["think_word_count"] for r in results if r.get("think_word_count")]

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total: {len(results)}")
    print(f"  Correct: {correct} ({correct/len(results)*100:.0f}%)")
    print(f"  Failed: {failed}")
    print(f"  Refusal: {refusal}")
    if think_wcs:
        print(f"  Think words: mean={sum(think_wcs)/len(think_wcs):.0f}, "
              f"min={min(think_wcs)}, max={max(think_wcs)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()