#!/usr/bin/env python3
"""
Generate ChartQA Solver SFT data with Qwen3-VL-235B-A22B-Thinking (local inference).

Adapted from generate_solver_sft_data-qwen3vl235b.py (CLEVR version).
Uses Qwen3-VL-235B to look at chart images and generate detailed reasoning chains.

Usage:
    python3 generate_solver_sft_data_chartqa_qwen3vl235b.py \
        --data_json /path/to/dataset/chartqa/train/train_augmented.json \
        --png_dir /path/to/dataset/chartqa/train/png \
        --start 0 --n 2000 \
        --offset 8000 \
        --output_dir chartqa_solver_sft_235b \
        --model_path /path/to/models/Qwen3-VL-235B-A22B-Thinking \
        --resume
"""

import os, json, argparse, random, re, time
from typing import Dict, Optional

import torch
from PIL import Image as PILImage
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
#  Configuration
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
#  Helpers
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
        # 5% tolerance for ChartQA (chart values are approximate)
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
#  Model loading
# ═══════════════════════════════════════════════════════════════════

def load_qwen_model(model_path):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"Loading Qwen3-VL from {model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("Qwen3-VL ready")
    return model, processor


# ═══════════════════════════════════════════════════════════════════
#  Generation
# ═══════════════════════════════════════════════════════════════════

def generate_one(model, processor, sid, question: str,
                 image_path: str, answer: str,
                 temperature: float, max_tokens: int) -> Dict:
    from qwen_vl_utils import process_vision_info

    if not os.path.exists(image_path):
        return {
            "id": sid, "question": question, "answer": answer,
            "image_path": image_path,
            "status": "failed", "error": "image not found",
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
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": user_content},
        ]},
    ]

    for attempt in range(3):
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature + attempt * 0.1,
                    do_sample=True,
                )
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            content = processor.batch_decode(
                generated_ids, skip_special_tokens=True)[0]

            # Handle thinking model output format
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
                if attempt < 2:
                    continue

        except Exception as e:
            if attempt == 2:
                return {
                    "id": sid, "question": question, "answer": answer,
                    "image_path": image_path,
                    "status": "failed", "error": str(e)[:200],
                }

    return {
        "id": sid, "question": question, "answer": answer,
        "image_path": image_path, "status": "failed",
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate ChartQA Solver SFT data with Qwen3-VL-235B-Thinking")
    parser.add_argument("--data_json", required=True, help="ChartQA train JSON")
    parser.add_argument("--png_dir", required=True, help="Directory with chart PNGs")
    parser.add_argument("--start", type=int, default=8000,
                        help="Start index into data_json")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of samples to process")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--max_tokens", type=int, default=8096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--min_think_words", type=int, default=300)
    parser.add_argument("--save_every", type=int, default=10)
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
        sid = f"chartqa_{args.start + idx}"
        if sid not in done_ids:
            todo.append((sid, s))
    print(f"TODO: {len(todo)} samples")
    if not todo:
        print("All done!")
        return

    # Load model
    model, processor = load_qwen_model(args.model_path)
    print(f"\ntemperature={args.temperature}, max_tokens={args.max_tokens}")

    # Process
    t0 = time.time()
    new_count = 0

    for sid, s in tqdm(todo, desc="Generating"):
        r = generate_one(
            model, processor,
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
            speed = new_count / elapsed if elapsed > 0 else 0
            tqdm.write(
                f"  [{new_count}/{len(todo)}] "
                f"correct={correct}/{total_done} ({correct/total_done*100:.0f}%), "
                f"speed={speed:.1f} samples/min, "
                f"saved to {os.path.basename(output_path)}")

    # Final save
    atomic_json_save(results, output_path)

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
