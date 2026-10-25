#!/usr/bin/env python3
"""
MathVista evaluation using HuggingFace InternVL chat() helper instead of vLLM.

vLLM's internvl_chat backend produces gibberish on this particular merged
checkpoint while HF works fine; this script bypasses vLLM. Slower but correct.

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 eval_mathvista_internvl_hf.py \\
        --model_path /path/to/VICO/internvl_trained_model/chartqa_internvl3-8b_solver_merged \\
        --data_json  /path/to/dataset/mathvista/mathvista_testmini.json \\
        --image_root /path/to/dataset/mathvista \\
        --prompt_style answer_tag
"""

import os, json, argparse, time
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Reuse extraction + matching from the vLLM-based script
from eval_mathvista_internvl import (
    build_query, extract_answer, mathvista_match,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# InternVL dynamic-tile preprocessing (lifted from official inference code)
def find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = w * h
    for ratio in target_ratios:
        tar_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tar_ar)
        if diff < best_ratio_diff or (
            diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_ratio_diff = diff
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    w, h = image.size
    ar = w / h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    tar = find_closest_aspect_ratio(ar, target_ratios, w, h, image_size)
    target_w, target_h = image_size * tar[0], image_size * tar[1]
    blocks = tar[0] * tar[1]
    resized = image.resize((target_w, target_h))
    proc = []
    for i in range(blocks):
        box = (
            (i % (target_w // image_size)) * image_size,
            (i // (target_w // image_size)) * image_size,
            ((i % (target_w // image_size)) + 1) * image_size,
            ((i // (target_w // image_size)) + 1) * image_size,
        )
        proc.append(resized.crop(box))
    if use_thumbnail and len(proc) > 1:
        proc.append(image.resize((image_size, image_size)))
    return proc


def load_image(path, input_size=448, max_num=6):
    image = Image.open(path).convert("RGB")
    tfm = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    return torch.stack([tfm(t) for t in tiles])  # [N, 3, H, W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt_style", default="answer_tag",
                        choices=["boxed", "answer_tag", "rethinker"])
    parser.add_argument("--max_num_tiles", type=int, default=6)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    if args.output is None:
        tag = Path(args.model_path).name
        args.output = f"eval_mathvista_{tag}_hf_results.json"
    print(f"model: {args.model_path}")
    print(f"out  : {args.output}")
    print(f"prompt_style: {args.prompt_style}")

    # ---- data ----
    with open(args.data_json) as f:
        data = json.load(f)
    if args.n > 0:
        data = data[:args.n]
    valid = []
    for s in data:
        img_path = os.path.join(args.image_root, s["image_path"])
        if os.path.exists(img_path):
            s["_full_image_path"] = img_path
            valid.append(s)
    print(f"Valid: {len(valid)}/{len(data)}")

    # ---- resume ----
    results, correct, total, done_ids = [], 0, 0, set()
    if args.resume and os.path.exists(args.output):
        try:
            prev = json.load(open(args.output))
            results = prev.get("results", [])
            done_ids = {str(r["id"]) for r in results}
            correct = sum(int(r.get("correct", 0)) for r in results)
            total = len(results)
            valid = [s for s in valid if str(s["id"]) not in done_ids]
            print(f"[resume] {total} done; {len(valid)} remaining")
        except Exception as e:
            print(f"[resume] failed: {e}; starting fresh")

    # ---- model ----
    print("Loading model with HF...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    # use flash-attn if available (much faster generation)
    try:
        import flash_attn  # noqa: F401
        attn_kw = {"attn_implementation": "flash_attention_2"}
        print("flash-attn detected, enabling flash_attention_2")
    except ImportError:
        attn_kw = {}
        print("flash-attn NOT available; using default (eager) attention — slow")

    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
        **attn_kw,
    ).cuda().eval()
    print("Model loaded.")

    SYSTEM_PROMPT = None
    if args.prompt_style == "answer_tag":
        SYSTEM_PROMPT = (
            "You are a helpful assistant. When the user asks a question, your response must include two parts: "
            "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. "
            "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
        )
    if SYSTEM_PROMPT:
        model.system_message = SYSTEM_PROMPT
    elif args.prompt_style == "rethinker":
        model.system_message = ""

    gen_cfg = dict(
        max_new_tokens=args.max_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature if args.temperature > 0 else None,
        pad_token_id=tok.eos_token_id,
    )
    gen_cfg = {k: v for k, v in gen_cfg.items() if v is not None}

    t0 = time.time()
    for s in tqdm(valid, desc="Evaluating"):
        try:
            pixel = load_image(s["_full_image_path"], max_num=args.max_num_tiles).to(torch.bfloat16).cuda()
            query = build_query(s)
            if args.prompt_style == "rethinker":
                query = query + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
            full_q = "<image>\n" + query
            response = model.chat(tok, pixel, full_q, gen_cfg)
        except Exception as e:
            response = f"[ERROR: {e}]"

        gt = str(s["answer"])
        pred_raw = extract_answer(response, args.prompt_style)
        match, pred_norm = mathvista_match(pred_raw, gt, s)
        correct += int(match)
        total += 1
        results.append({
            "id": s["id"],
            "task": s.get("task", ""),
            "question_type": s.get("metadata", {}).get("question_type", ""),
            "answer_type": s.get("metadata", {}).get("answer_type", ""),
            "question": s["question"],
            "gt": gt,
            "pred_raw": pred_raw,
            "pred_normalized": pred_norm,
            "correct": match,
            "choices": s.get("choices", []),
            "response": response,
        })
        if total % 5 == 0:
            tqdm.write(f"  [{total}] acc={correct/total*100:.1f}%")

        # incremental save
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "model": args.model_path,
                "prompt_style": args.prompt_style,
                "total": total,
                "correct": correct,
                "accuracy": round(correct / total * 100, 2) if total else 0.0,
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        os.replace(tmp, args.output)

    elapsed = time.time() - t0
    print(f"\nDone. {correct}/{total} = {correct/total*100:.2f}%   ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
