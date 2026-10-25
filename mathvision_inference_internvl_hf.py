#!/usr/bin/env python3
"""MathVision evaluation via HF (bypasses vLLM)."""

import os, json, argparse, time
from pathlib import Path
from tqdm import tqdm

from internvl_hf_utils import load_model, chat_single, make_gen_cfg
from mathvision_inference_internvl import extract_answer, check_answer, SYSTEM_PROMPT

RETHINKER_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)
BOXED_INSTRUCTION = "Please answer the question and put the final answer to \\boxed{}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument("--image_base_dir", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_num_tiles", type=int, default=6)
    p.add_argument("--prompt_style", default="answer_tag",
                   choices=["answer_tag", "rethinker", "boxed"],
                   help="answer_tag: <think>/<answer> system prompt (training format, default); "
                        "rethinker: no system prompt, append 'reason step by step + boxed'; "
                        "boxed: no system prompt, append 'put final in boxed'.")
    args = p.parse_args()

    if args.image_base_dir is None:
        args.image_base_dir = os.path.dirname(os.path.abspath(args.test_file))
    if args.output is None:
        suffix = "" if args.prompt_style == "answer_tag" else f"_{args.prompt_style}"
        args.output = f"mathvision_eval_{Path(args.model).name}{suffix}_hf.json"
    print(f"out: {args.output}  (prompt_style={args.prompt_style})")

    data = json.load(open(args.test_file))
    if args.max_samples > 0:
        data = data[:args.max_samples]
    samples = []
    for s in data:
        ip = os.path.join(args.image_base_dir, s['image'])
        if os.path.exists(ip):
            s['_image_path'] = ip
            samples.append(s)
    print(f"Valid: {len(samples)}")

    results, correct, total, done_ids = [], 0, 0, set()
    if os.path.exists(args.output):
        try:
            prev = json.load(open(args.output))
            results = prev.get("results", [])
            done_ids = {r["id"] for r in results}
            correct = sum(int(r.get("correct", 0)) for r in results)
            total = len(results)
            print(f"[resume] {total} done")
        except Exception:
            pass
    samples = [s for s in samples if s["id"] not in done_ids]

    sys_prompt = SYSTEM_PROMPT if args.prompt_style == "answer_tag" else None
    tok, model = load_model(args.model, system_prompt=sys_prompt)
    gen_cfg = make_gen_cfg(tok, args.max_new_tokens, args.temperature)

    t0 = time.time()
    for s in tqdm(samples, desc="MathVision"):
        q = s['question']
        if s.get('options'):
            q += "\n\nOptions:\n" + "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(s['options']))
        if args.prompt_style == "rethinker":
            q += RETHINKER_SUFFIX
        elif args.prompt_style == "boxed":
            q += "\n\n" + BOXED_INSTRUCTION
        try:
            resp = chat_single(model, tok, s['_image_path'], q, gen_cfg, args.max_num_tiles)
        except Exception as e:
            resp = f"[ERROR: {e}]"
        pred = extract_answer(resp)
        gt = str(s['answer'])
        ok = check_answer(pred, gt, s.get('options', []))
        correct += int(ok)
        total += 1
        has_fmt = ("<think>" in resp and "</think>" in resp and
                   "<answer>" in resp and "</answer>" in resp)
        results.append({
            "id": s['id'], "question": s['question'][:200],
            "answer": gt, "prediction": pred, "correct": ok,
            "format_ok": has_fmt, "response": resp,
            "response_len": len(resp),
            "subject": s.get('subject', 'unknown'),
            "level": s.get('level', ''),
            "has_options": bool(s.get('options', [])),
        })
        if total % 5 == 0:
            tqdm.write(f"  [{total}] acc={correct/total*100:.1f}%")
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"model": args.model, "test_file": args.test_file,
                      "accuracy_all": correct/total*100, "correct": correct, "total": total,
                      "results": results, "_partial": True}, f, indent=2, ensure_ascii=False)
        os.replace(tmp, args.output)

    print(f"\nDone. {correct}/{total} = {correct/total*100:.2f}%")


if __name__ == "__main__":
    main()
