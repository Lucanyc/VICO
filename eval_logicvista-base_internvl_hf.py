#!/usr/bin/env python3
"""LogicVista evaluation via HF (bypasses vLLM)."""

import os, json, re, argparse, time
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from internvl_hf_utils import load_model, chat_single, make_gen_cfg
# the original logicvista filename has a hyphen — can't import; pull from
# eval_mathvista which the original also re-uses.
from eval_mathvista import extract_boxed


DEFAULT_DATA_ROOT = "/path/to/dataset/LogicVista_json"
DEFAULT_TEST_JSON = os.path.join(DEFAULT_DATA_ROOT, "test.json")
BOXED_INSTRUCTION = "Please answer the question and put the final answer to \\boxed{}"

RETHINKER_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)
ANSWER_TAG_SYS = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. "
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def extract_letter_set(text):
    if not text:
        return set()
    raw = re.findall(r'\b([A-Ia-i])\b|\(([A-Ia-i])\)', text)
    return {(m[0] or m[1]).upper() for m in raw if (m[0] or m[1])}


def extract_answer_set(text):
    # Try <answer>...</answer> first (answer_tag prompt style)
    am = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", text)
    if am:
        s = extract_letter_set(am.group(1))
        if s:
            return s
    boxed = extract_boxed(text)
    if boxed:
        s = extract_letter_set(boxed)
        if s:
            return s
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        s = extract_letter_set(line)
        if s:
            return s
        break
    all_l = list(re.finditer(r'\b([A-Ia-i])\b', text))
    if all_l:
        return {all_l[-1].group(1).upper()}
    return set()


def normalize_gt(ans):
    return extract_letter_set(str(ans))


def atomic_save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--test_json", default=DEFAULT_TEST_JSON)
    p.add_argument("--image_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--output", default=None)
    p.add_argument("--n", type=int, default=-1)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_num_tiles", type=int, default=6)
    p.add_argument("--prompt_style", default="boxed",
                   choices=["boxed", "rethinker", "answer_tag"],
                   help="boxed: bare 'put final in boxed' (single-letter outputs); "
                        "rethinker: prepend reasoning request, still extract via boxed; "
                        "answer_tag: <think>/<answer> system prompt (training format).")
    p.add_argument("--max_tokens_override", type=int, default=None,
                   help="default boxed=2048 / rethinker=8192 / answer_tag=8192")
    args = p.parse_args()
    if args.max_tokens_override is not None:
        args.max_tokens = args.max_tokens_override
    elif args.prompt_style != "boxed" and args.max_tokens < 4096:
        args.max_tokens = 8192

    if args.output is None:
        suffix = "" if args.prompt_style == "boxed" else f"_{args.prompt_style}"
        args.output = f"eval_logicvista_base_{Path(args.model_path).name}{suffix}_hf.json"
    print(f"out: {args.output}  (prompt_style={args.prompt_style}, max_tokens={args.max_tokens})")

    data = json.load(open(args.test_json))
    if args.n > 0:
        data = data[:args.n]
    samples = []
    for s in data:
        ip = os.path.join(args.image_root, s["image_path"])
        if os.path.exists(ip):
            samples.append({**s, "image_path": ip})
    print(f"Valid: {len(samples)}/{len(data)}")

    results, correct, total, done_ids = [], 0, 0, set()
    if os.path.exists(args.output):
        try:
            prev = json.load(open(args.output))
            results = prev.get("results", [])
            done_ids = {r["id"] for r in results if r.get("id") is not None}
            correct = sum(int(r.get("correct", 0)) for r in results)
            total = len(results)
            print(f"[resume] {total} done")
        except Exception:
            pass
    samples = [s for s in samples if s["id"] not in done_ids]

    sys_prompt = ANSWER_TAG_SYS if args.prompt_style == "answer_tag" else None
    tok, model = load_model(args.model_path, system_prompt=sys_prompt)
    gen_cfg = make_gen_cfg(tok, args.max_tokens, args.temperature)

    t0 = time.time()
    for s in tqdm(samples, desc="LogicVista"):
        if args.prompt_style == "rethinker":
            query = f"{s['question']}{RETHINKER_SUFFIX}"
        elif args.prompt_style == "answer_tag":
            query = s["question"]
        else:  # boxed
            query = f"{s['question']}\n\n{BOXED_INSTRUCTION}"
        try:
            resp = chat_single(model, tok, s["image_path"], query, gen_cfg, args.max_num_tiles)
        except Exception as e:
            resp = f"[ERROR: {e}]"
        pred_set = extract_answer_set(resp)
        gt_set = normalize_gt(s["answer"])
        match = (len(pred_set) > 0) and (pred_set == gt_set)
        correct += int(match)
        total += 1
        results.append({
            "id": s["id"], "question": s["question"],
            "skill": s.get("skill", []),
            "broad_capability": s.get("broad_capability", []),
            "gt": str(s["answer"]),
            "gt_set": sorted(gt_set), "pred_set": sorted(pred_set),
            "correct": match, "response": resp,
        })
        if total % 5 == 0:
            tqdm.write(f"  [{total}] acc={correct/total*100:.1f}%")
        skill_c, skill_t = defaultdict(int), defaultdict(int)
        for r in results:
            for sk in r.get("skill") or ["<none>"]:
                skill_t[sk] += 1
                skill_c[sk] += int(r.get("correct", False))
        atomic_save_json(args.output, {
            "model": args.model_path, "prompt": BOXED_INSTRUCTION,
            "total": total, "correct": correct,
            "accuracy": round(correct / total * 100, 2) if total else 0.0,
            "per_skill": {sk: {"correct": skill_c[sk], "total": skill_t[sk],
                               "accuracy": round(skill_c[sk] / skill_t[sk] * 100, 2)}
                          for sk in sorted(skill_t)},
            "time_s": round(time.time() - t0, 1), "results": results,
        })
    print(f"\nDone. {correct}/{total} = {correct/total*100:.2f}%")


if __name__ == "__main__":
    main()
