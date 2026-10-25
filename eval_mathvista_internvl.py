#!/usr/bin/env python3
"""
Evaluate on MathVista testmini — InternVL adapted.

Adds InternVL3-8B support via the MODEL_ALIASES / resolve_model /
mm_kwargs_for helpers (see evaluate.py for the canonical pattern).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 eval_mathvista_internvl.py \\
        --model_path internvl3-8b \\
        --data_json /path/to/dataset/mathvista/mathvista_testmini.json \\
        --image_root /path/to/dataset/mathvista \\
        --tp 1 --prompt_style boxed \\
        --output eval_mathvista_internvl3-8b_results.json
"""

import os, json, re, argparse, time
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance

try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False
    print("[WARN] math_verify not installed, falling back to numeric tolerance. Install: pip install math_verify")


MODEL_ALIASES = {
    "qwen3vl8b":    "/path/to/models/Qwen3-VL-8B-Instruct",
    "qwen2.5vl7b":  "/path/to/models/Qwen2.5-VL-7B-Instruct",
    "internvl3-8b": "/path/to/models/InternVL3-8B-Instruct",
}


def resolve_model(arg: str) -> tuple[str, str]:
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
    name = Path(model_path).name.lower()
    if "qwen" in name and "vl" in name:
        return {
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }
    return {}


def build_hint(question_type, answer_type, precision):
    if question_type == "multi_choice":
        return "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
    if answer_type == "integer":
        return "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."
    if answer_type == "float":
        if precision == 1:
            return "Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."
        elif precision == 2:
            return "Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."
        else:
            return "Hint: Please answer the question requiring a floating-point number and provide the final value at the end."
    if answer_type == "list":
        return "Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."
    return ""


def build_query(sample):
    meta = sample.get("metadata", {})
    question_type = meta.get("question_type", "free_form")
    answer_type = meta.get("answer_type", "text")
    precision = meta.get("precision")
    unit = meta.get("unit")
    choices = sample.get("choices", [])

    question_text = f"Question: {sample['question']}"
    if unit:
        question_text += f" (Unit: {unit})"

    choices_text = ""
    if choices:
        lines = ["Choices:"]
        for i, c in enumerate(choices):
            lines.append(f"({chr(ord('A') + i)}) {c}")
        choices_text = "\n".join(lines)

    hint_text = build_hint(question_type, answer_type, precision)

    elements = [question_text, choices_text, hint_text]
    return "\n".join(e for e in elements if e)


def extract_boxed(text):
    boxed_start = "\\boxed{"
    positions = []
    pos = 0
    while True:
        pos = text.find(boxed_start, pos)
        if pos == -1:
            break
        positions.append(pos)
        pos += 1
    results = []
    for sp in positions:
        depth, i = 0, sp + len(boxed_start) - 1
        while i < len(text):
            if text[i] == '{': depth += 1
            elif text[i] == '}': depth -= 1
            if depth == 0:
                results.append(text[sp + len(boxed_start):i])
                break
            i += 1
    return results[-1].strip() if results else None


def extract_answer_tag(text):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_answer(text, prompt_style="boxed"):
    if prompt_style == "answer_tag":
        ans = extract_answer_tag(text)
        if ans: return ans
        ans = extract_boxed(text)
        if ans: return ans
    else:
        ans = extract_boxed(text)
        if ans: return ans
        ans = extract_answer_tag(text)
        if ans: return ans

    for pat in [
        r'(?:the |my |final )?answer is[:\s]*([^\n.;]+)',
        r'(?:答案|结果)[是为：:\s]*([^\n.;。]+)',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    lines = text.strip().split("\n")
    for line in reversed(lines):
        if line.strip():
            m = re.search(r'\b(\d+(?:\.\d+)?)\b(?:\s*\.?\s*$)', line)
            if m:
                return m.group(1)

    return None


def parse_mcq(text):
    if not text or not text.strip():
        return ""

    response = text.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(char)
    response = " " + response + " "

    all_choices = ["A", "B", "C", "D", "E", "F", "G", "H"]
    candidates = []

    for fmt, suffix in [("paren", lambda c: f"({c})"), ("period", lambda c: f"{c}."),
                        ("colon", lambda c: f"{c}:"), ("rparen", lambda c: f"{c})"),
                        ("space", lambda c: f"{c} ")]:
        for choice in all_choices:
            pat = suffix(choice)
            if pat in response:
                candidates.append((choice, response.rfind(pat), fmt))

    answer_phrases = [
        "the answer is", "answer is", "the correct answer is", "correct answer is",
        "i choose", "i select", "my answer is",
    ]
    for phrase in answer_phrases:
        if phrase in response.lower():
            idx = response.lower().find(phrase)
            for choice in all_choices:
                cpos = response.find(choice, idx)
                if cpos != -1:
                    candidates.append((choice, cpos, "phrase"))

    for choice in all_choices:
        if response.strip().startswith(choice):
            candidates.append((choice, 0, "start"))
        if response.strip().endswith(choice):
            candidates.append((choice, len(response) - 1, "end"))

    if not candidates:
        for choice in all_choices:
            if choice in response:
                candidates.append((choice, response.rfind(choice), "fallback"))

    if candidates:
        priority = {"start": 10, "end": 9, "phrase": 7, "paren": 6, "period": 5,
                     "colon": 4, "rparen": 3, "space": 2, "fallback": 0}
        candidates.sort(key=lambda x: (priority.get(x[2], 0), -x[1]), reverse=True)
        return candidates[0][0]

    return ""


def normalize_answer(extraction, choices, question_type, answer_type, precision):
    if extraction is None:
        return None

    extraction = str(extraction).strip()

    if question_type == "multi_choice":
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if letter:
            extraction = letter[0].upper()

        options = [chr(ord('A') + i) for i in range(len(choices))]
        if extraction.upper() in options:
            idx = options.index(extraction.upper())
            return choices[idx]
        else:
            if choices:
                distances = [levenshtein_distance(extraction.lower(), str(c).lower()) for c in choices]
                return choices[distances.index(min(distances))]
            return extraction

    elif answer_type == "integer":
        try:
            return str(int(float(extraction)))
        except (ValueError, TypeError):
            return None

    elif answer_type == "float":
        try:
            p = precision if precision else 2
            return str(round(float(extraction), p))
        except (ValueError, TypeError):
            return None

    elif answer_type == "list":
        return str(extraction)

    return extraction


def safe_equal(prediction, answer):
    if prediction is None or answer is None:
        return False
    return str(prediction).strip() == str(answer).strip()


def mathvista_match(pred_raw, gt, sample):
    if pred_raw is None:
        return False, None

    meta = sample.get("metadata", {})
    question_type = meta.get("question_type", "free_form")
    answer_type = meta.get("answer_type", "text")
    precision = meta.get("precision")
    choices = sample.get("choices", [])

    if question_type == "multi_choice" and choices:
        mcq_letter = parse_mcq(pred_raw)
        if mcq_letter:
            pred_raw = mcq_letter

    prediction = normalize_answer(pred_raw, choices, question_type, answer_type, precision)
    gt_normalized = str(gt).strip()

    if safe_equal(prediction, gt_normalized):
        return True, prediction

    if HAS_MATH_VERIFY and prediction is not None:
        try:
            gold = parse(gt_normalized)
            pred = parse(str(prediction))
            if verify(gold, pred):
                return True, prediction
        except Exception:
            pass

    if prediction is not None:
        try:
            p = float(str(prediction).replace(',', '').replace('%', '').strip())
            g = float(str(gt_normalized).replace(',', '').replace('%', '').strip())
            if g == 0:
                if abs(p) < 1e-6:
                    return True, prediction
            elif abs(p - g) <= abs(g) * 0.05 + 1e-9:
                return True, prediction
        except (ValueError, TypeError):
            pass

    return False, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help=f"alias ({' | '.join(MODEL_ALIASES)}) or local model path")
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--prompt_style", default="boxed", choices=["boxed", "answer_tag", "rethinker"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_path, model_tag = resolve_model(args.model_path)
    args.model_path = model_path
    if args.output is None:
        args.output = f"eval_mathvista_{model_tag}_results.json"
    print(f"model: {model_path}  (tag={model_tag})")
    print(f"out  : {args.output}")

    with open(args.data_json) as f:
        data = json.load(f)
    if args.n > 0:
        data = data[:args.n]
    print(f"Loaded {len(data)} samples")

    valid = []
    for s in data:
        img_path = os.path.join(args.image_root, s["image_path"])
        if os.path.exists(img_path):
            s["_full_image_path"] = img_path
            valid.append(s)
    print(f"Valid samples: {len(valid)}")

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
            print(f"[Resume] Loaded {total} finished samples (correct={correct}); "
                  f"remaining {len(valid)}/{before}")
        except Exception as e:
            print(f"[Resume] Failed to load {args.output}: {e}. Starting fresh.")
            results, correct, total = [], 0, 0

    if not valid:
        print("[Resume] Nothing to do — all samples already completed.")

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

    resized_dir = os.path.join(args.image_root, "_resized_tmp")
    os.makedirs(resized_dir, exist_ok=True)
    MAX_PIXELS = 1536 * 1536

    from PIL import Image as PILImage
    resized_count = 0
    for s in valid:
        img_path = s["_full_image_path"]
        try:
            with PILImage.open(img_path) as img:
                w, h = img.size
                if w * h > MAX_PIXELS:
                    scale = (MAX_PIXELS / (w * h)) ** 0.5
                    new_w, new_h = int(w * scale), int(h * scale)
                    resized_path = os.path.join(resized_dir, os.path.basename(img_path))
                    img_resized = img.resize((new_w, new_h), PILImage.LANCZOS)
                    img_resized.save(resized_path)
                    s["_full_image_path"] = resized_path
                    resized_count += 1
        except Exception:
            pass
    if resized_count > 0:
        print(f"Resized {resized_count} large images (>{MAX_PIXELS} pixels)")

    for i in tqdm(range(0, len(valid), args.batch_size), desc="Evaluating"):
        batch = valid[i:i + args.batch_size]
        messages_batch = []
        for s in batch:
            query = build_query(s)

            if args.prompt_style == "answer_tag":
                sys_prompt = (
                    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
                    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. "
                    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
                )
            else:
                sys_prompt = None

            if args.prompt_style == "rethinker":
                query = query + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

            content = [
                {"type": "image_url", "image_url": {
                    "url": f"file://{os.path.realpath(s['_full_image_path'])}"}},
                {"type": "text", "text": query},
            ]
            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": content})
            messages_batch.append(messages)

        outputs = llm.chat(messages_batch, sampling_params=sampling_params,
                           chat_template_content_format="string")

        for s, out in zip(batch, outputs):
            resp = out.outputs[0].text
            pred_raw = extract_answer(resp, args.prompt_style)
            gt = str(s["answer"])
            match, pred_normalized = mathvista_match(pred_raw, gt, s)
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
                "pred_normalized": pred_normalized,
                "correct": match,
                "choices": s.get("choices", []),
                "response": resp,
            })

        acc = correct / total * 100
        tqdm.write(f"  [{total}] acc={acc:.1f}%")

        tmp_path = args.output + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({
                "model": args.model_path,
                "prompt_style": args.prompt_style,
                "total": total,
                "correct": correct,
                "accuracy": round(acc, 2),
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, args.output)

    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    qtype_correct = defaultdict(int)
    qtype_total = defaultdict(int)

    for r in results:
        task_correct[r["task"]] += int(r["correct"])
        task_total[r["task"]] += 1
        qtype_correct[r["question_type"]] += int(r["correct"])
        qtype_total[r["question_type"]] += 1

    elapsed = time.time() - t0
    acc = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"MathVista Evaluation Results")
    print(f"  Model: {args.model_path}")
    print(f"  Prompt style: {args.prompt_style}")
    print(f"  Samples: {total}")
    print(f"  Overall Accuracy: {correct}/{total} = {acc:.1f}%")
    print(f"\n  Per question type:")
    for qt in sorted(qtype_total.keys()):
        qc = qtype_correct[qt]
        qt_tot = qtype_total[qt]
        print(f"    {qt}: {qc}/{qt_tot} = {qc/qt_tot*100:.1f}%")
    print(f"\n  Per task:")
    for task in sorted(task_total.keys()):
        tc = task_correct[task]
        tt = task_total[task]
        print(f"    {task}: {tc}/{tt} = {tc/tt*100:.1f}%")
    print(f"\n  Time: {elapsed:.0f}s ({elapsed/total:.1f}s/sample)")
    print(f"{'='*60}")

    none_preds = sum(1 for r in results if r["pred_raw"] is None)
    print(f"  Extraction failures (pred=None): {none_preds}/{total}")

    summary = {
        "model": args.model_path,
        "prompt_style": args.prompt_style,
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 2),
        "per_question_type": {qt: {"correct": qtype_correct[qt], "total": qtype_total[qt],
                                    "accuracy": round(qtype_correct[qt]/qtype_total[qt]*100, 2)}
                              for qt in sorted(qtype_total.keys())},
        "per_task": {t: {"correct": task_correct[t], "total": task_total[t],
                         "accuracy": round(task_correct[t]/task_total[t]*100, 2)}
                     for t in sorted(task_total.keys())},
        "time_s": round(elapsed, 1),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
