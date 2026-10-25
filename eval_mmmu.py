#!/usr/bin/env python3
"""
Evaluate on MMMU validation/test split.

Supports two evaluation modes:
  1. Rule-based (default): lmms-eval style exact/substring matching.
  2. LLM-as-a-Judge (--use_llm_judge): Uses an OpenAI-compatible model (e.g. GPT-5-4)
     to judge whether the model's answer is semantically correct.

LLM Judge usage:
    # With OpenAI API
    CUDA_VISIBLE_DEVICES=0,1 python3 eval_mmmu.py \
        --model_path Qwen/Qwen3-VL-8B-Instruct \
        --tp 2 --prompt_style qwen3_vl --split test \
        --use_llm_judge \
        --judge_model gpt-5-4 \
        --judge_api_key $OPENAI_API_KEY \
        --output eval_mmmu_judge.json

    # With local vLLM-served judge (e.g. Qwen2.5-14B-Instruct)
    CUDA_VISIBLE_DEVICES=0,1 python3 eval_mmmu.py \
        --model_path Qwen/Qwen3-VL-8B-Instruct \
        --tp 2 --prompt_style qwen3_vl --split test \
        --use_llm_judge \
        --judge_model Qwen2.5-14B-Instruct \
        --judge_api_base http://localhost:8000/v1 \
        --judge_api_key EMPTY \
        --output eval_mmmu_judge.json

Prompt styles:
  * default    — lmms-eval `mmmu_val` default format (best for Qwen2.5-VL).
  * qwen3_vl   — lmms-eval `mmmu_val_qwen` format (best for Qwen3-VL Instruct).
  * rethinker  — default prompt + boxed reasoning (for VL-Rethinker-7B).
  * thinklite  — default prompt + ThinkLite-VL trigger.
  * answer_tag — default prompt + <think></think><answer></answer> system.
"""

import os, json, re, ast, random, argparse, time
import numpy as np
from collections import defaultdict
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image as PILImage
import concurrent.futures


# ══════════════════════════════════════════════════════════════════════
# LLM-as-a-Judge
# ══════════════════════════════════════════════════════════════════════

# Judge prompt — two modes available:
# 1. "mmzero" mode: exact prompt from MM-Zero llm_judge_eval.py (for fair comparison)
# 2. "lenient" mode: more generous judging (our default)

JUDGE_SYSTEM_MMZERO = (
    "You are an answer correctness judge. Given a question, the gold (correct) answer, "
    "and the model's answer, determine if the model's answer is correct: equivalent to the "
    "gold answer or semantically the same. Consider numeric equality (e.g. 14 vs 14.0), "
    "option equivalence (A vs A.), and paraphrases. Answer with exactly one word: Yes or No."
)

JUDGE_SYSTEM_LENIENT = (
    "You are a generous answer-correctness judge. Your job is to decide whether "
    "a student's answer is essentially correct, even if it is worded differently "
    "from the gold answer.\n\n"
    "## IMPORTANT: When in doubt, lean toward YES.\n\n"
    "## Accept ALL of the following as correct:\n"
    "- Same meaning in different words: 'increased' == 'went up' == 'rose'\n"
    "- Numeric equivalence with ANY rounding: 14 == 14.0 == ~14 == 'about 14' == 13.97\n"
    "- Accept rounding differences up to 5% of the gold value\n"
    "- Option letter in any format: A == (A) == A. == a == 'Option A' == the full text of option A\n"
    "- If the model selected the correct option's CONTENT even without stating the letter, mark correct\n"
    "- Units present or absent: '5 m' == '5 meters' == '5'\n"
    "- Extra units that don't contradict: '100 kPa' == '100'\n"
    "- Fraction/decimal/percentage interchangeable: 1/2 == 0.5 == 50%\n"
    "- LaTeX vs plain text: '\\frac{1}{2}' == '1/2' == '0.5'\n"
    "- Scientific notation: '3e5' == '300000' == '3 × 10^5'\n"
    "- Sign/negative: '-3' == 'negative 3' == '−3'\n"
    "- Comma separators: '1,000' == '1000'\n"
    "- Ordinal/cardinal: 'first' == '1st' == '1'\n"
    "- Abbreviations and synonyms: 'approx' == 'approximately' == '~'\n"
    "- Minor spelling differences or typos in the answer\n"
    "- If the model's response contains the correct answer ANYWHERE in it (even buried in explanation), mark YES\n"
    "- Equivalent chemical/math/physics expressions\n"
    "- Correct reasoning that arrives at the right answer, even if phrased as a sentence\n\n"
    "## Mark NO only when:\n"
    "- The answer is clearly and unambiguously WRONG (different number, different option, contradictory meaning)\n"
    "- The answer is a completely unrelated response\n\n"
    "Answer with exactly one word: Yes or No."
)

LLM_JUDGE_USER_TEMPLATE = (
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Model answer: {model_answer}\n\n"
    "Is the model answer correct? Answer Yes or No."
)


def create_judge_client(api_base, api_key):
    """Create an OpenAI client for the judge model."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")

    kwargs = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def judge_single(client, model_name, question, gold_answer, model_answer,
                 options=None, question_type=None, max_retries=3, judge_style="mmzero"):
    """
    Call the LLM judge for a single sample.
    judge_style: "mmzero" uses MM-Zero's exact prompt, "lenient" uses our generous prompt.
    Returns: (is_correct: bool, judge_response: str)
    """
    # Select judge system prompt
    system_prompt = JUDGE_SYSTEM_MMZERO if judge_style == "mmzero" else JUDGE_SYSTEM_LENIENT

    # For MCQ, enrich the gold answer with full option text AND list all options
    # so the judge can understand what each letter means
    if question_type == "multiple-choice" and options:
        options_text = "\n".join(
            f"  {chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)
        )
        if len(gold_answer) == 1:
            idx = ord(gold_answer.upper()) - ord("A")
            if 0 <= idx < len(options):
                gold_display = f"{gold_answer}. {options[idx]}"
            else:
                gold_display = gold_answer
        else:
            gold_display = gold_answer
        question_display = f"{question}\nOptions:\n{options_text}"
    else:
        gold_display = gold_answer
        question_display = question

    user_msg = LLM_JUDGE_USER_TEMPLATE.format(
        question=question_display,
        gold=gold_display,
        model_answer=model_answer,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            judge_text = response.choices[0].message.content.strip().lower()
            is_correct = judge_text.startswith("yes")
            return is_correct, judge_text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: Judge API failed after {max_retries} retries: {e}")
                return None, f"ERROR: {e}"


def judge_batch(client, model_name, samples, results_list, max_workers=16, judge_style="mmzero"):
    """
    Judge a batch of samples in parallel using ThreadPoolExecutor.
    Updates results_list in-place with judge verdicts.
    """
    def _judge_one(idx):
        r = results_list[idx]
        s = samples[idx]

        # Send the full model response to the judge (truncated at 2000 chars
        # to avoid token limits but still generous)
        model_answer = r.get("response", "")
        if len(model_answer) > 2000:
            model_answer_for_judge = model_answer[:2000] + "\n[...truncated]"
        else:
            model_answer_for_judge = model_answer

        is_correct, judge_resp = judge_single(
            client, model_name,
            question=s["question"],
            gold_answer=s["answer"],
            model_answer=model_answer_for_judge,
            options=s.get("options_parsed"),
            question_type=s.get("question_type"),
            judge_style=judge_style,
        )
        return idx, is_correct, judge_resp

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_judge_one, i): i for i in range(len(results_list))}
        for future in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures), desc="LLM Judge"):
            idx, is_correct, judge_resp = future.result()
            if is_correct is not None:
                results_list[idx]["correct_judge"] = is_correct
                results_list[idx]["judge_response"] = judge_resp
            else:
                # Fallback to rule-based result
                results_list[idx]["correct_judge"] = results_list[idx]["correct"]
                results_list[idx]["judge_response"] = "FALLBACK_TO_RULE"


# ══════════════════════════════════════════════════════════════════════
# Prompt construction — aligned with lmms-eval MMMU
# ══════════════════════════════════════════════════════════════════════

def parse_options(options):
    """Render options list as 'A. xxx\\nB. xxx\\n...'. No trailing newline."""
    letters = [chr(ord("A") + i) for i in range(len(options))]
    return "\n".join(f"{l}. {o}" for l, o in zip(letters, options))


def build_query_default(sample):
    q = sample["question"]
    if sample["question_type"] == "multiple-choice":
        opts = parse_options(sample["options_parsed"])
        return f"{q}\n{opts}\n\nAnswer with the option's letter from the given choices directly."
    else:
        return f"{q}\n\nAnswer the question using a single word or phrase."


def build_query_qwen3vl(sample):
    q = sample["question"]
    if sample["question_type"] == "multiple-choice":
        opts = parse_options(sample["options_parsed"])
        return f"Question: {q}\nOptions:\n{opts}\nAnswer with the option letter only."
    else:
        return f"Question: {q}\n\nPlease answer the question directly."


# ---- MM-Zero prompt (from eval_generate.py) ----
# MM-Zero uses a UNIFIED prompt for ALL question types (MCQ + open).
# It always asks for step-by-step reasoning + \boxed{} answer.
# For MCQ, it includes options in the question text but the instruction
# is the same — answer in \boxed{}.
MMZERO_SOLVER_INSTRUCTION = (
    "Please reason step by step based on the question and image. "
    "Put your final answer inside \\boxed{} as a single number, single word, "
    "or short phrase only (e.g. \\boxed{42}, \\boxed{blue}, \\boxed{Q1})"
    "\u2014no units, no full sentences, so the grader can match it."
)


def build_query_mmzero(sample):
    """MM-Zero eval_generate.py prompt — unified for MCQ and open."""
    q = sample["question"]
    if sample["question_type"] == "multiple-choice":
        opts = parse_options(sample["options_parsed"])
        return f"{q}\n{opts}\n\n{MMZERO_SOLVER_INSTRUCTION}"
    else:
        return f"{q}\n\n{MMZERO_SOLVER_INSTRUCTION}"


RETHINKER_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

THINKLITE_SUFFIX = (
    r"You FIRST think about the reasoning process as an internal monologue "
    r"and then provide the final answer. The reasoning process MUST BE enclosed "
    r"within <think> </think> tags. The final answer MUST BE put in \boxed{}."
)

ANSWER_TAG_SYS = (
    "You are a helpful assistant. When the user asks a question, your response "
    "must include two parts: first, the reasoning process enclosed in "
    "<think>...</think> tags, then the final answer enclosed in "
    "<answer>...</answer> tags. Please provide a clear, concise response within "
    "<answer> </answer> tags that directly addresses the question."
)


def build_prompt(sample, style):
    if style == "qwen3_vl":
        return None, build_query_qwen3vl(sample)
    if style == "mmzero":
        return None, build_query_mmzero(sample)
    if style == "default":
        return None, build_query_default(sample)
    if style == "rethinker":
        return None, build_query_default(sample) + RETHINKER_SUFFIX
    if style == "thinklite":
        return None, build_query_default(sample) + THINKLITE_SUFFIX
    if style == "answer_tag":
        return ANSWER_TAG_SYS, build_query_default(sample)
    raise ValueError(f"unknown prompt_style: {style}")


# ══════════════════════════════════════════════════════════════════════
# Answer extraction — ported from lmms-eval / MMMU official eval_utils
# ══════════════════════════════════════════════════════════════════════

def extract_boxed(text):
    boxed_start = "\\boxed{"
    positions, pos = [], 0
    while True:
        pos = text.find(boxed_start, pos)
        if pos == -1:
            break
        positions.append(pos)
        pos += 1
    for sp in positions[::-1]:
        depth, i = 0, sp + len(boxed_start) - 1
        while i < len(text):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[sp + len(boxed_start):i].strip()
            i += 1
    return None


def extract_answer_tag(text):
    m = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", text)
    return m.group(1).strip() if m else None


def strip_thinking(text):
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE).strip()


def preprocess_response(text, prompt_style, split_think=False):
    if split_think:
        text = strip_thinking(text)
    if prompt_style == "answer_tag":
        tagged = extract_answer_tag(text)
        if tagged:
            return tagged
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    return text


# ---- MMMU official MCQ parsing ----------------------------------------

def get_multi_choice_info(options):
    all_choices, index2ans = [], {}
    for i, option in enumerate(options):
        choice = chr(ord("A") + i)
        index2ans[choice] = option
        all_choices.append(choice)
    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    for ch in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(ch)
    response = " " + response + " "

    index_ans, ans_with_brack, candidates = True, False, []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if not candidates:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if not candidates:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if not candidates and len(response.split()) > 5:
        for idx, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(idx)
                index_ans = False

    if not candidates:
        return random.choice(all_choices)

    if len(candidates) > 1:
        starts = []
        if index_ans:
            if ans_with_brack:
                starts = [response.rfind(f"({c})") for c in candidates]
            else:
                starts = [response.rfind(f" {c} ") for c in candidates]
        else:
            starts = [response.lower().rfind(index2ans[c].lower()) for c in candidates]
        return candidates[int(np.argmax(starts))]
    return candidates[0]


# ---- MMMU official open-ended parsing ---------------------------------

def extract_numbers(s):
    p_comma = r"-?\b\d{1,3}(?:,\d{3})+\b"
    p_sci = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    p_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    return re.findall(p_comma, s) + re.findall(p_sci, s) + re.findall(p_simple, s)


def check_is_number(s):
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(s):
    s = s.strip()
    if check_is_number(s):
        return [round(float(s.replace(",", "")), 2)]
    s = s.lower()
    if len(s) == 1:
        return [" " + s, s + " "]
    return [s]


def parse_open_response(response):
    def get_key_subresponses(resp):
        resp = resp.strip().strip(".").lower()
        subs = re.split(r"\.\s(?=[A-Z])|\n", resp)
        indicators = ["could be ", "so ", "is ", "thus ", "therefore ",
                      "final ", "answer ", "result "]
        keys = []
        for i, r in enumerate(subs):
            ind = indicators + (["="] if i == len(subs) - 1 else [])
            shortest = None
            for marker in ind:
                if marker in r:
                    cand = r.split(marker)[-1].strip()
                    if shortest is None or len(cand) < len(shortest):
                        shortest = cand
            if shortest and shortest not in [":", ",", ".", "!", "?", ";", "'"]:
                keys.append(shortest)
        return keys or [resp]

    keys = get_key_subresponses(response)
    pred_list = keys.copy()
    for r in keys:
        pred_list.extend(extract_numbers(r))

    normalized = []
    for p in pred_list:
        if isinstance(p, str):
            normalized.extend(normalize_str(p))
        else:
            normalized.append(p)

    out = []
    for p in normalized:
        if p not in out:
            out.append(p)
    return out


def eval_multi_choice(gold, pred):
    if isinstance(gold, list):
        return pred in gold
    return gold == pred


def eval_open(gold, preds):
    if isinstance(gold, list):
        norm_gold = []
        for g in gold:
            norm_gold.extend(normalize_str(g))
    else:
        norm_gold = normalize_str(gold)
    for p in preds:
        if isinstance(p, str):
            for g in norm_gold:
                if isinstance(g, str) and g in p:
                    return True
        else:
            if p in norm_gold:
                return True
    return False


def mmmu_match(response, sample, prompt_style, split_think):
    clean = preprocess_response(response, prompt_style, split_think=split_think)
    if sample["question_type"] == "multiple-choice":
        index2ans, all_choices = get_multi_choice_info(sample["options_parsed"])
        parsed = parse_multi_choice_response(clean, all_choices, index2ans)
        return eval_multi_choice(sample["answer"], parsed), parsed
    else:
        parsed = parse_open_response(clean)
        return eval_open(sample["answer"], parsed), (parsed[0] if parsed else None)


# ══════════════════════════════════════════════════════════════════════
# MMMU data loading
# ══════════════════════════════════════════════════════════════════════

def load_mmmu(split="validation", cache_dir=None, subjects=None, local_dir=None):
    ALL_SUBJECTS = [
        "Accounting", "Agriculture", "Architecture_and_Engineering", "Art",
        "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry",
        "Clinical_Medicine", "Computer_Science", "Design", "Diagnostics_and_Laboratory_Medicine",
        "Economics", "Electronics", "Energy_and_Power", "Finance", "Geography",
        "History", "Literature", "Manage", "Marketing", "Materials", "Math",
        "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology",
        "Public_Health", "Sociology",
    ]

    if local_dir:
        available = []
        for d in os.listdir(local_dir):
            if d.startswith("MMMU_") and not d.startswith("MMMU__"):
                subj_name = d.replace("MMMU_", "")
                available.append(subj_name)
        if subjects:
            wanted = [s for s in subjects if s in available]
        else:
            wanted = available
        print(f"Local mode: found {len(available)} subjects in {local_dir}")
    else:
        wanted = subjects or ALL_SUBJECTS

    out = []
    for subj in wanted:
        if local_dir:
            from datasets import load_from_disk
            ds_path = os.path.join(local_dir, f"MMMU_{subj}")
            ds = load_from_disk(ds_path)[split]
        else:
            ds = load_dataset("lmms-lab/MMMU", subj, split=split, cache_dir=cache_dir)

        for item in ds:
            raw_opts = item.get("options", "[]")
            try:
                opts_parsed = ast.literal_eval(raw_opts) if isinstance(raw_opts, str) else list(raw_opts)
            except Exception:
                opts_parsed = []

            images = []
            for k in [f"image_{i}" for i in range(1, 8)]:
                img = item.get(k)
                if img is not None:
                    images.append(img)

            out.append({
                "id": item["id"],
                "subject": subj,
                "question": item["question"],
                "question_type": item["question_type"],
                "options_parsed": opts_parsed,
                "answer": item["answer"],
                "images": images,
                "subfield": item.get("subfield", ""),
                "topic_difficulty": item.get("topic_difficulty", ""),
            })
    return out


def replace_image_tokens(text, n_images):
    for i in range(1, n_images + 1):
        text = text.replace(f"<image {i}>", "")
    return re.sub(r"\s+", " ", text).strip()


def save_pil_to_disk(pil_img, tmpdir, key):
    path = os.path.join(tmpdir, f"{key}.png")
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")
    MAX_PIXELS = 1536 * 1536
    w, h = pil_img.size
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    pil_img.save(path)
    return path


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", default="eval_mmmu_results.json")
    parser.add_argument("--cache_dir", default=None, help="HF datasets cache dir")
    parser.add_argument("--local_dir", default=None,
                        help="Local dir with MMMU_* datasets (e.g. /home/.../dataset)")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--n", type=int, default=-1, help="limit #samples (debug)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--prompt_style", default="default",
                        choices=["default", "qwen3_vl", "mmzero", "rethinker", "thinklite", "answer_tag"])
    parser.add_argument("--split", default="validation", choices=["dev", "validation", "test"])
    parser.add_argument("--split_think", action="store_true",
                        help="Strip <think>...</think> before parsing")
    parser.add_argument("--tmpdir", default="/tmp/mmmu_imgs")

    # LLM Judge arguments
    parser.add_argument("--use_llm_judge", action="store_true",
                        help="Use LLM-as-a-Judge for evaluation (in addition to rule-based)")
    parser.add_argument("--judge_model", type=str, default="gpt-5-4",
                        help="Judge model name (default: gpt-5-4)")
    parser.add_argument("--judge_api_base", type=str, default=None,
                        help="OpenAI-compatible API base URL (default: OpenAI official)")
    parser.add_argument("--judge_api_key", type=str, default=None,
                        help="API key for judge model (default: OPENAI_API_KEY env var)")
    parser.add_argument("--judge_max_workers", type=int, default=32,
                        help="Parallel threads for judge API calls (default: 32)")
    parser.add_argument("--judge_style", type=str, default="mmzero",
                        choices=["mmzero", "lenient"],
                        help="Judge prompt style: 'mmzero' matches MM-Zero paper exactly, "
                             "'lenient' is more generous (default: mmzero)")

    args = parser.parse_args()

    if args.prompt_style in {"rethinker", "thinklite", "mmzero"} and args.max_tokens < 4096:
        args.max_tokens = 8192
    if args.split_think and args.max_tokens < 4096:
        args.max_tokens = 16384

    random.seed(42)

    # ---- Load data ----
    print(f"Loading MMMU {args.split} split...")
    data = load_mmmu(split=args.split, cache_dir=args.cache_dir, subjects=args.subjects,
                     local_dir=args.local_dir)
    print(f"Loaded {len(data)} samples across {len(set(s['subject'] for s in data))} subjects")
    if args.n > 0:
        data = data[:args.n]
        print(f"Debug mode: truncated to {len(data)} samples")

    # Materialize images to disk
    os.makedirs(args.tmpdir, exist_ok=True)
    for s in tqdm(data, desc="Persisting images"):
        paths = []
        for i, img in enumerate(s["images"]):
            paths.append(save_pil_to_disk(img, args.tmpdir, f"{s['id']}_{i}"))
        s["image_paths"] = paths
        s["images"] = None

    # ---- Load model ----
    print(f"Loading model: {args.model_path}")
    allowed_path = os.path.realpath(args.tmpdir)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 7},
        allowed_local_media_path=allowed_path,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # ---- Inference loop ----
    results = []
    correct_rule, total = 0, 0
    t0 = time.time()

    for i in tqdm(range(0, len(data), args.batch_size), desc="Evaluating"):
        batch = data[i:i + args.batch_size]
        messages_batch = []
        for s in batch:
            sys_prompt, user_text = build_prompt(s, args.prompt_style)
            n_img = len(s["image_paths"])
            user_text = replace_image_tokens(user_text, n_img)

            content = []
            for p in s["image_paths"]:
                content.append({"type": "image_url",
                                "image_url": {"url": f"file://{os.path.realpath(p)}"}})
            content.append({"type": "text", "text": user_text})

            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": content})
            messages_batch.append(messages)

        outputs = llm.chat(messages_batch, sampling_params=sampling_params)

        for s, out in zip(batch, outputs):
            resp = out.outputs[0].text
            is_correct, parsed = mmmu_match(resp, s, args.prompt_style, args.split_think)
            correct_rule += int(is_correct)
            total += 1
            results.append({
                "id": s["id"],
                "subject": s["subject"],
                "subfield": s["subfield"],
                "question_type": s["question_type"],
                "difficulty": s["topic_difficulty"],
                "question": s["question"],
                "options": s["options_parsed"],
                "gt": s["answer"],
                "pred_parsed": parsed,
                "correct": is_correct,           # rule-based
                "correct_judge": None,            # will be filled by judge
                "judge_response": None,
                "response": resp,
                "n_images": len(s["image_paths"]),
            })

        tqdm.write(f"  [{total}/{len(data)}] rule_acc={correct_rule/total*100:.1f}%")

    inference_time = time.time() - t0

    # ---- LLM Judge (optional) ----
    if args.use_llm_judge:
        judge_api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY")
        if not judge_api_key:
            print("WARNING: No judge API key provided. Set --judge_api_key or OPENAI_API_KEY.")
            print("         Skipping LLM judge, using rule-based results only.")
        else:
            print(f"\nRunning LLM-as-a-Judge with {args.judge_model}...")
            print(f"  API base: {args.judge_api_base or 'OpenAI official'}")
            print(f"  Max workers: {args.judge_max_workers}")

            client = create_judge_client(args.judge_api_base, judge_api_key)
            t_judge = time.time()
            judge_batch(client, args.judge_model, data, results,
                       max_workers=args.judge_max_workers,
                       judge_style=args.judge_style)
            judge_time = time.time() - t_judge
            print(f"  Judge completed in {judge_time:.0f}s")

    # ---- Determine which accuracy to use as primary ----
    use_judge = args.use_llm_judge and results[0].get("correct_judge") is not None
    correct_key = "correct_judge" if use_judge else "correct"
    eval_method = "LLM Judge" if use_judge else "Rule-based"

    correct = sum(1 for r in results if r[correct_key])

    # ---- Breakdown ----
    subj_c, subj_t = defaultdict(int), defaultdict(int)
    qt_c, qt_t = defaultdict(int), defaultdict(int)
    diff_c, diff_t = defaultdict(int), defaultdict(int)

    SUBJECT_TO_CATEGORY = {
        "Art": "Art & Design", "Art_Theory": "Art & Design",
        "Design": "Art & Design", "Music": "Art & Design",
        "Accounting": "Business", "Economics": "Business",
        "Finance": "Business", "Manage": "Business", "Marketing": "Business",
        "Biology": "Science", "Chemistry": "Science",
        "Geography": "Science", "Math": "Science", "Physics": "Science",
        "Basic_Medical_Science": "Health & Medicine",
        "Clinical_Medicine": "Health & Medicine",
        "Diagnostics_and_Laboratory_Medicine": "Health & Medicine",
        "Pharmacy": "Health & Medicine", "Public_Health": "Health & Medicine",
        "History": "Humanities & Social Science",
        "Literature": "Humanities & Social Science",
        "Sociology": "Humanities & Social Science",
        "Psychology": "Humanities & Social Science",
        "Agriculture": "Tech & Engineering",
        "Architecture_and_Engineering": "Tech & Engineering",
        "Computer_Science": "Tech & Engineering",
        "Electronics": "Tech & Engineering",
        "Energy_and_Power": "Tech & Engineering",
        "Materials": "Tech & Engineering",
        "Mechanical_Engineering": "Tech & Engineering",
    }
    cat_c, cat_t = defaultdict(int), defaultdict(int)

    for r in results:
        c = int(r[correct_key])
        subj_c[r["subject"]] += c; subj_t[r["subject"]] += 1
        qt_c[r["question_type"]] += c; qt_t[r["question_type"]] += 1
        diff_c[r["difficulty"]] += c; diff_t[r["difficulty"]] += 1
        cat = SUBJECT_TO_CATEGORY.get(r["subject"], "Unknown")
        cat_c[cat] += c; cat_t[cat] += 1

    acc = correct / total * 100 if total else 0.0
    acc_rule = correct_rule / total * 100 if total else 0.0

    print(f"\n{'='*60}")
    print(f"MMMU {args.split.capitalize()} Results  [{eval_method}]")
    print(f"  Model: {args.model_path}")
    print(f"  Prompt style: {args.prompt_style}  |  split_think: {args.split_think}")
    print(f"  Samples: {total}")
    print(f"  Rule-based Accuracy:  {correct_rule}/{total} = {acc_rule:.2f}%")
    if use_judge:
        print(f"  LLM Judge Accuracy:   {correct}/{total} = {acc:.2f}%")
        # Show delta
        delta = acc - acc_rule
        print(f"  Delta (Judge - Rule): {delta:+.2f}%")

    print(f"\n  Per category ({eval_method}):")
    for c in sorted(cat_t):
        print(f"    {c:32s}: {cat_c[c]:4d}/{cat_t[c]:4d} = {cat_c[c]/cat_t[c]*100:5.2f}%")

    print(f"\n  Per question type ({eval_method}):")
    for qt in sorted(qt_t):
        print(f"    {qt:16s}: {qt_c[qt]:4d}/{qt_t[qt]:4d} = {qt_c[qt]/qt_t[qt]*100:5.2f}%")

    print(f"\n  Per difficulty ({eval_method}):")
    for d in sorted(diff_t):
        print(f"    {d:8s}: {diff_c[d]:4d}/{diff_t[d]:4d} = {diff_c[d]/diff_t[d]*100:5.2f}%")

    print(f"\n  Inference time: {inference_time:.0f}s ({inference_time/max(total,1):.1f}s/sample)")
    print(f"{'='*60}")

    summary = {
        "model": args.model_path,
        "prompt_style": args.prompt_style,
        "split_think": args.split_think,
        "eval_method": eval_method,
        "total": total,
        "correct_rule": correct_rule,
        "accuracy_rule": round(acc_rule, 2),
        "correct_judge": correct if use_judge else None,
        "accuracy_judge": round(acc, 2) if use_judge else None,
        "judge_model": args.judge_model if use_judge else None,
        "per_category": {c: {"correct": cat_c[c], "total": cat_t[c],
                              "accuracy": round(cat_c[c]/cat_t[c]*100, 2)}
                         for c in sorted(cat_t)},
        "per_subject": {s: {"correct": subj_c[s], "total": subj_t[s],
                             "accuracy": round(subj_c[s]/subj_t[s]*100, 2)}
                        for s in sorted(subj_t)},
        "per_question_type": {qt: {"correct": qt_c[qt], "total": qt_t[qt],
                                    "accuracy": round(qt_c[qt]/qt_t[qt]*100, 2)}
                              for qt in sorted(qt_t)},
        "per_difficulty": {d: {"correct": diff_c[d], "total": diff_t[d],
                                "accuracy": round(diff_c[d]/diff_t[d]*100, 2)}
                           for d in sorted(diff_t)},
        "time_inference_s": round(inference_time, 1),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()