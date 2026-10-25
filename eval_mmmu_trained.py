#!/usr/bin/env python3
"""
Evaluate trained (co-evolve) solver models on MMMU.

Optimized specifically for models trained with <think>...</think><answer>...</answer>
format (e.g. round_3/solver_genenv_merged_step210, vico_round3_solver_merged_step110).

Improvements over eval_mmmu.py (borrowed from eval_mathvista.py):
  1. MathVista-style prompt with explicit format hint ("...at the end").
  2. Structured Question:/Choices: (A) (B)... format (stronger attention anchor).
  3. Priority-based MCQ parser (parse_mcq) — multiple patterns with ranking.
  4. Levenshtein fallback for MCQ when no letter is extracted (no random guessing).
  5. Multi-level answer extraction: <answer> -> \\boxed{} -> "the answer is X" -> last number.
  6. Default max_tokens=8192 (thinking chain often needs >2048).

Prompt styles:
  * answer_tag (default) — matches trained model's <think>/<answer> format, with MV hints.
  * boxed               — base-model style with \\boxed{} output.
  * plain               — no special format, just MV hints (closest to lmms-eval MMMU).

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python3 eval_mmmu_trained.py \\
        --model_path /path/to/VICO/round_3/solver_genenv_merged_step210 \\
        --tp 2 --prompt_style answer_tag --split validation \\
        --output eval_mmmu_step210.json

    # With LLM judge
    CUDA_VISIBLE_DEVICES=0,1 python3 eval_mmmu_trained.py \\
        --model_path .../vico_round3_solver_merged_step110 \\
        --tp 2 --prompt_style answer_tag --split validation \\
        --use_llm_judge --judge_model gpt-5-4 --judge_api_key $OPENAI_API_KEY \\
        --output eval_mmmu_vico110.json
"""

import os, json, re, ast, random, argparse, time
import numpy as np
from collections import defaultdict
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image as PILImage
import concurrent.futures

try:
    from Levenshtein import distance as levenshtein_distance
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("[WARN] python-Levenshtein not installed; MCQ text-match fallback disabled.")


# ══════════════════════════════════════════════════════════════════════
# LLM-as-a-Judge
# ══════════════════════════════════════════════════════════════════════

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
    "- Same meaning in different words\n"
    "- Numeric equivalence with rounding up to 5% of the gold value\n"
    "- Option letter in any format: A == (A) == A. == a == full text of option A\n"
    "- If the model selected the correct option's CONTENT even without stating the letter, mark correct\n"
    "- Units present or absent\n"
    "- Fraction/decimal/percentage interchangeable\n"
    "- If the model's response contains the correct answer ANYWHERE in it, mark YES\n\n"
    "## Mark NO only when:\n"
    "- The answer is clearly and unambiguously WRONG\n"
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
    system_prompt = JUDGE_SYSTEM_MMZERO if judge_style == "mmzero" else JUDGE_SYSTEM_LENIENT

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
        question=question_display, gold=gold_display, model_answer=model_answer,
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
            return judge_text.startswith("yes"), judge_text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: Judge API failed after {max_retries} retries: {e}")
                return None, f"ERROR: {e}"


def judge_batch(client, model_name, samples, results_list, max_workers=16, judge_style="mmzero"):
    def _judge_one(idx):
        r = results_list[idx]
        s = samples[idx]
        model_answer = r.get("response", "")
        if len(model_answer) > 2000:
            model_answer = model_answer[:2000] + "\n[...truncated]"
        return (idx, *judge_single(
            client, model_name,
            question=s["question"], gold_answer=s["answer"], model_answer=model_answer,
            options=s.get("options_parsed"), question_type=s.get("question_type"),
            judge_style=judge_style,
        ))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_judge_one, i): i for i in range(len(results_list))}
        for future in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures), desc="LLM Judge"):
            idx, is_correct, judge_resp = future.result()
            if is_correct is not None:
                results_list[idx]["correct_judge"] = is_correct
                results_list[idx]["judge_response"] = judge_resp
            else:
                results_list[idx]["correct_judge"] = results_list[idx]["correct"]
                results_list[idx]["judge_response"] = "FALLBACK_TO_RULE"


# ══════════════════════════════════════════════════════════════════════
# Prompt construction — MathVista-inspired with format hints
# ══════════════════════════════════════════════════════════════════════

MCQ_HINT = (
    "Hint: Please answer the question and provide the correct option letter, "
    "e.g., A, B, C, D, at the end."
)

OPEN_HINT = (
    "Hint: Please answer the question and provide the final answer "
    "as a single word, phrase, or number at the end."
)

# Compact system prompt matching trained <think>/<answer> distribution,
# but kept shorter than training-time system to avoid over-biasing on SCENE/VICO.
ANSWER_TAG_SYS = (
    "You are a helpful assistant. Think step by step in <think>...</think> tags, "
    "then give the final answer in <answer>...</answer> tags."
)

BOXED_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def build_query(sample):
    """MathVista-style structured query with type-appropriate hint."""
    q = f"Question: {sample['question']}"
    if sample["question_type"] == "multiple-choice":
        lines = ["Choices:"]
        for i, c in enumerate(sample["options_parsed"]):
            lines.append(f"({chr(ord('A') + i)}) {c}")
        choices_text = "\n".join(lines)
        return f"{q}\n{choices_text}\n{MCQ_HINT}"
    else:
        return f"{q}\n{OPEN_HINT}"


def build_prompt(sample, style):
    query = build_query(sample)
    if style == "answer_tag":
        return ANSWER_TAG_SYS, query
    if style == "boxed":
        return None, query + BOXED_SUFFIX
    if style == "plain":
        return None, query
    raise ValueError(f"unknown prompt_style: {style}")


# ══════════════════════════════════════════════════════════════════════
# Answer extraction — multi-strategy chain
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


def extract_answer_is(text):
    """Match 'the answer is X' / '答案是 X' patterns (from MathVista)."""
    for pat in [
        r'(?:the |my |final )?answer is[:\s]*([^\n.;]+)',
        r'(?:答案|结果)[是为：:\s]*([^\n.;。]+)',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('.').strip()
    return None


def extract_last_number(text):
    """Last number in last non-empty line (from MathVista)."""
    for line in reversed(text.strip().split("\n")):
        if line.strip():
            m = re.search(r'\b(-?\d+(?:\.\d+)?)\b\s*\.?\s*$', line.strip())
            if m:
                return m.group(1)
    return None


def extract_response(text, prompt_style):
    """
    Multi-strategy extraction. Strip <think> first, then try
    tag/boxed (by style), then fallbacks.
    """
    clean = strip_thinking(text)
    candidates = []

    if prompt_style == "answer_tag":
        candidates.append(extract_answer_tag(clean))
        candidates.append(extract_boxed(clean))
    elif prompt_style == "boxed":
        candidates.append(extract_boxed(clean))
        candidates.append(extract_answer_tag(clean))
    else:
        candidates.append(extract_answer_tag(clean))
        candidates.append(extract_boxed(clean))

    for c in candidates:
        if c:
            return c

    ans_is = extract_answer_is(clean)
    if ans_is:
        return ans_is

    num = extract_last_number(clean)
    if num:
        return num

    return clean


# ══════════════════════════════════════════════════════════════════════
# MCQ parsing — priority-ranked (ported from eval_mathvista.py)
# ══════════════════════════════════════════════════════════════════════

def parse_mcq(text, num_choices):
    """
    Priority-based MCQ letter extraction.
    Returns uppercase letter or empty string.
    """
    if not text or not text.strip():
        return ""

    response = text.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(char)
    response = " " + response + " "

    all_choices = [chr(ord("A") + i) for i in range(num_choices)]
    candidates = []

    # (A), A., A:, A), A followed by space
    patterns = [
        ("paren",  lambda c: f"({c})"),
        ("period", lambda c: f"{c}."),
        ("colon",  lambda c: f"{c}:"),
        ("rparen", lambda c: f"{c})"),
        ("space",  lambda c: f"{c} "),
    ]
    for fmt, suffix in patterns:
        for choice in all_choices:
            pat = suffix(choice)
            if pat in response:
                candidates.append((choice, response.rfind(pat), fmt))

    # "the answer is A" phrases
    answer_phrases = [
        "the answer is", "answer is", "the correct answer is", "correct answer is",
        "i choose", "i select", "my answer is", "option",
    ]
    for phrase in answer_phrases:
        lo = response.lower()
        if phrase in lo:
            idx = lo.find(phrase)
            for choice in all_choices:
                cpos = response.find(choice, idx)
                if cpos != -1:
                    candidates.append((choice, cpos, "phrase"))

    # start / end
    stripped = response.strip()
    for choice in all_choices:
        if stripped.startswith(choice):
            candidates.append((choice, 0, "start"))
        if stripped.endswith(choice):
            candidates.append((choice, len(response) - 1, "end"))

    # fallback: any occurrence
    if not candidates:
        for choice in all_choices:
            if choice in response:
                candidates.append((choice, response.rfind(choice), "fallback"))

    if candidates:
        priority = {"start": 10, "end": 9, "phrase": 7, "paren": 6, "period": 5,
                    "colon": 4, "rparen": 3, "space": 2, "fallback": 0}
        # Prefer higher priority, then later position (model's final answer tends to be last)
        candidates.sort(key=lambda x: (priority.get(x[2], 0), x[1]), reverse=True)
        return candidates[0][0]

    return ""


def mcq_match_with_levenshtein(extracted, options):
    """
    If letter extraction fails, find the option whose text is closest
    (in edit distance) to the extracted string. Returns letter.
    """
    if not HAS_LEVENSHTEIN or not extracted or not options:
        return None
    distances = [levenshtein_distance(extracted.lower(), str(o).lower()) for o in options]
    best = int(np.argmin(distances))
    # Only accept if reasonably close (<=50% of option length, avoids random matches)
    if distances[best] <= max(len(options[best]), 4) * 0.5:
        return chr(ord("A") + best)
    return None


# ══════════════════════════════════════════════════════════════════════
# Open-ended parsing (MMMU official, kept as-is)
# ══════════════════════════════════════════════════════════════════════

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


def mmmu_match(response, sample, prompt_style):
    """Full MMMU matching with polished extraction + MCQ parsing."""
    clean = extract_response(response, prompt_style)
    if sample["question_type"] == "multiple-choice":
        options = sample["options_parsed"]
        letter = parse_mcq(clean, len(options))
        if not letter:
            # Fallback 1: Levenshtein against option texts
            letter = mcq_match_with_levenshtein(clean, options)
        if not letter:
            # Fallback 2: random (last resort, better than None)
            letter = random.choice([chr(ord("A") + i) for i in range(len(options))])
        return eval_multi_choice(sample["answer"], letter), letter
    else:
        parsed = parse_open_response(clean)
        return eval_open(sample["answer"], parsed), (parsed[0] if parsed else None)


# ══════════════════════════════════════════════════════════════════════
# MMMU data loading
# ══════════════════════════════════════════════════════════════════════

ALL_SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art",
    "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry",
    "Clinical_Medicine", "Computer_Science", "Design", "Diagnostics_and_Laboratory_Medicine",
    "Economics", "Electronics", "Energy_and_Power", "Finance", "Geography",
    "History", "Literature", "Manage", "Marketing", "Materials", "Math",
    "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology",
    "Public_Health", "Sociology",
]

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


def load_mmmu(split="validation", cache_dir=None, subjects=None, local_dir=None):
    if local_dir:
        available = []
        for d in os.listdir(local_dir):
            if d.startswith("MMMU_") and not d.startswith("MMMU__"):
                available.append(d.replace("MMMU_", ""))
        wanted = [s for s in (subjects or available) if s in available]
        print(f"Local mode: {len(available)} subjects found in {local_dir}")
    else:
        wanted = subjects or ALL_SUBJECTS

    out = []
    for subj in wanted:
        if local_dir:
            from datasets import load_from_disk
            ds = load_from_disk(os.path.join(local_dir, f"MMMU_{subj}"))[split]
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
    parser.add_argument("--output", default="eval_mmmu_trained_results.json")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--prompt_style", default="answer_tag",
                        choices=["answer_tag", "boxed", "plain"])
    parser.add_argument("--split", default="validation", choices=["dev", "validation", "test"])
    parser.add_argument("--tmpdir", default="/tmp/mmmu_imgs_trained")
    parser.add_argument("--gpu_mem_util", type=float, default=0.85)

    # LLM Judge
    parser.add_argument("--use_llm_judge", action="store_true")
    parser.add_argument("--judge_model", type=str, default="gpt-5-4")
    parser.add_argument("--judge_api_base", type=str, default=None)
    parser.add_argument("--judge_api_key", type=str, default=None)
    parser.add_argument("--judge_max_workers", type=int, default=32)
    parser.add_argument("--judge_style", type=str, default="mmzero",
                        choices=["mmzero", "lenient"])

    # Resume / incremental save
    parser.add_argument("--resume", action="store_true",
                        help="Resume from --output (skip samples already present there)")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save partial results every N batches (default 5)")

    args = parser.parse_args()
    random.seed(42)

    # ---- Load data ----
    print(f"Loading MMMU {args.split} split...")
    data = load_mmmu(split=args.split, cache_dir=args.cache_dir,
                     subjects=args.subjects, local_dir=args.local_dir)
    print(f"Loaded {len(data)} samples across {len(set(s['subject'] for s in data))} subjects")
    if args.n > 0:
        data = data[:args.n]
        print(f"Debug mode: truncated to {len(data)} samples")

    # ---- Resume: load already-done results ----
    prev_results = []
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            prev = json.load(f)
        prev_results = prev.get("results", [])
        done_ids = {r["id"] for r in prev_results}
        print(f"Resume: loaded {len(done_ids)} done samples from {args.output}")
        data = [s for s in data if s["id"] not in done_ids]
        print(f"  Remaining: {len(data)}")

    os.makedirs(args.tmpdir, exist_ok=True)
    for s in tqdm(data, desc="Persisting images"):
        paths = []
        for i, img in enumerate(s["images"]):
            paths.append(save_pil_to_disk(img, args.tmpdir, f"{s['id']}_{i}"))
        s["image_paths"] = paths
        s["images"] = None

    # ---- Load model ----
    print(f"Loading model: {args.model_path}")
    print(f"  prompt_style: {args.prompt_style}  |  max_tokens: {args.max_tokens}")
    allowed_path = os.path.realpath(args.tmpdir)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
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

    # ---- Inference ----
    results = list(prev_results)
    correct_rule = sum(1 for r in results if r.get("correct"))
    total = len(results)
    extract_failed = sum(
        1 for r in results
        if r.get("pred_parsed") is None
        or (isinstance(r.get("pred_parsed"), str) and not r["pred_parsed"].strip())
    )
    t0 = time.time()

    def _save_partial():
        partial = {
            "model": args.model_path,
            "prompt_style": args.prompt_style,
            "total": total,
            "correct_rule_running": correct_rule,
            "accuracy_rule_running": round(correct_rule / total * 100, 2) if total else 0.0,
            "extract_failures": extract_failed,
            "partial": True,
            "results": results,
        }
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(partial, f, ensure_ascii=False, indent=2)
        os.replace(tmp, args.output)

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
            is_correct, parsed = mmmu_match(resp, s, args.prompt_style)
            correct_rule += int(is_correct)
            total += 1

            # Track extraction failures (empty parsed or only whitespace)
            if parsed is None or (isinstance(parsed, str) and not parsed.strip()):
                extract_failed += 1

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
                "correct": is_correct,
                "correct_judge": None,
                "judge_response": None,
                "response": resp,
                "n_images": len(s["image_paths"]),
            })

        tqdm.write(f"  [{total}/{len(data) + total}] "
                   f"rule_acc={correct_rule/total*100:.1f}% "
                   f"extract_fail={extract_failed}")

        batch_idx = i // args.batch_size
        if args.save_every > 0 and (batch_idx + 1) % args.save_every == 0:
            _save_partial()

    inference_time = time.time() - t0

    # ---- LLM Judge ----
    if args.use_llm_judge:
        judge_api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY")
        if not judge_api_key:
            print("WARNING: No judge API key, skipping.")
        else:
            print(f"\nRunning LLM-as-a-Judge with {args.judge_model}...")
            client = create_judge_client(args.judge_api_base, judge_api_key)
            t_judge = time.time()
            judge_batch(client, args.judge_model, data, results,
                       max_workers=args.judge_max_workers, judge_style=args.judge_style)
            print(f"  Judge completed in {time.time() - t_judge:.0f}s")

    use_judge = args.use_llm_judge and results[0].get("correct_judge") is not None
    correct_key = "correct_judge" if use_judge else "correct"
    eval_method = "LLM Judge" if use_judge else "Rule-based"

    correct = sum(1 for r in results if r[correct_key])

    # ---- Breakdown ----
    subj_c, subj_t = defaultdict(int), defaultdict(int)
    qt_c, qt_t = defaultdict(int), defaultdict(int)
    diff_c, diff_t = defaultdict(int), defaultdict(int)
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
    print(f"  Prompt style: {args.prompt_style}")
    print(f"  Samples: {total}")
    print(f"  Rule-based Accuracy: {correct_rule}/{total} = {acc_rule:.2f}%")
    if use_judge:
        print(f"  LLM Judge Accuracy:  {correct}/{total} = {acc:.2f}%")
        print(f"  Delta: {acc - acc_rule:+.2f}%")
    print(f"  Extraction failures: {extract_failed}/{total}")

    print(f"\n  Per category:")
    for c in sorted(cat_t):
        print(f"    {c:32s}: {cat_c[c]:4d}/{cat_t[c]:4d} = {cat_c[c]/cat_t[c]*100:5.2f}%")
    print(f"\n  Per question type:")
    for qt in sorted(qt_t):
        print(f"    {qt:16s}: {qt_c[qt]:4d}/{qt_t[qt]:4d} = {qt_c[qt]/qt_t[qt]*100:5.2f}%")
    print(f"\n  Per difficulty:")
    for d in sorted(diff_t):
        print(f"    {d:8s}: {diff_c[d]:4d}/{diff_t[d]:4d} = {diff_c[d]/diff_t[d]*100:5.2f}%")

    print(f"\n  Inference time: {inference_time:.0f}s ({inference_time/max(total,1):.1f}s/sample)")
    print(f"{'='*60}")

    summary = {
        "model": args.model_path,
        "prompt_style": args.prompt_style,
        "eval_method": eval_method,
        "total": total,
        "correct_rule": correct_rule,
        "accuracy_rule": round(acc_rule, 2),
        "correct_judge": correct if use_judge else None,
        "accuracy_judge": round(acc, 2) if use_judge else None,
        "judge_model": args.judge_model if use_judge else None,
        "extract_failures": extract_failed,
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
