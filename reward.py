#!/usr/bin/env python3
"""
vico Reward Function for verl GRPO

Binary reward following vico relaxed accuracy conventions:
  +1.0 if format correct AND answer correct
  -1.0 otherwise

Format correct = response contains <think>...</think> AND <answer>...</answer>.
Answer correct:
  - Numeric: within 5% relative tolerance
  - Text: normalized exact match (strip punctuation, whitespace, case-insensitive)
"""

import re
import unicodedata
from typing import Any, Callable, Optional

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.registry import register


# ═══════════════════════════════════════════════════════════════════
#  Format checking
# ═══════════════════════════════════════════════════════════════════

def check_format(text):
    """
    Check if response follows <think>...</think><answer>...</answer> format.
    Returns (is_valid, details_dict).
    """
    details = {
        "has_think_open": "<think>" in text,
        "has_think_close": "</think>" in text,
        "has_answer_open": "<answer>" in text,
        "has_answer_close": "</answer>" in text,
    }
    details["format_valid"] = all(details.values())
    return details["format_valid"], details


# ═══════════════════════════════════════════════════════════════════
#  Answer extraction
# ═══════════════════════════════════════════════════════════════════

def extract_answer(text):
    """
    Extract answer from response. Priority:
      1. <answer>X</answer> tags
      2. \\boxed{X}
      3. Fallback patterns
    """
    # Try <answer>...</answer> first
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try \\boxed{...}
    for pat in [r'\\boxed\{', r'\\boxed \{']:
        m = re.search(pat, text)
        if m:
            s, d, i = m.end(), 1, m.end()
            while i < len(text) and d > 0:
                if text[i] == '{': d += 1
                elif text[i] == '}': d -= 1
                i += 1
            if d == 0:
                ans = text[s:i-1].strip()
                ans = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', ans)
                return re.sub(r'[\\%$]', '', ans).strip()

    # Fallback: common patterns
    text_clean = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text_clean = re.sub(r'<think>[\s\S]*$', '', text_clean).strip()

    for p in ["The answer is:", "The answer is", "Answer:", "Answer"]:
        if text_clean.lower().startswith(p.lower()):
            text_clean = text_clean[len(p):].strip()
            return re.sub(r'[\\%$]', '', text_clean.rstrip('.')).strip()

    return text_clean.strip()


# ═══════════════════════════════════════════════════════════════════
#  Answer normalization and checking (vico style)
# ═══════════════════════════════════════════════════════════════════

def normalize_text(s):
    """
    Normalize text for vico comparison:
      - lowercase
      - strip leading/trailing whitespace
      - collapse multiple spaces
      - strip punctuation (.,;:!?'"-)
      - normalize & → and
      - normalize unicode (accents etc.)
      - strip articles (a, an, the)
    """
    s = str(s).strip().lower()
    # Unicode normalize
    s = unicodedata.normalize("NFKD", s)
    # & → and
    s = s.replace("&", " and ")
    # Remove punctuation (keep alphanumeric, spaces, %, $)
    s = re.sub(r'[.,;:!?\'"()\[\]{}\-–—/\\]', ' ', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Strip trailing/leading articles
    s = re.sub(r'^(the|a|an)\s+', '', s)
    s = re.sub(r'\s+(the|a|an)$', '', s)
    return s


def try_parse_number(s):
    """Try to parse a string as a number. Returns (True, float) or (False, None)."""
    s = str(s).strip()
    # Remove common formatting: $, %, commas, spaces
    cleaned = re.sub(r'[$%,\s]', '', s)
    # Handle trailing % or units
    cleaned = re.sub(r'(million|billion|trillion|k|m|b)$', '', cleaned, flags=re.IGNORECASE)
    try:
        return True, float(cleaned)
    except (ValueError, TypeError):
        return False, None


def check_correct(gt, pred):
    """
    vico relaxed accuracy:
      - Numeric: 5% relative tolerance
      - Text: normalized exact match
    """
    gt_str = str(gt).strip()
    pred_str = str(pred).strip()

    if not pred_str:
        return False

    # Try numeric comparison first
    gt_is_num, gt_num = try_parse_number(gt_str)
    pred_is_num, pred_num = try_parse_number(pred_str)

    if gt_is_num and pred_is_num:
        if gt_num == 0:
            return abs(pred_num) < 1e-6
        return abs(gt_num - pred_num) / abs(gt_num) <= 0.05

    # Text comparison with normalization
    gt_norm = normalize_text(gt_str)
    pred_norm = normalize_text(pred_str)

    if gt_norm == pred_norm:
        return True

    # Also check if pred contains gt or vice versa (for short GT like country names)
    # Only when GT is short (≤3 words) to avoid false positives
    if len(gt_norm.split()) <= 3:
        # pred starts with gt (e.g. pred="Italy in 2020" gt="Italy")
        if pred_norm.startswith(gt_norm):
            return True

    return False


def compute_score(data_source, completion, reference, extra_info=None):
    """
    Binary reward (ReVPT style):
      +1.0 if format correct AND answer correct
      -1.0 otherwise
    """
    format_ok, _ = check_format(completion)
    pred = extract_answer(completion)
    answer_ok = check_correct(reference, pred)

    if format_ok and answer_ok:
        return 1.0
    return -1.0


# ═══════════════════════════════════════════════════════════════════
#  verl Reward Manager
# ═══════════════════════════════════════════════════════════════════

@register("vico_reward")
class VICORewardManager(AbstractRewardManager):
    """Reward manager for vico in verl GRPO."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Compute rewards for a batch of responses."""

        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        ground_truths = [
            data_item.non_tensor_batch["reward_model"]["ground_truth"]
            for data_item in data
        ]

        already_printed = 0
        for i in range(len(data)):
            score = compute_score(
                data_source=None,
                completion=sequences_str[i],
                reference=ground_truths[i],
            )
            reward_tensor[i, valid_response_length[i].item() - 1] = score

            if already_printed < self.num_examine:
                already_printed += 1
                pred = extract_answer(sequences_str[i])
                gt = ground_truths[i]
                fmt_ok, _ = check_format(sequences_str[i])
                correct = check_correct(gt, pred)
                resp_len = len(sequences_str[i])
                print(f"  [vico Reward] gt='{gt}', pred='{pred}', score={score:.1f}, "
                      f"correct={correct}, format={fmt_ok}, "
                      f"resp_len={resp_len} chars")
                if already_printed <= 2:
                    print(f"    Response (first 300 chars): {sequences_str[i][:300]}")
                    print(f"    Response (last 200 chars):  ...{sequences_str[i][-200:]}")

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor


if __name__ == "__main__":
    tests = [
        # ── Numeric: same as scene ──
        ("<think>Reading the chart.</think><answer>6.68</answer>",
         "6.68", 1.0, "exact numeric match"),
        ("<think>Reading the chart.</think><answer>6.7</answer>",
         "6.68", 1.0, "numeric within 5% tolerance"),
        ("<think>Reading the chart.</think><answer>45</answer>",
         "45", 1.0, "integer match"),
        ("<think>Reading.</think><answer>2020</answer>",
         "2020.0", 1.0, "int vs float match"),
        ("<think>Reading.</think><answer>100,000</answer>",
         "100000", 1.0, "comma-formatted number"),
        ("<think>Reading.</think><answer>312,015</answer>",
         "312015", 1.0, "comma-formatted number 2"),
        ("<think>Wrong.</think><answer>50</answer>",
         "6.68", -1.0, "wrong numeric answer"),

        # ── Text: normalized matching ──
        ("<think>Looking at the chart.</think><answer>Italy</answer>",
         "Italy", 1.0, "exact text match"),
        ("<think>Looking.</think><answer>italy</answer>",
         "Italy", 1.0, "case insensitive"),
        ("<think>Looking.</think><answer>Italy.</answer>",
         "Italy", 1.0, "trailing period stripped"),
        ("<think>Looking.</think><answer>Bausch & Lomb</answer>",
         "Bausch & Lomb", 1.0, "ampersand exact"),
        ("<think>Looking.</think><answer>Bausch and Lomb</answer>",
         "Bausch & Lomb", 1.0, "& vs and normalized"),
        ("<think>Looking.</think><answer>AT&T</answer>",
         "AT&T", 1.0, "AT&T exact"),
        ("<think>Looking.</think><answer>AT and T</answer>",
         "AT&T", 1.0, "AT&T vs AT and T"),
        ("<think>Looking.</think><answer>March 2021</answer>",
         "March 2021", 1.0, "multi-word text"),
        ("<think>Looking.</think><answer>march  2021</answer>",
         "March 2021", 1.0, "extra spaces collapsed"),
        ("<think>Looking.</think><answer>The Italy</answer>",
         "Italy", 1.0, "article stripped"),
        ("<think>Looking.</think><answer>Italy in 2020</answer>",
         "Italy", 1.0, "pred starts with short GT"),
        ("<think>Looking.</think><answer>Germany</answer>",
         "Italy", -1.0, "wrong text answer"),

        # ── Format: must have both tags ──
        ("The answer is Italy", "Italy", -1.0, "no format tags"),
        ("<answer>Italy</answer>", "Italy", -1.0, "no think tag"),
        ("<think>OK</think>Italy", "Italy", -1.0, "no answer tag"),
        ("", "Italy", -1.0, "empty response"),
    ]

    print("vico Reward Tests (binary: +1 / -1)")
    print("=" * 70)
    all_pass = True
    for resp, gt, expected, desc in tests:
        score = compute_score(None, resp, gt)
        fmt_ok, _ = check_format(resp)
        pred = extract_answer(resp)
        correct = check_correct(gt, pred)
        status = "PASS" if abs(score - expected) < 0.01 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status}: score={score:+.1f} (exp={expected:+.1f}) | "
              f"correct={correct}, format={fmt_ok}, pred='{pred[:40]}' | {desc}")

    print(f"\n  {'All tests passed!' if all_pass else 'SOME TESTS FAILED!'}")
