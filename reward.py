#!/usr/bin/env python3
"""
VICO Reward Function for verl GRPO

Binary reward following configurable exact or relative numeric matching:
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
#  Answer normalization and checking
# ═══════════════════════════════════════════════════════════════════

def normalize_text(s):
    """
    Normalize text for VICO comparison:
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


def check_correct(gt, pred, metric="relaxed", tolerance=0.05):
    from vico.metrics import matches
    return matches(pred, gt, metric, tolerance)


def compute_score(data_source=None, completion=None, reference=None, extra_info=None,
                  solution_str=None, ground_truth=None, metric="relaxed",
                  tolerance=0.05, require_format=True):
    """Binary format/answer reward; also accepts the standard verl keyword names."""
    completion = completion if completion is not None else solution_str
    reference = reference if reference is not None else ground_truth
    if completion is None or reference is None:
        raise ValueError("Reward requires a completion and reference answer")
    format_ok = bool(re.fullmatch(
        r"\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*", completion))
    correct = check_correct(reference, extract_answer(completion), metric, tolerance)
    return 1.0 if (format_ok or not require_format) and correct else -1.0


# ═══════════════════════════════════════════════════════════════════
#  verl Reward Manager
# ═══════════════════════════════════════════════════════════════════

@register("vico_reward")
class VICORewardManager(AbstractRewardManager):
    """Reward manager for VICO in verl GRPO."""

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
        self.compute_score = compute_score or globals()["compute_score"]

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
            score = self.compute_score(
                data_source=data[i].non_tensor_batch.get(self.reward_fn_key),
                completion=sequences_str[i],
                reference=ground_truths[i],
            )
            response_length = int(valid_response_length[i].item())
            if response_length > 0:
                reward_tensor[i, response_length - 1] = score

            if already_printed < self.num_examine:
                already_printed += 1
                pred = extract_answer(sequences_str[i])
                gt = ground_truths[i]
                fmt_ok, _ = check_format(sequences_str[i])
                correct = check_correct(gt, pred)
                resp_len = len(sequences_str[i])
                print(f"  [VICO Reward] gt='{gt}', pred='{pred}', score={score:.1f}, "
                      f"correct={correct}, format={fmt_ok}, "
                      f"resp_len={resp_len} chars")
                if already_printed <= 2:
                    print(f"    Response (first 300 chars): {sequences_str[i][:300]}")
                    print(f"    Response (last 200 chars):  ...{sequences_str[i][-200:]}")

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor
