#!/usr/bin/env python3
"""
VL Reasoning Reward for Qwen3-VL Tool-Augmented Agent

Reward design:
  +3  tool_call + answer correct   (best: used tool and got it right)
  +1  no tool_call + answer correct (ok: got it right without tool)
  -3  answer wrong or no answer    (bad: wrong or incomplete)

Format requirement:
  Must have <answer>...</answer> to extract prediction.
  Tool calls (<tool_call>...</tool_call>) are optional but rewarded.
  <think>...</think> blocks are expected but not strictly enforced.

Compatible with verl GRPO framework.
"""

import re
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional, Union, List, Dict
from collections import Counter

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.registry import register


# ═══════════════════════════════════════════════════════════════════
#  Answer extraction
# ═══════════════════════════════════════════════════════════════════

def extract_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags."""
    if not text:
        return ""
    # Last <answer> tag (in case of multiple)
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if matches:
        return clean_answer(matches[-1].strip())
    return ""


def clean_answer(text: str) -> str:
    """Clean and normalize extracted answer text."""
    if not text:
        return ""
    # Remove \boxed{}
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        text = boxed.group(1)
    # Remove LaTeX commands
    text = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[{}^_$]', '', text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════
#  Answer correctness checking
# ═══════════════════════════════════════════════════════════════════

def normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.strip().lower()
    # Remove common units and modifiers
    text = re.sub(
        r'\b(million|billion|trillion|thousand|percent|approximately|about|'
        r'around|nearly|over|kg|km|miles|meters|dollars|euros|pounds|'
        r'cm|mm|m²|m2|cm²|cm2|degrees?|°)\b',
        '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\\%$,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def check_correct(gt: str, pred: str) -> bool:
    """Check if prediction matches ground truth."""
    if not pred:
        return False

    gt_s = normalize_for_comparison(str(gt))
    pred_s = normalize_for_comparison(str(pred))

    if not pred_s:
        return False

    # Exact string match
    if gt_s == pred_s:
        return True

    # Multi-choice: single letter match (A/B/C/D/E)
    gt_letter = re.match(r'^([a-e])$', gt_s)
    pred_letter = re.match(r'^([a-e])$', pred_s)
    if gt_letter and pred_letter and gt_letter.group(1) == pred_letter.group(1):
        return True
    # Extract letter from pred like "the answer is B" → "b"
    if gt_letter:
        letters = re.findall(r'\b([a-e])\b', pred_s)
        if letters and letters[-1] == gt_letter.group(1):
            return True

    # Numeric comparison (5% tolerance)
    try:
        gn = float(re.sub(r'[^0-9.\-eE]', '', gt_s))
        pn = float(re.sub(r'[^0-9.\-eE]', '', pred_s))
        if gn == 0:
            return abs(pn) < 1e-6
        if abs(gn - pn) / abs(gn) <= 0.05:
            return True
        # Scale awareness (million/billion)
        for scale in [1e3, 1e6, 1e9]:
            if abs(gn * scale - pn) / max(abs(gn * scale), 1e-9) <= 0.05:
                return True
            if abs(pn * scale - gn) / max(abs(pn * scale), 1e-9) <= 0.05:
                return True
    except (ValueError, ZeroDivisionError):
        pass

    # Integer vs float: "3" == "3.0"
    try:
        if float(gt_s) == float(pred_s):
            return True
    except:
        pass

    # Number word matching: "three" == "3"
    NUM_WORDS = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    }
    gt_num = NUM_WORDS.get(gt_s)
    pred_num = NUM_WORDS.get(pred_s)
    if gt_num and pred_num and gt_num == pred_num:
        return True
    if gt_num and pred_s == gt_num:
        return True
    if pred_num and gt_s == pred_num:
        return True

    # Substring match for text answers (both directions)
    if len(gt_s) >= 2 and (gt_s in pred_s or pred_s in gt_s):
        return True

    return False


# ═══════════════════════════════════════════════════════════════════
#  Tool call detection
# ═══════════════════════════════════════════════════════════════════

VALID_TOOLS = {'chartmoe', 'sam3_detect', 'easyocr_tool', 'calculator',
               'diagramformalizer', 'pixel_reasoner',
               # aliases
               'easyocr', 'sam3', 'detect', 'ocr', 'extract_table',
               'calculate', 'crop_zoom'}


def check_tool_usage(text: str) -> bool:
    """Check if response contains at least one valid tool call."""
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    for m in matches:
        try:
            obj = json.loads(m.strip())
            tool_name = (obj.get('tool') or '').strip().lower()
            if tool_name in VALID_TOOLS:
                return True
        except (json.JSONDecodeError, AttributeError):
            continue
    return False


# ═══════════════════════════════════════════════════════════════════
#  Reward computation
# ═══════════════════════════════════════════════════════════════════

def compute_score(completion: str, reference: str) -> Dict[str, Any]:
    """
    Compute reward score.

    Returns:
        dict with keys: reward, answer_correct, tool_used, predicted, format_ok
    """
    # Extract answer
    predicted = extract_answer(completion)
    has_answer = bool(predicted)

    # Check answer correctness
    answer_correct = check_correct(reference, predicted) if has_answer else False

    # Check tool usage
    tool_used = check_tool_usage(completion)

    # Compute reward
    if not has_answer:
        # No <answer> tag → can't evaluate → bad
        reward = -3.0
    elif answer_correct and tool_used:
        # Best: used tool and got it right
        reward = 3.0
    elif answer_correct and not tool_used:
        # OK: got it right without tool (discouraged but not punished)
        reward = 1.0
    else:
        # Wrong answer (regardless of tool usage)
        reward = -3.0

    return {
        'reward': reward,
        'answer_correct': answer_correct,
        'tool_used': tool_used,
        'predicted': predicted,
        'format_ok': has_answer,
    }


# ═══════════════════════════════════════════════════════════════════
#  verl Reward Manager
# ═══════════════════════════════════════════════════════════════════

@register("vl_reasoning_qwen3")
class VLReasoningQwen3RewardManager(AbstractRewardManager):
    """
    Reward manager for tool-augmented Qwen3-VL in verl GRPO.

    Reward table:
      +3  tool_call + answer correct
      +1  no tool_call + answer correct
      -3  answer wrong or no answer
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int = 5,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        # Stats tracking
        self.step_count = 0
        self.stats = Counter()

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:

        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
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

        batch_rewards = []
        already_printed = 0

        for i in range(len(data)):
            result = compute_score(
                completion=sequences_str[i],
                reference=ground_truths[i],
            )

            score = result['reward']
            reward_tensor[i, valid_response_length[i].item() - 1] = score
            batch_rewards.append(score)

            # Track stats
            self.stats['total'] += 1
            if result['answer_correct']:
                self.stats['correct'] += 1
            if result['tool_used']:
                self.stats['tool_used'] += 1
            if not result['format_ok']:
                self.stats['no_answer_tag'] += 1
            self.stats[f'reward_{score:.0f}'] += 1

            # Print examples
            if already_printed < self.num_examine:
                already_printed += 1
                gt = ground_truths[i]
                pred = result['predicted']
                tool = result['tool_used']
                correct = result['answer_correct']
                resp_preview = sequences_str[i][:200].replace('\n', ' ')

                print(f"  [Reward] gt={gt}, pred={pred}, score={score:+.0f}, "
                      f"correct={correct}, tool={tool}")
                if already_printed <= 2:
                    print(f"    Response: {resp_preview}...")

        # Periodic batch stats
        self.step_count += 1
        if self.step_count % 10 == 0 and batch_rewards:
            n = self.stats['total']
            print(f"\n  [Step {self.step_count}] Batch: mean_reward={np.mean(batch_rewards):.2f}, "
                  f"correct={sum(1 for r in batch_rewards if r > 0)}/{len(batch_rewards)}")
            print(f"    Cumulative ({n} samples): "
                  f"correct={self.stats['correct']/n*100:.0f}%, "
                  f"tool_used={self.stats['tool_used']/n*100:.0f}%, "
                  f"no_answer={self.stats['no_answer_tag']/n*100:.0f}%")
            print(f"    Rewards: +3={self.stats.get('reward_3', 0)}, "
                  f"+1={self.stats.get('reward_1', 0)}, "
                  f"-3={self.stats.get('reward_-3', 0)}")

        if return_dict:
            return {"reward_tensor": reward_tensor}
        return reward_tensor


# ═══════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        # +3: tool_call + answer correct
        ('<think>Let me extract the data.</think>\n'
         '<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>\n'
         '<think>The table shows France at 176.</think>\n'
         '<answer>France</answer>',
         "France", 3.0, "tool + correct answer"),

        ('<tool_call>{"tool": "sam3_detect", "task": "detect_objects"}</tool_call>\n'
         '<think>3 cubes detected, subtract 1 cylinder.</think>\n'
         '<answer>2</answer>',
         "2", 3.0, "tool + correct number"),

        # +3: multi-tool + correct
        ('<think>Need table first.</think>\n'
         '<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>\n'
         '<think>Got data. Now calculate sum.</think>\n'
         '<tool_call>{"tool": "calculator", "expression": "170+153+181"}</tool_call>\n'
         '<think>Sum is 504.</think>\n'
         '<answer>504</answer>',
         "504", 3.0, "multi-tool + correct"),

        # +1: no tool + correct answer
        ('<think>I can see France has the longest bar.</think>\n'
         '<answer>France</answer>',
         "France", 1.0, "no tool but correct"),

        ('<think>Counting: 3 objects.</think><answer>3</answer>',
         "3", 1.0, "no tool, simple correct"),

        # +1: numeric tolerance
        ('<think>Reading the value.</think><answer>4.85</answer>',
         "4.9", 1.0, "no tool, within 5% tolerance"),

        # -3: wrong answer with tool
        ('<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>\n'
         '<think>I see Germany at 43.</think>\n'
         '<answer>Germany</answer>',
         "France", -3.0, "tool used but wrong answer"),

        # -3: wrong answer without tool
        ('<think>I think it is 5.</think><answer>5</answer>',
         "3", -3.0, "no tool, wrong answer"),

        # -3: no answer tag
        ('<think>The answer is France.</think>',
         "France", -3.0, "no <answer> tag"),

        # -3: empty response
        ('', "3", -3.0, "empty response"),

        # -3: tool call but no answer (ran out of turns)
        ('<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>\n'
         '<think>I need to calculate...</think>\n'
         '<tool_call>{"tool": "calculator", "expression": "1+2"}</tool_call>',
         "3", -3.0, "tools called but no answer"),

        # -3: format garbage
        ('I think the answer is 3. Final answer: 3',
         "3", -3.0, "no format tags at all"),

        # Edge: number word matching
        ('<think>Three objects.</think><answer>three</answer>',
         "3", 1.0, "number word match"),

        # Edge: integer vs float
        ('<think>Count.</think><answer>3.0</answer>',
         "3", 1.0, "int/float match 3.0 == 3"),

        # Edge: choice letter
        ('<tool_call>{"tool": "easyocr_tool", "task": "read_text"}</tool_call>\n'
         '<think>Option B is correct.</think>\n<answer>B</answer>',
         "B", 3.0, "choice letter match"),

        # Edge: choice letter extracted from text
        ('<think>Looking at options.</think>\n<answer>The answer is B</answer>',
         "B", 1.0, "choice letter in text"),

        # Edge: substring match
        ('<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>\n'
         '<think>France leads.</think>\n<answer>France</answer>',
         "France", 3.0, "text exact match"),

        # Edge: LaTeX in answer
        ('<think>Area.</think><answer>$12\\text{ cm}^2$</answer>',
         "12", 1.0, "LaTeX cleaned to number"),

        # Edge: scale awareness (has tool_call → +3)
        ('<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>\n'
         '<think>Value is 4.2 million.</think>\n<answer>4200000</answer>',
         "4.2", 3.0, "scale: 4200000 vs 4.2 (million) + tool"),

        # Edge: invalid tool_call JSON (still counts as no tool)
        ('<tool_call>not valid json</tool_call>\n'
         '<think>Guessing.</think>\n<answer>5</answer>',
         "5", 1.0, "invalid tool JSON → no tool credit"),

        # Edge: tool_call with wrong tool name (not in whitelist)
        ('<tool_call>{"tool": "fake_tool", "task": "do_stuff"}</tool_call>\n'
         '<think>Result.</think>\n<answer>5</answer>',
         "5", 1.0, "unknown tool → no tool credit"),
    ]

    print("VL Reasoning Qwen3 Reward Tests")
    print(f"  +3 = tool + correct, +1 = no tool + correct, -3 = wrong/no answer")
    print("=" * 70)

    all_pass = True
    for resp, gt, expected, desc in tests:
        result = compute_score(resp, gt)
        score = result['reward']
        status = "PASS" if abs(score - expected) < 0.01 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status}: score={score:+.0f} (exp={expected:+.0f}) | "
              f"correct={result['answer_correct']}, tool={result['tool_used']}, "
              f"pred='{result['predicted'][:30]}' | {desc}")

    print(f"\n  {'All tests passed!' if all_pass else 'SOME TESTS FAILED!'}")
