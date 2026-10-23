"""Explicit answer extraction and configurable exact/numeric scoring."""
import math
import re

PROMPTS = {
    "boxed": r"Answer the question about the image. Reason step by step and put your final answer in \boxed{}.",
    "answer_tag": "Answer the question about the image. Think step by step in <think> tags, then give the final answer in <answer> tags.",
    "direct": "Answer the question about the image. Output only the final answer.",
}


def extract_answer(text, style="boxed"):
    if style == "direct":
        return text.strip()
    if style == "answer_tag":
        matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return matches[-1].strip() if matches else None
    last = None
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth, start, i = 1, match.end(), match.end()
        while i < len(text) and depth:
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        if depth == 0:
            last = text[start:i - 1].strip()
    return last


def matches(pred, answer, metric="relaxed", tolerance=0.05):
    if pred is None:
        return False
    pred, answer = str(pred).strip().casefold(), str(answer).strip().casefold()
    pred = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", pred).strip("$ ")
    if pred == answer:
        return True
    if metric == "exact":
        return False
    if metric != "relaxed":
        raise ValueError(f"Unknown metric: {metric}")
    try:
        p, a = float(pred.replace(",", "")), float(answer.replace(",", ""))
        return (math.isfinite(p) and math.isfinite(a)
                and abs(p - a) <= abs(a) * tolerance + 1e-9)
    except ValueError:
        return False
