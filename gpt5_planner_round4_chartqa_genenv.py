#!/usr/bin/env python3
"""
Round-4 GPT-5 Planner for ChartQA GenEnv.

Context for this round:
  The Solver has already been GRPO-trained on round2 trajectories. It has
  learned to defeat the round2/round3 bag of tricks (extra distractor rows,
  renamed entities, small numeric perturbations, slight range compression).

  Round2/round3 trajectory yield:
    p̂=0 : ~42%   p̂=1 : ~42%   sweet(0.35-0.65): ~5-7%

  This planner shifts from *quantity-of-distractors* tweaks to
  *structural / semantic* edits that the hardened Solver has not yet
  been trained on. It also refuses to further hurt charts when p̂ is
  already low (delete-to-simplify turned out to push more samples to p̂=0
  without moving them into the sweet zone).

Interface compatible with CLEVRQwen3VLPlanner.get_plan() — drop-in
replacement for `ChartQAGPT5Planner` when the loop selects gpt5 planner mode.
"""

import base64
import json
import os
import re
import time
from io import BytesIO

from PIL import Image as PILImage


class ChartQAGPT5PlannerRound4:

    SYSTEM_PROMPT = """You are an expert chart editing planner (Round 4).

You choose exactly ONE editing action that will move a GRPO-trained Solver's
pass rate (p̂) on a chart question into [0.3, 0.7].

You are given:
- the current chart image
- the question about the chart
- the correct answer
- the current Solver pass rate p̂
- previous edit history and outcomes

IMPORTANT CONTEXT:
The Solver has already been RL-trained on a prior round of chart edits.
It is now strong at:
  - Reading single-value answers off two-column and multi-column bar charts
  - Ignoring simple distractor rows (2-5 extra rows with similar values)
  - Renamed answer entities (e.g., "Nigeria" → "Niger")
  - Small numeric perturbations (±5-10%)
  - Mild value-range compression

These "quantity" tricks are largely DEFEATED. Do not repeat them.

Available tool:
- chart_rerender(edit_instruction): modify the chart's underlying data table
  and re-render. The new answer will be recomputed from the modified data.

=========================================================================
STRATEGY BY p̂
=========================================================================

When p̂ >= 0.7 (Solver still answers correctly — need a STRUCTURAL edit):
Pick ONE of these types. Prefer a type NOT already tried in history.

  (a) Near-tie ambiguity
      Change a non-answer row so its value is within ±1.5% of the answer
      value. Forces the Solver to read tiny numeric differences.

  (b) Semantic alias mismatch
      Rewrite the answer row's entity label as an abbreviation / synonym
      that does NOT textually match the question wording
      (e.g., question "United Kingdom" -> label "UK" / "Britain";
      "automobile" -> "car"; "USD" -> "U.S. dollars").
      Ensures the Solver must resolve semantics, not string-match.

  (c) Unit / scale transform
      Change the measurement unit (% <-> per-100; millions <-> thousands;
      billions <-> millions). Recompute the answer value consistently.
      The Solver must now track the unit in the axis label.

  (d) Aggregate requirement
      Split the answer row into TWO rows whose numeric values SUM to the
      original answer. The Solver now has to add two bars to answer.

  (e) Ordinal reframing
      Insert enough rows so the answer bar is NOT the tallest or
      visually-dominant one. The Solver must sort and count, not just
      pick the biggest.

  (f) Cross-row computation
      Make the answer equal to the DIFFERENCE between two rows
      (e.g., year-over-year change, two-entity gap). Requires subtraction,
      not a single lookup.

  (g) Chart-type stress
      Make the chart visually harder WITHOUT deleting data: rotate axis
      labels, compress the y-axis range to the top 10% of values so
      bar-height differences become tiny, or make non-answer bars share
      colors.

When 0.3 <= p̂ <= 0.7 (IN SWEET ZONE):
  Output tool="no_edit" and stop. Do not touch a chart that's already
  landing in-zone.

When p̂ <= 0.2 (Solver can never get it right — the prior edit or the
  current chart is broken / unreadable):
  Do NOT delete rows. Do NOT "simplify" by gutting data — that creates
  near-trivial charts that don't teach the Solver anything and also
  often end at p̂=0 because the render is still broken. Instead, FIX
  READABILITY at the answer row only:
    - Increase font size of the answer row's label
    - Make only the answer bar a unique color (others uniform/gray)
    - Un-rotate overlapping axis text
    - Remove ONLY purely overlapping / unreadable annotations,
      not data rows
  Keep data content intact.

=========================================================================
HISTORY USE
=========================================================================

Read the history before choosing. Rules:
  - If the LAST attempt stayed in the same extreme zone (p̂ still >0.8
    or still <0.2), SWITCH strategy type — do not repeat the same type
    with a bigger knob.
  - If the history oscillates between too-easy and too-hard, the chosen
    TYPE is unstable — pick a different letter (a)-(g).
  - If a type from (a)-(g) was tried and produced any movement toward
    [0.3, 0.7], keep it and refine the parameter (e.g., shrink the gap
    from 1.5% to 0.8%).
  - Never re-apply an edit type that is listed in the AVOID section of
    this prompt.

=========================================================================
OUTPUT
=========================================================================

Output ONLY a JSON object (no markdown, no backticks):
{"tool": "chart_rerender", "params": {"edit_instruction": "<specific edit>", "edit_type": "<one of: a|b|c|d|e|f|g|no_edit|readability>"}, "reasoning": "<why this edit type, referencing history>"}

For the sweet-zone case, use:
{"tool": "no_edit", "params": {"edit_instruction": "p_hat already in sweet zone", "edit_type": "no_edit"}, "reasoning": "p̂ in [0.3, 0.7]"}
"""

    def __init__(self, api_key=None, model="gpt-5"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY env var or pass api_key.")
        self.model = model

        base_url = os.environ.get("OPENAI_BASE_URL")
        kwargs = {"api_key": self.api_key}
        if base_url:
            kwargs["base_url"] = base_url

        from openai import OpenAI
        self.client = OpenAI(**kwargs)
        print(f"ChartQA Round4 GPT-5 Planner initialized (model={model})")

    def _load_image_as_base64(self, image_path):
        with PILImage.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), PILImage.Resampling.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _build_user_prompt(self, question, answer, p_hat, history):
        prompt = f"""Question: {question}
Answer: {answer}
Current Solver pass rate (p̂): {p_hat:.3f}
"""
        if history:
            prompt += "\nPrevious attempts on this sample (each edit was applied on top of the previous chart):\n"
            for i, h in enumerate(history, 1):
                params = h.get("params", {})
                instr = params.get("edit_instruction", "")[:100]
                etype = params.get("edit_type", "?")
                p_val = h.get("p_hat", "?")
                p_str = f"{p_val:.3f}" if isinstance(p_val, (int, float)) else str(p_val)
                err = h.get("error", "")
                err_str = f" [FAILED: {err[:40]}]" if err else ""
                new_a = h.get("new_answer", "")
                ans_str = f", answer→{new_a}" if new_a else ""
                prompt += (f"  Attempt {i}: type={etype}  \"{instr}...\"  "
                           f"→ p̂={p_str}{ans_str}{err_str}\n")

            types_tried = [h.get("params", {}).get("edit_type") for h in history]
            types_tried = [t for t in types_tried if t and t not in ("no_edit", "readability", "?")]
            if types_tried:
                prompt += f"\nTypes already tried: {types_tried}. Prefer a type NOT in this list.\n"

            phats = [h["p_hat"] for h in history if isinstance(h.get("p_hat"), (int, float))]
            if phats:
                stuck_high = all(p > 0.8 for p in phats)
                stuck_low = all(p < 0.2 for p in phats)
                if stuck_high:
                    prompt += ("\nAll attempts left p̂ > 0.8. The Solver is beating every edit type "
                               "tried so far. Switch to a structurally DIFFERENT type (d, e, or f) "
                               "and make the edit uncompromising.\n")
                elif stuck_low:
                    prompt += ("\nAll attempts left p̂ < 0.2. Previous edits broke the chart's "
                               "readability instead of raising difficulty. Switch to readability "
                               "repair (see p̂ <= 0.2 branch) — do NOT add more perturbations.\n")
        else:
            prompt += "\nNo previous attempts on this sample.\n"

        prompt += "\nLook at the chart image carefully. Pick a strategy and output JSON only."
        return prompt

    def get_plan(self, image_path, question, answer, p_hat_original,
                 scene_objects=None, history=None, max_retries=3, **kwargs):
        """
        Interface compatible with CLEVRQwen3VLPlanner.get_plan().
        Returns a dict with keys: tool, params, reasoning.
        """
        history = history or []
        user_prompt = self._build_user_prompt(question, answer, p_hat_original, history)

        b64_image = None
        if image_path and os.path.exists(image_path):
            try:
                b64_image = self._load_image_as_base64(image_path)
            except Exception as e:
                print(f"[Round4Planner] image load failed: {e}")

        user_content = [{"type": "text", "text": user_prompt}]
        if b64_image:
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
            })

        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                )
                raw = resp.choices[0].message.content or ""
                plan = self._parse_json(raw)
                if plan is not None:
                    plan.setdefault("params", {})
                    plan["params"].setdefault("edit_instruction",
                                              "No instruction returned")
                    plan["params"].setdefault("edit_type", "?")
                    plan.setdefault("reasoning", "")
                    plan.setdefault("tool", "chart_rerender")
                    return plan
                last_err = f"JSON parse failed: {raw[:120]}"
            except Exception as e:
                last_err = str(e)
                print(f"[Round4Planner] attempt {attempt+1} error: {last_err[:120]}")
                time.sleep(1.5 * (attempt + 1))

        # Fallback plan so the loop keeps going.
        return {
            "tool": "chart_rerender",
            "params": {
                "edit_instruction": (
                    "Add a near-tie distractor row whose value is within 1.5% "
                    "of the answer value."
                ),
                "edit_type": "a",
            },
            "reasoning": f"planner_error_fallback: {last_err}",
        }

    @staticmethod
    def _parse_json(text):
        if not text:
            return None
        # Strip code fences if present.
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                          flags=re.MULTILINE)
        # Try direct parse.
        try:
            return json.loads(stripped)
        except Exception:
            pass
        # Try to locate the first balanced JSON object.
        depth = 0
        start = -1
        for i, ch in enumerate(stripped):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = stripped[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        start = -1
        return None
