#!/usr/bin/env python3
"""
GPT-5 Planner for vico GenEnv — high-quality reasoning chains for Rewriter training.

Unlike SimplePlanner (template text), GPT-5 sees the actual chart image and produces
detailed reasoning about what to modify and why. This output is used to:
1. Train a local Rewriter model (like scene's gpt5_planner_scene.py → Rewriter SFT)
2. Generate better trajectories (GPT-5 reasoning > SimplePlanner templates)

Interface compatible with LocalPlanner.get_plan().
"""
import os, json, re, time, base64
from io import BytesIO
from typing import Dict, List, Optional
from PIL import Image as PILImage


class APIPlanner:

    SYSTEM_PROMPT = """You are an expert chart editing planner.

Your task is to choose exactly ONE editing action that will move a Solver model's pass rate (p̂) on a chart question closer to 0.5.

You are given:
- the current chart image
- the question about the chart
- the correct answer
- the current Solver pass rate p̂
- previous edit history and outcomes

Goal: bring p̂ toward [0.35, 0.65], ideally near 0.5.

Available tool:
- chart_rerender(edit_instruction): Modify the chart's underlying data table and re-render.
  The answer will be recomputed automatically from the modified data.

CALIBRATED STRATEGIES (choose based on current p̂):

When p̂ = 1.0 (Solver gets it right every time — need MAJOR changes):
- Add 5-10 new rows with values within 1-2% of the answer value
- Duplicate the answer entity with a slightly different name (e.g., "Nigeria" → add "Niger")
- Change the answer value to be identical to another row's value (force ambiguity)
- Triple the number of data series/columns to overwhelm visual parsing
- Compress ALL values into a narrow range (e.g., all between 40-45 instead of 10-90)

When p̂ = 0.75-0.99 (Solver mostly right — need moderate changes):
- Add 2-3 rows with near-miss values close to the answer
- Add one extra data series with similar values
- Slightly compress the value range to make visual distinctions harder

When p̂ = 0.65-0.75 (almost in sweet zone — need small nudge):
- Add 1 distractor row with a value very close to the answer
- Slightly change the scale or axis range

When p̂ = 0.35-0.65 (IN SWEET ZONE — minimal or no change needed):
- Make very small adjustments if any

When p̂ = 0.25-0.35 (almost in sweet zone — simplify slightly):
- Remove 1-2 confusing rows that have values close to the answer
- Make the answer value slightly more distinct

When p̂ = 0.0-0.25 (Solver always wrong — need to SIMPLIFY significantly):
- Remove ALL rows except the answer row and 1-2 clearly different rows
- Make the answer value stand out (e.g., 80 vs others at 10-20)
- Remove extra data series, keep only the most relevant one
- Simplify labels, reduce chart density drastically

CRITICAL RULES:
1. Be SPECIFIC — describe exactly what data to add/change/remove with concrete values.
2. Match your modification intensity to the p̂ level above.
3. If previous attempts overshot (p̂ went from 1.0 to 0.0), use HALF the modification intensity.
4. If previous attempts undershot (p̂ barely changed), DOUBLE the modification intensity.
5. Learn from the DIRECTION of change in history, not just the final p̂.

Output ONLY a JSON object (no markdown, no backticks):
{"tool": "chart_rerender", "params": {"edit_instruction": "<specific modification>"}, "reasoning": "<detailed reasoning>"}
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
        print(f"vico GPT-5 Planner initialized (model={model})")

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
            prompt += "\nPrevious attempts (each started from modified chart):\n"
            for i, h in enumerate(history, 1):
                instr = h.get("params", {}).get("edit_instruction", "")[:80]
                p_val = h.get("p_hat", "?")
                p_str = f"{p_val:.3f}" if isinstance(p_val, (int, float)) else str(p_val)
                err = h.get("error", "")
                err_str = f" [FAILED: {err[:40]}]" if err else ""
                new_a = h.get("new_answer", "")
                ans_str = f", answer→{new_a}" if new_a else ""
                reasoning = h.get("reasoning", "")
                reason_str = f" — {reasoning[:60]}" if reasoning else ""
                prompt += f'  Attempt {i}: "{instr}..." → p̂={p_str}{ans_str}{err_str}{reason_str}\n'

            phats = [h["p_hat"] for h in history if isinstance(h.get("p_hat"), (int, float))]
            if phats:
                too_hard = sum(1 for p in phats if p < 0.35)
                too_easy = sum(1 for p in phats if p > 0.65)
                overshot = sum(1 for i in range(1, len(phats))
                              if phats[i-1] > 0.65 and phats[i] < 0.35)
                undershot = sum(1 for i in range(1, len(phats))
                               if abs(phats[i] - phats[i-1]) < 0.1)

                if overshot > 0:
                    prompt += (f"\nWARNING: {overshot} attempt(s) overshot — p̂ swung from too-easy to too-hard. "
                              f"Use HALF the modification intensity. Make smaller, more targeted changes.\n")
                elif undershot > 0:
                    prompt += (f"\nWARNING: {undershot} attempt(s) barely changed p̂. "
                              f"DOUBLE the modification intensity. Your changes are too mild.\n")
                elif too_hard and too_easy:
                    prompt += (f"\nResults are oscillating ({too_easy} too easy, {too_hard} too hard). "
                              f"Try a DIFFERENT strategy instead of adjusting intensity.\n")
                elif too_hard:
                    prompt += (f"\n{too_hard} attempt(s) were too aggressive (p̂ < 0.35). "
                              f"SIMPLIFY: remove confusing rows, make the answer more distinct.\n")
                elif too_easy:
                    prompt += (f"\n{too_easy} attempt(s) were too mild (p̂ > 0.65). "
                              f"Be MORE AGGRESSIVE: add more distractors, compress value ranges.\n")
        else:
            prompt += "\nNo previous attempts. This is the first try.\n"

        prompt += "\nAnalyze the chart image carefully, then decide what modification to make. Output JSON only."
        return prompt

    def get_plan(self, image_path, question, answer, p_hat_original,
                 scene_objects=None, history=None, max_retries=3, **kwargs):
        """
        Get an editing plan for a chart.
        Interface compatible with LocalPlanner.get_plan().
        """
        history = history or []
        user_prompt = self._build_user_prompt(question, answer, p_hat_original, history)

        try:
            img_b64 = self._load_image_as_base64(image_path)
        except Exception as e:
            print(f"    [vico GPT5] Cannot load image: {e}")
            return self._fallback_plan(p_hat_original)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "text", "text": user_prompt},
                        ]},
                    ],
                    max_completion_tokens=1024,
                )
                raw = response.choices[0].message.content.strip()
                plan = self._parse_plan(raw)
                if plan:
                    return plan
                print(f"    [vico GPT5] Parse failed ({attempt+1}), raw: {raw[:200]}")
            except Exception as e:
                print(f"    [vico GPT5] API error ({attempt+1}): {e}")
                time.sleep(2 ** attempt)

        return self._fallback_plan(p_hat_original)

    def _parse_plan(self, raw):
        """Parse LLM output into a structured plan."""
        raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
        raw = re.sub(r'\s*```$', '', raw).strip()

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            matches = list(re.finditer(r'\{', raw))
            plan = None
            if matches:
                for m in matches:
                    start = m.start()
                    depth = 0
                    for i in range(start, len(raw)):
                        if raw[i] == '{': depth += 1
                        elif raw[i] == '}': depth -= 1
                        if depth == 0:
                            try:
                                plan = json.loads(raw[start:i+1])
                                break
                            except json.JSONDecodeError:
                                continue
                    if plan:
                        break

            if plan is None:
                ei_match = re.search(r'"edit_instruction"\s*:\s*"([^"]*)', raw)
                r_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', raw)
                if ei_match:
                    plan = {
                        "tool": "chart_rerender",
                        "params": {"edit_instruction": ei_match.group(1)},
                        "reasoning": r_match.group(1)[:200] if r_match else "(parsed)",
                    }

            if plan is None:
                return None

        plan.setdefault("tool", "chart_rerender")
        plan.setdefault("params", {})
        plan.setdefault("reasoning", "")

        if "edit_instruction" not in plan.get("params", {}):
            return None

        return plan

    def _fallback_plan(self, p_hat):
        """Fallback when API fails."""
        if p_hat > 0.65:
            return {"tool": "chart_rerender",
                    "params": {"edit_instruction": "Add 3 rows with values very close to the answer value to create confusion"},
                    "reasoning": f"Fallback: p̂={p_hat:.3f} too high, add near-miss distractors"}
        elif p_hat < 0.35:
            return {"tool": "chart_rerender",
                    "params": {"edit_instruction": "Remove rows with values similar to the answer to reduce confusion"},
                    "reasoning": f"Fallback: p̂={p_hat:.3f} too low, simplify chart"}
        return {"tool": "chart_rerender",
                "params": {"edit_instruction": "Make minimal changes to the data values"},
                "reasoning": f"Fallback: p̂={p_hat:.3f} near target, minimal adjustment"}
