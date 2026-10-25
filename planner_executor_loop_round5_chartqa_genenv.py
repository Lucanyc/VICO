#!/usr/bin/env python3
"""
Round-5 Planner-Executor-Solver Loop for ChartQA GenEnv.

v3: grid sampling with seed-driven variants (replaces bisection).

Why: v2 bisection converged to the p̂ cliff boundary, producing mostly
     p̂ ∈ {0, 1} (zero GRPO advantage). Determinism removed the exploration
     that GPT-5.2's randomness accidentally provided.
How: For each sample × edit type, sweep a small grid of κ values and at
     each κ generate multiple seed variants. Each variant is a different
     (deterministic-given-seed) edit. Pool all turns; downstream GRPO
     builder keeps those with p̂ ∈ (0, 1).

Turns per sample = len(KAPPA_GRID) × SEEDS_PER_KAPPA (default 3×3 = 9).
All turns are written to the trajectory with `exec_success=True` so
build_solver_grpo_data_chartqa.py can harvest them.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chartqa_scene_generator_round5 import (
    ChartQASceneGeneratorR5,
    is_max_question,
    is_min_question,
    is_superlative_question,
)


# =============================================================================
# Utilities
# =============================================================================

def atomic_json_save(data, filepath):
    tmpf = filepath + ".tmp"
    with open(tmpf, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    try:
        os.replace(tmpf, filepath)
    except OSError:
        with open(filepath, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def verify_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def clean_answer(text):
    if not text:
        return ""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    for pat in [r'\\boxed\{', r'\\boxed \{']:
        m2 = re.search(pat, text)
        if m2:
            s, d, i = m2.end(), 1, m2.end()
            while i < len(text) and d > 0:
                if text[i] == '{':
                    d += 1
                elif text[i] == '}':
                    d -= 1
                i += 1
            if d == 0:
                return text[s:i - 1].strip()
    text = text.strip()
    for p in ["The answer is:", "The answer is", "Answer:", "Answer"]:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()
            break
    text = re.sub(r'[\\%$]', '', text.rstrip('.')).strip()
    text = re.sub(
        r'\b(million|billion|trillion|thousand|percent|approximately|about|around|nearly|over|'
        r'kg|km|miles|meters|dollars|euros|pounds|yen|yuan|rupees|'
        r'australian|american|british|canadian|chinese|indian|japanese|'
        r'us|usd|eur|gbp|cad|aud)\b',
        '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def strip_thinking(text):
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    return re.sub(r'<think>[\s\S]*$', '', text).strip()


def check_correct(gt, pred):
    gt_s = str(gt).strip().lower()
    pred_s = clean_answer(str(pred)).lower()
    if not pred_s:
        return False
    if gt_s == pred_s:
        return True
    try:
        gn = float(gt_s.replace(',', '').replace('%', ''))
        pn = float(pred_s.replace(',', '').replace('%', ''))
        if gn != 0 and abs(gn - pn) / abs(gn) <= 0.005:
            return True
        for scale in [1e3, 1e6, 1e9]:
            if gn != 0 and abs(gn * scale - pn) / abs(gn * scale) <= 0.005:
                return True
            if pn != 0 and abs(pn * scale - gn) / abs(pn * scale) <= 0.005:
                return True
    except Exception:
        pass
    return False


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ChartContext:
    csv_path: str
    question: str
    n_rows: int
    n_cols: int
    value_min: float
    value_max: float
    value_range: float
    answer_value: Optional[float]
    row_labels: List[str]
    col_labels: List[str]


@dataclass
class EditSpec:
    type_name: str
    kappa: float
    params: Dict
    render_config: Optional[Dict] = None
    log: str = ""


@dataclass
class FeasibilityResult:
    is_feasible: bool
    reason: str = ""
    kappa_max: float = 1.0


def load_chart_context(csv_path: str, question: str, answer: str) -> ChartContext:
    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    label_col = df.columns[0] if df.columns[0] not in numeric_cols else None
    row_labels = (df[label_col].astype(str).tolist()
                  if label_col else [str(i) for i in range(len(df))])
    if numeric_cols:
        vals = df[numeric_cols].values.flatten()
        value_min = float(vals.min())
        value_max = float(vals.max())
        value_range = value_max - value_min
    else:
        value_min = value_max = value_range = 0.0
    try:
        answer_value = float(str(answer).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        answer_value = None
    return ChartContext(
        csv_path=csv_path,
        question=question,
        n_rows=len(df),
        n_cols=len(numeric_cols),
        value_min=value_min,
        value_max=value_max,
        value_range=value_range,
        answer_value=answer_value,
        row_labels=row_labels,
        col_labels=numeric_cols,
    )


# =============================================================================
# EditType classes (parameterize κ → params; stay as in v2)
# =============================================================================

class EditType:
    name = "base"
    description = ""

    def kappa_max(self, ctx: ChartContext) -> float:
        return 1.0

    def feasibility(self, kappa: float, ctx: ChartContext) -> FeasibilityResult:
        kmax = self.kappa_max(ctx)
        if kmax < 1e-3:
            return FeasibilityResult(False, f"{self.name}: κ_max=0", 0.0)
        if kappa > kmax + 1e-6:
            return FeasibilityResult(False,
                                     f"{self.name}: κ={kappa:.2f} > κ_max={kmax:.2f}",
                                     kmax)
        return FeasibilityResult(True, "", kmax)

    def compute_params(self, kappa: float, ctx: ChartContext) -> EditSpec:
        raise NotImplementedError


class NearTieEdit(EditType):
    """κ → ε range. Sign constraint decided from question type."""
    name = "near_tie"
    EPS_EASY = 5.0
    EPS_HARD = 0.3

    def kappa_max(self, ctx):
        if ctx.n_rows < 2 or ctx.answer_value is None or ctx.answer_value == 0:
            return 0.0
        min_eps = 100.0 * 0.5 / abs(ctx.answer_value)
        if min_eps >= self.EPS_EASY:
            return 0.0
        if min_eps <= self.EPS_HARD:
            return 1.0
        return (self.EPS_EASY - min_eps) / (self.EPS_EASY - self.EPS_HARD)

    def compute_params(self, kappa, ctx):
        kappa = max(0.0, min(self.kappa_max(ctx), kappa))
        eps = self.EPS_EASY * (1 - kappa) + self.EPS_HARD * kappa
        if is_max_question(ctx.question):
            sign_constraint = "below"
        elif is_min_question(ctx.question):
            sign_constraint = "above"
        else:
            sign_constraint = "any"
        return EditSpec(
            type_name=self.name, kappa=kappa,
            params={"eps_pct": round(eps, 3), "sign_constraint": sign_constraint},
            log=f"near_tie ε={eps:.2f}% sc={sign_constraint}",
        )


class DistractorCountEdit(EditType):
    name = "distractor_count"
    K_EASY = 1
    K_HARD = 6

    def kappa_max(self, ctx):
        if ctx.n_cols == 0 or ctx.value_range <= 0:
            return 0.0
        max_k = min(self.K_HARD, 14 - ctx.n_rows)
        if max_k < self.K_EASY:
            return 0.0
        return (max_k - self.K_EASY) / (self.K_HARD - self.K_EASY)

    def compute_params(self, kappa, ctx):
        kappa = max(0.0, min(self.kappa_max(ctx), kappa))
        k = self.K_EASY + int(round(kappa * (self.K_HARD - self.K_EASY)))
        if is_max_question(ctx.question):
            mode = "below_ans"
        elif is_min_question(ctx.question):
            mode = "above_ans"
        else:
            mode = "uniform"
        return EditSpec(
            type_name=self.name, kappa=kappa,
            params={"k": k, "value_mode": mode},
            log=f"distractor k={k} mode={mode}",
        )


class OrdinalRankEdit(EditType):
    name = "ordinal_rank"
    R_EASY = 2
    R_HARD = 5

    def kappa_max(self, ctx):
        if ctx.answer_value is None:
            return 0.0
        if is_superlative_question(ctx.question):
            return 0.0
        max_r = min(self.R_HARD, 13 - ctx.n_rows + 1)
        if max_r < self.R_EASY:
            return 0.0
        return (max_r - self.R_EASY) / (self.R_HARD - self.R_EASY)

    def feasibility(self, kappa, ctx):
        if is_superlative_question(ctx.question):
            return FeasibilityResult(False, "ordinal_rank unsafe for superlative Q", 0.0)
        return super().feasibility(kappa, ctx)

    def compute_params(self, kappa, ctx):
        kappa = max(0.0, min(self.kappa_max(ctx), kappa))
        r = self.R_EASY + int(round(kappa * (self.R_HARD - self.R_EASY)))
        return EditSpec(
            type_name=self.name, kappa=kappa,
            params={"rank": r},
            log=f"ordinal rank={r}",
        )


class YCompressEdit(EditType):
    name = "y_compress"
    Q_EASY = 0.0
    Q_HARD = 0.90

    def kappa_max(self, ctx):
        if ctx.n_cols == 0 or ctx.value_range <= 0 or ctx.value_min < 0:
            return 0.0
        return 1.0

    def compute_params(self, kappa, ctx):
        kappa = max(0.0, min(self.kappa_max(ctx), kappa))
        q = self.Q_EASY * (1 - kappa) + self.Q_HARD * kappa
        y_min = ctx.value_min + q * (ctx.value_max - ctx.value_min)
        return EditSpec(
            type_name=self.name, kappa=kappa,
            params={"quantile": round(q, 3), "y_axis_min": round(y_min, 3)},
            render_config={"y_axis_min": float(y_min)},
            log=f"y_compress q={q:.2f}",
        )


EDIT_TYPES: Dict[str, EditType] = {
    et.name: et for et in [
        NearTieEdit(), DistractorCountEdit(), OrdinalRankEdit(), YCompressEdit()
    ]
}


def available_types(ctx: ChartContext, min_kappa_max: float = 0.1) -> List[str]:
    return [n for n, et in EDIT_TYPES.items() if et.kappa_max(ctx) >= min_kappa_max]


# =============================================================================
# Planners (type picker)
# =============================================================================

class SimpleTypePlanner:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def pick_type(self, ctx, p_hat, available, history):
        tried = {h.get("type_name") for h in history if h.get("type_name")}
        untried = [t for t in available if t not in tried]
        pool = untried if untried else available
        return self.rng.choice(pool) if pool else "near_tie"


class GPT5TypePlanner:
    SYSTEM_PROMPT = """You pick ONE chart-edit TYPE to create useful GRPO training signal (p̂ ∈ (0,1), not 0 or 1).

You do NOT set parameters. Downstream code sweeps a κ grid and multiple
seed variants per κ. Your job is to pick the single most promising TYPE.

Types:
  - near_tie: add a distractor row whose value is ε% from the answer
  - distractor_count: append k plausible distractor rows
  - ordinal_rank: append distractors so the answer is the r-th largest
                  (filtered out automatically on superlative questions)
  - y_compress: compress y-axis origin (visual-only, no CSV change)

Heuristics:
  - Lookup Qs (most common) → near_tie or distractor_count
  - Small/medium charts with room → ordinal_rank
  - Already-hard charts → y_compress (gentlest)
  - Prefer a type NOT in the tried list

Output ONLY: {"type": "<name>", "reasoning": "<one sentence>"}
"""

    def __init__(self, model="gpt-5", api_key=None):
        from openai import OpenAI
        self.model = model
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY")
        kw = {"api_key": api_key}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kw["base_url"] = base_url
        self.client = OpenAI(**kw)
        print(f"GPT-5 type planner initialized (model={model})")

    def pick_type(self, ctx, p_hat, available, history):
        tried = sorted({h.get("type_name") for h in history if h.get("type_name")} - {None})
        user = (
            f"Chart: {ctx.n_rows} rows x {ctx.n_cols} value cols; "
            f"value range [{ctx.value_min:.2f}, {ctx.value_max:.2f}]; "
            f"answer value {ctx.answer_value}.\n"
            f"Question: {ctx.question[:120]}\n"
            f"Solver p̂ on original: {p_hat:.3f}\n"
            f"Available types: {available}\n"
            f"Already tried: {tried if tried else 'none'}\n\n"
            f"Output JSON only."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            )
            raw = resp.choices[0].message.content or ""
            m = re.search(r'\{[\s\S]*?\}', raw)
            if m:
                out = json.loads(m.group(0))
                t = str(out.get("type", "")).strip()
                if t in available:
                    return t
                print(f"  [GPT5] invalid type '{t}', falling back")
        except Exception as e:
            print(f"  [GPT5] error: {str(e)[:100]}")
        tried_set = {h.get("type_name") for h in history if h.get("type_name")}
        for t in available:
            if t not in tried_set:
                return t
        return available[0]


# =============================================================================
# Solver
# =============================================================================

class SolverEvaluator:
    SYSTEM_PROMPT = (
        "Please answer the question about the chart image. "
        "Think step by step in <think> tags, then give your final answer in <answer> tags."
    )

    def __init__(self, model_path, n_rollouts=8, temperature=0.7):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        self.n_rollouts = n_rollouts
        orig = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        devs = orig.split(",") if orig else []
        solver_gpu = devs[1] if len(devs) >= 2 else "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = solver_gpu
        print(f"Loading Solver: {model_path} (GPU {solver_gpu})")
        self.llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=4096,
                       trust_remote_code=True, gpu_memory_utilization=0.80)
        os.environ["CUDA_VISIBLE_DEVICES"] = orig
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=1024,
                                              n=n_rollouts)
        self.processor = AutoProcessor.from_pretrained(model_path)
        print(f"Solver ready (n_rollouts={n_rollouts}, T={temperature})")

    def evaluate(self, image_path, question, answer):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}]},
        ]
        prompt = self.processor.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True)
        try:
            pil = Image.open(image_path).convert("RGB")
            max_pixels = 4096 * 4096
            w, h = pil.size
            if w * h > max_pixels:
                ratio = (max_pixels / (w * h)) ** 0.5
                pil = pil.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        except Exception as e:
            print(f"    Cannot open image: {e}")
            return 0.0
        try:
            outs = self.llm.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": pil}}],
                self.sampling_params,
            )
        except ValueError as e:
            print(f"    Image incompatible: {str(e)[:80]}")
            return 0.0
        correct = 0
        for i, c in enumerate(outs[0].outputs):
            raw = strip_thinking(c.text)
            pred = clean_answer(raw)
            ok = check_correct(answer, raw)
            if ok:
                correct += 1
            print(f"    rollout {i}: pred='{pred[:40]}' gt='{answer}' correct={ok}")
        p = correct / self.n_rollouts
        print(f"    → {correct}/{self.n_rollouts} correct, p̂={p:.3f}")
        return p


# =============================================================================
# Main loop — grid sampling (κ × seed variant)
# =============================================================================

def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def run_loop(type_planner, generator, solver, sample,
             tables_dir, png_dir, output_dir,
             kappa_grid: List[float],
             seeds_per_kappa: int,
             target_lo: float = 0.125,
             target_hi: float = 0.875):
    sample_id = sample.get("_id", "unknown")
    imgname = sample["imgname"]
    question = sample["query"]
    original_answer = str(sample["label"])
    tier = sample.get("tier", 2)
    answer_cell = sample.get("answer_cell")

    basename = os.path.splitext(imgname)[0]
    csv_path = os.path.join(tables_dir, f"{basename}.csv")
    img_path = os.path.join(png_dir, imgname)

    traj = {
        "sample_id": sample_id, "imgname": imgname,
        "question": question[:100], "original_answer": original_answer,
        "tier": tier, "answer_cell": answer_cell,
        "turns": [], "status": "pending",
        "edit_type": None,
        "p_hat_original": None,
        "final_phat": None, "final_image_path": None, "final_answer": None,
        "total_turns": 0,
        "useful_turns": 0,
    }

    if not os.path.exists(img_path):
        traj["status"] = "failed"; traj["error"] = f"Image not found"
        return traj
    if not os.path.exists(csv_path):
        traj["status"] = "failed"; traj["error"] = f"CSV not found"
        return traj

    print(f"  [Original] Evaluating...")
    p0 = solver.evaluate(img_path, question, original_answer)
    traj["p_hat_original"] = p0

    # Skip cases with no learning value from editing:
    if p0 < 0.1:
        traj.update({"status": "too_hard_original", "final_phat": p0,
                     "final_image_path": img_path, "final_answer": original_answer})
        return traj
    # (p0 already in sweet band → still run grid, gives us richer data at various κ)
    if target_lo <= p0 <= target_hi:
        # Original is already useful — keep it as a turn-0 baseline but still
        # continue with grid to try harder variants.
        traj["note"] = "original_already_in_band"

    try:
        ctx = load_chart_context(csv_path, question, original_answer)
    except Exception as e:
        traj["status"] = "failed"; traj["error"] = f"CSV load failed: {e}"
        return traj

    avail = available_types(ctx, min_kappa_max=0.1)
    if not avail:
        traj.update({"status": "no_feasible_type", "final_phat": p0,
                     "final_image_path": img_path, "final_answer": original_answer})
        return traj

    type_name = type_planner.pick_type(ctx, p0, avail, traj["turns"])
    if type_name not in EDIT_TYPES:
        type_name = avail[0]
    et = EDIT_TYPES[type_name]
    traj["edit_type"] = type_name
    kmax = et.kappa_max(ctx)
    print(f"  [Plan] type={type_name} (available={avail}, κ_max={kmax:.2f})")

    # Grid sampling: (κ × variant). All turns recorded, no early exit.
    turn_idx = 0
    for kappa_raw in kappa_grid:
        # Clip κ to feasible range for this chart+type
        kappa = min(kappa_raw, kmax)
        feas = et.feasibility(kappa, ctx)
        if not feas.is_feasible:
            print(f"  [κ={kappa_raw:.2f}] infeasible even after clipping → skip")
            continue

        for variant in range(seeds_per_kappa):
            turn_idx += 1
            spec = et.compute_params(kappa, ctx)
            edit_seed = abs(hash(f"{sample_id}|{type_name}|κ{kappa:.3f}|v{variant}")) % 100000

            print(f"  [Turn {turn_idx}] κ={kappa:.3f} variant={variant} "
                  f"seed={edit_seed}  {spec.log}")

            mod_path = os.path.join(output_dir, f"{sample_id}_t{turn_idx}.png")
            edit_spec_dict = {
                "type_name": spec.type_name,
                "params": spec.params,
                "render_config": spec.render_config,
            }
            gen_result = generator.generate_deterministic(
                csv_path=csv_path,
                question=question,
                original_answer=original_answer,
                answer_cell=answer_cell,
                tier=tier,
                edit_spec=edit_spec_dict,
                output_path=mod_path,
                sample_id=sample_id,
                edit_seed=edit_seed,
            )

            if not gen_result.get("success"):
                err = (gen_result.get("error") or "?")[:120]
                print(f"    EDIT/RENDER FAIL: {err}")
                traj["turns"].append({
                    "turn": turn_idx, "type_name": type_name,
                    "kappa": kappa, "variant": variant, "edit_seed": edit_seed,
                    "params": spec.params, "render_config": spec.render_config,
                    "status": "render_fail", "exec_success": False,
                    "error": err, "p_hat": None,
                })
                continue

            if not verify_image(mod_path):
                print(f"    image verify failed")
                traj["turns"].append({
                    "turn": turn_idx, "type_name": type_name,
                    "kappa": kappa, "variant": variant, "edit_seed": edit_seed,
                    "params": spec.params,
                    "status": "image_invalid", "exec_success": False,
                    "p_hat": None,
                })
                continue

            new_answer = gen_result["new_answer"]
            n_before = gen_result.get("n_rows_before")
            n_after = gen_result.get("n_rows_after")
            print(f"    rows {n_before}→{n_after}, answer {original_answer}→{new_answer}")

            ph = solver.evaluate(mod_path, question, str(new_answer))

            traj["turns"].append({
                "turn": turn_idx, "type_name": type_name,
                "kappa": kappa, "variant": variant, "edit_seed": edit_seed,
                "params": spec.params, "render_config": spec.render_config,
                "status": "ok", "exec_success": True,
                "modified_image": mod_path,
                "new_answer": new_answer,
                "n_rows_before": n_before, "n_rows_after": n_after,
                "p_hat": ph,
            })
            if 0 < ph < 1:
                traj["useful_turns"] += 1

    # --- Trajectory-level summary ---
    traj["total_turns"] = len(traj["turns"])

    # Pick the "representative" final from useful turns (closest to 0.5).
    useful = [tn for tn in traj["turns"]
              if isinstance(tn.get("p_hat"), float) and 0 < tn["p_hat"] < 1]
    if useful:
        best = min(useful, key=lambda t: abs(t["p_hat"] - 0.5))
        traj.update({
            "status": "success",
            "final_phat": best["p_hat"],
            "final_image_path": best.get("modified_image"),
            "final_answer": best.get("new_answer"),
        })
    else:
        # No useful turn — still report final as best attempt's image if any
        any_ok = [tn for tn in traj["turns"]
                  if isinstance(tn.get("p_hat"), float)]
        if any_ok:
            traj.update({
                "status": "partial",
                "final_phat": any_ok[-1]["p_hat"],
                "final_image_path": any_ok[-1].get("modified_image"),
                "final_answer": any_ok[-1].get("new_answer"),
            })
        else:
            traj["status"] = "all_failed"
    return traj


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--tables_dir", required=True)
    parser.add_argument("--png_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_rollouts", type=int, default=8)
    # v3: no max_turns; turns = len(kappa_grid) × seeds_per_kappa
    parser.add_argument("--kappa_grid", type=str, default="0.25,0.5,0.75",
                        help="Comma-separated κ values to probe per sample.")
    parser.add_argument("--seeds_per_kappa", type=int, default=3,
                        help="Number of seed variants to generate at each κ.")
    parser.add_argument("--target_lo", type=float, default=0.125,
                        help="For summary reporting only.")
    parser.add_argument("--target_hi", type=float, default=0.875,
                        help="For summary reporting only.")
    parser.add_argument("--planner_mode", default="gpt5", choices=["gpt5", "simple"])
    parser.add_argument("--planner_model", default="gpt-5")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    kappa_grid = parse_float_list(args.kappa_grid)
    if not kappa_grid:
        raise ValueError("--kappa_grid must have at least one value")
    print(f"Grid: κ ∈ {kappa_grid}, seeds/κ = {args.seeds_per_kappa}, "
          f"turns/sample = {len(kappa_grid) * args.seeds_per_kappa}")

    with open(args.data_json) as f:
        all_data = json.load(f)
    samples = all_data[args.start:args.start + args.n]
    for i, s in enumerate(samples):
        s["_id"] = f"chart_r5_{args.start + i}"
    print(f"Samples: {len(samples)} (index {args.start}-{args.start + args.n - 1})")

    trajs, done = [], set()
    if args.resume and os.path.exists(args.output_json):
        with open(args.output_json) as f:
            trajs = json.load(f)
        done = set(t["sample_id"] for t in trajs)
        print(f"Resume: {len(trajs)} done")

    todo = [(s["_id"], s) for s in samples if s["_id"] not in done]
    print(f"To process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    if args.planner_mode == "gpt5":
        planner = GPT5TypePlanner(model=args.planner_model)
    else:
        planner = SimpleTypePlanner(seed=args.seed)

    generator = ChartQASceneGeneratorR5()
    solver = SolverEvaluator(args.model_path, args.n_rollouts)

    for idx, (sid, s) in enumerate(todo):
        print(f"\n{'─' * 60}\n[{idx + 1}/{len(todo)}] {sid}")
        print(f"  Q: {s['query'][:70]}\n  A: {s['label']}, tier={s.get('tier', '?')}")
        tr = run_loop(planner, generator, solver, s,
                      args.tables_dir, args.png_dir, args.output_dir,
                      kappa_grid=kappa_grid,
                      seeds_per_kappa=args.seeds_per_kappa,
                      target_lo=args.target_lo, target_hi=args.target_hi)
        trajs.append(tr)
        useful = tr.get("useful_turns", 0)
        total = tr.get("total_turns", 0)
        print(f"  → {tr['status']} | useful={useful}/{total}  "
              f"p̂_orig={tr.get('p_hat_original')}  edit_type={tr.get('edit_type')}")
        atomic_json_save(trajs, args.output_json)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}\nRESULTS\n{'=' * 60}")
    sc = Counter(t["status"] for t in trajs)
    print(f"  Total samples: {len(trajs)}, Status: {dict(sc)}")

    all_turns = [tn for t in trajs for tn in t.get("turns", [])]
    valid_turns = [tn for tn in all_turns
                   if isinstance(tn.get("p_hat"), float)]
    useful_turns = [tn for tn in valid_turns if 0 < tn["p_hat"] < 1]
    print(f"  Total turns: {len(all_turns)}  "
          f"(valid p̂: {len(valid_turns)}, useful 0<p̂<1: {len(useful_turns)})")
    if valid_turns:
        phats = [tn["p_hat"] for tn in valid_turns]
        print(f"  p̂ mean: {sum(phats)/len(phats):.3f}")
        dist = Counter(round(p, 3) for p in phats)
        print(f"  p̂ distribution:")
        for p, c in sorted(dist.items()):
            bar = "█" * min(c, 40)
            print(f"    p̂={p:.3f}: {c:3d} {bar}")

    in_band = [tn for tn in valid_turns
               if args.target_lo <= tn["p_hat"] <= args.target_hi]
    print(f"  In target band [{args.target_lo}, {args.target_hi}]: "
          f"{len(in_band)}/{len(valid_turns)} turns")

    # Per-type
    type_stats = {}
    for tn in all_turns:
        n = tn.get("type_name", "?")
        st = type_stats.setdefault(n, {"total": 0, "useful": 0, "render_fail": 0})
        st["total"] += 1
        if tn.get("status") == "render_fail":
            st["render_fail"] += 1
        ph = tn.get("p_hat")
        if isinstance(ph, float) and 0 < ph < 1:
            st["useful"] += 1
    if type_stats:
        print(f"\n  Per-type:")
        for n, st in sorted(type_stats.items()):
            rate = st["useful"] / st["total"] if st["total"] else 0
            print(f"    {n:20s} {st['useful']:3d}/{st['total']:3d} useful "
                  f"({100*rate:5.1f}%), {st['render_fail']} render_fail")

    print(f"\n  Saved: {args.output_json}")


if __name__ == "__main__":
    main()
