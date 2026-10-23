#!/usr/bin/env python3
"""
Round-5 VICO scene generator with DETERMINISTIC-given-seed editors.

Key v3 change (vs. v2 deterministic-only):
  Previously, same (sample, κ) always produced the same edit. This made the
  p̂ response binary (solver's output is stable across rollouts, so no middle
  ground appeared). We lost the useful random exploration that GPT-5.2
  provided in round 4.

  v3 keeps determinism GIVEN a seed, but exposes an explicit `edit_seed`
  argument. The loop varies seed across (κ, variant) grid cells so each
  sample gets multiple independent edit variants. Seed-level variation:
    • NearTieEditor: randomizes which non-answer row is picked, the sign
      (within question constraints), and the exact offset magnitude ∈ [0.3,1]·ε.
    • DistractorCountEditor / OrdinalRankEditor: already use seed for values;
      also randomize label suffixes.

Interface:
  gen = TypedTableEditor()
  result = gen.generate_deterministic(
      csv_path=..., question=..., original_answer=...,
      answer_cell={"row": r, "col": c}, tier=1,
      edit_spec={"type_name": "near_tie",
                 "params": {"eps_pct": 2.0, "sign_constraint": "any"},
                 "render_config": None},
      output_path=..., sample_id=...,
      edit_seed=42,          # NEW: controls editor randomization
  )

Answer guarantee: our editors never touch the answer row's value at ans_col.
The generator still runs an invariant check that rejects any edit that
drifts the answer by >1% (catches bugs). new_answer == original_answer
by construction. No GPT-5.2 calls in the edit path.
"""

import os
import random
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from table_editor import TableEditor, RANDOM_PALETTE


# =============================================================================
# Helpers
# =============================================================================

_NUMERIC_CLEAN_RE = re.compile(r'[$%,MBKmbk\s]')


def _parse_num(s):
    if s is None:
        return None
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        try:
            return float(s)
        except Exception:
            return None
    try:
        cleaned = _NUMERIC_CLEAN_RE.sub('', str(s)).strip()
        if cleaned == "" or cleaned == "-":
            return None
        return float(cleaned)
    except Exception:
        return None


def _col_numeric_range(df, col):
    if col not in df.columns:
        return None
    vals = []
    for v in df[col].tolist():
        x = _parse_num(v)
        if x is not None:
            vals.append(x)
    if not vals:
        return None
    return min(vals), max(vals)


SUPERLATIVE_MAX_KEYS = ["maximum", "max ", "largest", "highest", "most",
                        "greatest", "biggest", "top "]
SUPERLATIVE_MIN_KEYS = ["minimum", "min ", "smallest", "lowest", "least", "fewest"]


def is_max_question(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in SUPERLATIVE_MAX_KEYS)


def is_min_question(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in SUPERLATIVE_MIN_KEYS)


def is_superlative_question(question: str) -> bool:
    return is_max_question(question) or is_min_question(question)


# =============================================================================
# Deterministic-given-seed CSV editors
# =============================================================================

class _Editor:
    name = "base"

    def apply(self, df, answer_cell, entity_value, orig_ans_val, params, seed):
        raise NotImplementedError


class NearTieEditor(_Editor):
    """Set one non-answer row's value within ±ε% of the answer.

    Seed-driven choices:
      • which non-answer row gets modified (among numeric candidates)
      • sign: '+' (above) / '-' (below) within the caller-provided constraint
      • exact offset magnitude, uniform in [0.3*eps, 1.0*eps]

    params:
      eps_pct         : float — upper bound on |offset|
      sign_constraint : 'any' | 'below' | 'above' — caller imposes this when
                        question type demands (e.g., max question → 'below').
    """
    name = "near_tie"

    def apply(self, df, answer_cell, entity_value, orig_ans_val, params, seed):
        eps = float(params["eps_pct"])
        sign_constraint = str(params.get("sign_constraint", "any"))
        rng = random.Random(seed)

        ans_col = answer_cell["col"]
        if ans_col not in df.columns:
            raise ValueError(f"answer col '{ans_col}' not in df columns {list(df.columns)}")

        entity_col = df.columns[0]
        ans_mask = (df[entity_col].astype(str).str.strip().str.lower()
                    == str(entity_value).strip().lower())
        if ans_mask.sum() == 0:
            raise ValueError(f"answer entity '{entity_value}' not found")

        # Candidate non-answer rows with numeric ans_col value
        candidates = []
        for idx in df.index[~ans_mask]:
            v = _parse_num(df.at[idx, ans_col])
            if v is not None:
                candidates.append(idx)
        if not candidates:
            raise ValueError("no non-answer row has numeric value in answer column")

        # 1. Seed-driven row choice
        target_idx = rng.choice(candidates)

        # 2. Seed-driven sign within constraint
        if sign_constraint == "below":
            sign = -1
        elif sign_constraint == "above":
            sign = +1
        else:
            sign = rng.choice([-1, +1])

        # 3. Seed-driven offset magnitude (not always max ε — explore the band)
        offset_pct = rng.uniform(0.3 * eps, eps) * sign
        new_val = orig_ans_val * (1.0 + offset_pct / 100.0)
        new_val = round(new_val, 2)

        df_mod = df.copy()
        df_mod.at[target_idx, ans_col] = new_val
        return df_mod


class DistractorCountEditor(_Editor):
    """Append k synthetic distractor rows.

    params:
      k          : int — how many rows to add
      value_mode : 'uniform' | 'below_ans' | 'above_ans'
    """
    name = "distractor_count"

    LABEL_POOLS = [
        ["Other_A", "Other_B", "Other_C", "Other_D", "Other_E", "Other_F"],
        ["Region_1", "Region_2", "Region_3", "Region_4", "Region_5", "Region_6"],
        ["Group_I", "Group_II", "Group_III", "Group_IV", "Group_V", "Group_VI"],
    ]

    def apply(self, df, answer_cell, entity_value, orig_ans_val, params, seed):
        k = int(params["k"])
        mode = str(params.get("value_mode", "uniform"))
        rng = random.Random(seed)

        entity_col = df.columns[0]
        ans_col = answer_cell["col"]

        col_ranges = {}
        for col in df.columns[1:]:
            r = _col_numeric_range(df, col)
            if r is not None:
                col_ranges[col] = r
        if ans_col not in col_ranges:
            raise ValueError(f"answer col '{ans_col}' has no numeric values")

        ans_vmin, ans_vmax = col_ranges[ans_col]
        # Seed-driven label pool choice
        pool = rng.choice(self.LABEL_POOLS)
        existing_labels = set(df[entity_col].astype(str).str.strip().tolist())

        def pick_label(i):
            base = pool[i % len(pool)]
            j = 0
            name = base
            while name in existing_labels:
                j += 1
                name = f"{base}_{j}"
            existing_labels.add(name)
            return name

        new_rows = []
        for i in range(k):
            row = {entity_col: pick_label(i)}
            for col in df.columns[1:]:
                if col in col_ranges:
                    vmin, vmax = col_ranges[col]
                    if col == ans_col:
                        if mode == "below_ans":
                            hi = max(vmin, orig_ans_val * 0.97)
                            v = rng.uniform(vmin, hi) if hi > vmin else vmin
                        elif mode == "above_ans":
                            lo = min(vmax, orig_ans_val * 1.03)
                            v = rng.uniform(lo, vmax) if vmax > lo else vmax
                        else:
                            v = None
                            for _ in range(10):
                                cand = rng.uniform(vmin, vmax)
                                if abs(cand - orig_ans_val) / max(abs(orig_ans_val), 1e-6) > 0.05:
                                    v = cand
                                    break
                            if v is None:
                                v = rng.uniform(vmin, vmax)
                    else:
                        v = rng.uniform(vmin, vmax)
                    row[col] = round(float(v), 2)
                else:
                    row[col] = ""
            new_rows.append(row)

        new_df = pd.DataFrame(new_rows)
        df_mod = pd.concat([df, new_df], ignore_index=True)
        return df_mod


class OrdinalRankEditor(_Editor):
    """Append r-1 rows each > orig_ans_val on the answer column.

    Caller MUST only apply this to non-superlative questions.

    params:
      rank : int ≥ 2
    """
    name = "ordinal_rank"

    LABEL_POOLS = [
        ["Upper_A", "Upper_B", "Upper_C", "Upper_D", "Upper_E"],
        ["Leader_1", "Leader_2", "Leader_3", "Leader_4", "Leader_5"],
        ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
    ]

    def apply(self, df, answer_cell, entity_value, orig_ans_val, params, seed):
        r = int(params["rank"])
        if r < 2:
            raise ValueError(f"rank must be >= 2, got {r}")
        rng = random.Random(seed)

        entity_col = df.columns[0]
        ans_col = answer_cell["col"]

        col_ranges = {}
        for col in df.columns[1:]:
            rr = _col_numeric_range(df, col)
            if rr is not None:
                col_ranges[col] = rr
        if ans_col not in col_ranges:
            raise ValueError(f"answer col '{ans_col}' has no numeric values")

        pool = rng.choice(self.LABEL_POOLS)
        existing_labels = set(df[entity_col].astype(str).str.strip().tolist())

        def pick_label(i):
            base = pool[i % len(pool)]
            j = 0
            name = base
            while name in existing_labels:
                j += 1
                name = f"{base}_{j}"
            existing_labels.add(name)
            return name

        new_rows = []
        for i in range(r - 1):
            factor = 1.10 + i * 0.15 + rng.uniform(-0.05, 0.05)
            v_ans = round(orig_ans_val * factor, 2)
            row = {entity_col: pick_label(i)}
            for col in df.columns[1:]:
                if col == ans_col:
                    row[col] = v_ans
                elif col in col_ranges:
                    vmin, vmax = col_ranges[col]
                    row[col] = round(rng.uniform(vmin, vmax), 2)
                else:
                    row[col] = ""
            new_rows.append(row)

        new_df = pd.DataFrame(new_rows)
        df_mod = pd.concat([df, new_df], ignore_index=True)
        return df_mod


EDITORS = {
    NearTieEditor.name: NearTieEditor(),
    DistractorCountEditor.name: DistractorCountEditor(),
    OrdinalRankEditor.name: OrdinalRankEditor(),
}


# =============================================================================
# Scene generator subclass
# =============================================================================

class TypedTableEditor(TableEditor):
    """render_chart adds y_axis_min; generate_deterministic accepts edit_seed."""

    def render_chart(self, df, output_path, question="", seed=None, y_axis_min=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        if df.empty or df.shape[1] < 2:
            return False
        x_col = df.columns[0]
        y_cols = df.columns[1:]
        x_labels = df[x_col].astype(str).tolist()
        x_pos = np.arange(len(x_labels))
        series_list = []
        for col in y_cols:
            try:
                cleaned = df[col].astype(str).str.replace(
                    r'[$%,MBKmbk\s]', '', regex=True).str.strip()
                vals = pd.to_numeric(cleaned, errors="coerce")
                if vals.notna().sum() > 0:
                    series_list.append((str(col), vals.fillna(0).tolist()))
            except Exception:
                pass
        if not series_list:
            print(f"  [RenderR5] FAILED: no plottable numeric columns")
            return False
        n_series = len(series_list)

        q_lower = (question or "").lower()
        if any(kw in q_lower for kw in ["line", "curve", "trend", "intersect"]):
            chart_type = "line"
        elif any(kw in q_lower for kw in ["bar", "column"]):
            chart_type = "bar"
        else:
            chart_type = random.choice(["bar", "line"])

        font_scale = random.uniform(0.55, 0.85)
        grid_alpha = random.uniform(0.2, 0.6)
        rotate_labels = random.choice([0, 25, 35, 45])
        bg_color = random.choice(["#ffffff", "#f0f0f0", "#e8e8e8", "#f5f5dc"])
        colors = RANDOM_PALETTE[:]
        random.shuffle(colors)
        fig_w = random.uniform(7, 13)
        fig_h = random.uniform(4.5, 6.5)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_facecolor(bg_color)
        ax.grid(True, axis="both", alpha=grid_alpha, linewidth=0.5)

        if chart_type == "bar":
            bar_width = 0.7 / max(n_series, 1)
            for i, (col_name, vals) in enumerate(series_list):
                offset = (i - n_series / 2 + 0.5) * bar_width
                ax.bar(x_pos + offset, vals, bar_width * 0.9,
                       label=col_name, color=colors[i % len(colors)],
                       edgecolor="#333333", linewidth=0.4,
                       alpha=random.uniform(0.75, 0.95))
        else:
            markers = ["o", "s", "^", "D", "v", "p", "*"]
            line_styles = ["-", "--", "-.", ":"]
            for i, (col_name, vals) in enumerate(series_list):
                ax.plot(x_pos, vals, label=col_name,
                        color=colors[i % len(colors)],
                        marker=random.choice(markers),
                        markersize=random.uniform(3, 7),
                        linewidth=random.uniform(1.0, 2.5),
                        linestyle=random.choice(line_styles),
                        alpha=random.uniform(0.75, 0.95))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=rotate_labels,
                           ha="right" if rotate_labels > 0 else "center",
                           fontsize=8 * font_scale + 3)
        ax.tick_params(axis="y", labelsize=8 * font_scale + 3)
        ax.set_xlabel(x_col, fontsize=9 * font_scale + 3)
        if n_series > 1:
            ax.legend(fontsize=7 * font_scale + 3, loc="best", framealpha=0.6)

        if y_axis_min is not None:
            try:
                _, cur_max = ax.get_ylim()
                if float(y_axis_min) < cur_max:
                    ax.set_ylim(bottom=float(y_axis_min))
            except Exception as e:
                print(f"  [RenderR5] y_axis_min ignored: {e}")

        plt.tight_layout(pad=random.uniform(0.3, 1.5))

        try:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception as e:
            print(f"  [RenderR5] Render error: {e}")
            plt.close(fig)
            return False

    def generate_deterministic(self, csv_path, question, original_answer,
                               answer_cell, tier, edit_spec,
                               output_path, sample_id,
                               edit_seed=None):
        """
        edit_spec schema:
          {
            "type_name": "near_tie" | "distractor_count" | "ordinal_rank" | "y_compress",
            "params":   {... editor-specific ...},
            "render_config": {"y_axis_min": float} | None
          }
        edit_seed: int or None. If None, derived from (sample_id, type_name).
                    Passing an explicit seed lets callers generate multiple
                    independent edit variants at the same (sample, type, κ).
        """
        result = {
            "success": False, "new_answer": None,
            "original_answer": original_answer,
            "n_rows_before": 0, "n_rows_after": 0,
            "error": None,
        }
        df = self.load_csv(csv_path)
        if df is None:
            result["error"] = "cannot load CSV"
            return result
        result["n_rows_before"] = len(df)

        render_seed = abs(hash(f"render|{sample_id}")) % 100000
        edit_spec = edit_spec or {}
        type_name = edit_spec.get("type_name")
        params = edit_spec.get("params", {}) or {}
        render_cfg = edit_spec.get("render_config") or {}
        y_axis_min = render_cfg.get("y_axis_min")

        if tier == 2 or not type_name:
            ok = self.render_chart(df, output_path, question,
                                   seed=render_seed, y_axis_min=y_axis_min)
            if ok:
                result.update({"success": True, "new_answer": original_answer,
                               "n_rows_after": len(df)})
            else:
                result["error"] = "render failed"
            return result

        if type_name == "y_compress":
            ok = self.render_chart(df, output_path, question,
                                   seed=render_seed, y_axis_min=y_axis_min)
            if ok:
                result.update({"success": True, "new_answer": original_answer,
                               "n_rows_after": len(df)})
            else:
                result["error"] = "render failed"
            return result

        if type_name not in EDITORS:
            result["error"] = f"unknown edit type: {type_name}"
            return result
        if answer_cell is None:
            result["error"] = "answer_cell missing"
            return result

        entity_value = self.get_entity_value(df, answer_cell)
        if not entity_value:
            result["error"] = "cannot extract answer entity"
            return result

        orig_ans_val = _parse_num(original_answer)
        if orig_ans_val is None:
            result["error"] = f"original_answer '{original_answer}' non-numeric"
            return result

        if edit_seed is None:
            edit_seed = abs(hash(f"edit|{sample_id}|{type_name}")) % 100000
        try:
            df_mod = EDITORS[type_name].apply(
                df, answer_cell, entity_value, orig_ans_val, params, int(edit_seed))
        except Exception as e:
            result["error"] = f"editor {type_name} failed: {e}"
            return result
        result["n_rows_after"] = len(df_mod)

        new_answer = self.extract_answer(df_mod, answer_cell, entity_value=entity_value)
        if new_answer is None:
            result["error"] = f"entity '{entity_value}' lost after edit"
            return result

        new_ans_val = _parse_num(new_answer)
        if new_ans_val is not None:
            rel = abs(new_ans_val - orig_ans_val) / max(abs(orig_ans_val), 1e-6)
            if rel > 0.01:
                result["error"] = (f"editor '{type_name}' accidentally changed "
                                   f"answer: {orig_ans_val} → {new_ans_val}")
                return result
        result["new_answer"] = new_answer

        ok = self.render_chart(df_mod, output_path, question,
                               seed=render_seed, y_axis_min=y_axis_min)
        if not ok:
            result["error"] = "render failed"
            return result
        result["success"] = True
        return result


if __name__ == "__main__":
    gen = TypedTableEditor()
    print("Round-5 generator ready (v3 with seed-driven variants).")
    print("Available editors:", list(EDITORS.keys()))
