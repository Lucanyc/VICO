#!/usr/bin/env python3
"""
VICO Scene Generator — GenEnv-style CSV modification + matplotlib rendering.

Flow:
  1. Load original CSV table
  2. GPT-5.2 modifies CSV (outputs modified table as JSON)
  3. Programmatically extract new answer from modified CSV
  4. Render new chart image via matplotlib
  5. Return (new_image, question, new_answer)

Answer correctness:
  - Tier 1 (82%): answer = value at answer_cell position in modified CSV
    → 100% clean, no computation needed
  - Tier 2 (18%): answer unchanged, only rendering modified

Usage:
    generator = TableEditor()
    result = generator.generate(
        csv_path="tables/00006834003065.csv",
        question="What is Nigeria's GDP?",
        original_answer="43.54",
        answer_cell={"row": 0, "col": "GDP"},
        tier=1,
        edit_instruction="Add distractor rows with similar values",
        output_path="modified_chart.png",
        p_hat=0.9,
    )
"""

import os, sys, json, re, copy, random, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Colors for rendering ──
RANDOM_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#1a5276", "#cc0000", "#008080", "#800080", "#808000",
    "#000080", "#800000", "#d2b48c", "#c0c0c0", "#ff00ff",
]


class TableEditor:
    """
    Modify CSV data + render chart + compute new answer.
    """

    SYSTEM_PROMPT = """You are a chart data modifier. You will receive a CSV table as JSON and a question about the chart.

Your job: modify the CSV data to make the chart harder for a Solver to answer correctly.
Follow the Rewriter's instruction. Output a COMPLETE modified table as JSON.

Effective modifications:
- Change numeric values to create near-miss values (similar numbers that confuse counting/comparison)
- Add new rows with similar values to increase chart density
- Change the answer value to a different number

CRITICAL RULES:
- NEVER rename, remove, or reorder values in the FIRST column (entity/label column)
- NEVER rename columns
- You can ONLY change values in non-first columns, and ADD new rows at the END
- Keep the same column structure (same number and names of columns)
- Output valid JSON only, no markdown, no explanation

Output format:
{"columns": ["col1", "col2", ...], "data": [["row1_val1", "row1_val2"], ["row2_val1", "row2_val2"], ...]}"""

    def __init__(self):
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            if not api_key:
                print("[TableEditor] No OPENAI_API_KEY")
                return
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
            print("[TableEditor] GPT-5.2 client initialized")
        except Exception as e:
            print(f"[TableEditor] Client init failed: {e}")

    def load_csv(self, csv_path: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [TableEditor] CSV read error: {e}")
            return None

    def extract_answer(self, df: pd.DataFrame, answer_cell: Dict,
                       entity_value: str = None) -> Optional[str]:
        """
        Extract answer from DataFrame using entity matching (not fixed row index).

        Strategy:
          1. If entity_value provided: find row where entity column matches → read answer column
          2. Fallback to fixed row index if entity matching fails
          3. Return None if both fail

        This is 100% clean because: we locate the answer by the ENTITY name
        (e.g., "2019", "Nigeria"), not by row position. Even if GPT-5.2
        reordered or inserted rows, we find the right one.
        """
        col = answer_cell.get("col")
        if col not in df.columns:
            return None

        # Strategy 1: entity matching (robust to row reordering)
        if entity_value:
            entity_col = df.columns[0]
            matches = df[df[entity_col].astype(str).str.strip().str.lower() == str(entity_value).strip().lower()]
            if len(matches) > 0:
                return str(matches.iloc[0][col]).strip()

        # Strategy 2: fixed row index (fallback)
        row = answer_cell.get("row", 0)
        if row < len(df):
            return str(df.iloc[row][col]).strip()

        return None

    def get_entity_value(self, df: pd.DataFrame, answer_cell: Dict) -> Optional[str]:
        """Get the entity value (first column) at the answer row. Used for entity matching."""
        try:
            row = answer_cell.get("row", 0)
            entity_col = df.columns[0]
            if row < len(df):
                return str(df.iloc[row][entity_col]).strip()
        except:
            pass
        return None

    def _df_to_json(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to JSON string for GPT-5.2 prompt."""
        return json.dumps({
            "columns": list(df.columns),
            "data": df.values.tolist(),
        }, ensure_ascii=False, default=str)

    def _json_to_df(self, json_str: str) -> Optional[pd.DataFrame]:
        """Parse GPT-5.2 output JSON back to DataFrame."""
        try:
            raw = json_str.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            obj = json.loads(raw)
            columns = obj.get("columns", [])
            data = obj.get("data", [])
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            print(f"  [TableEditor] JSON parse error: {e}")
            return None

    def _modify_csv(self, df: pd.DataFrame, question: str,
                    edit_instruction: str, answer_cell: Optional[Dict],
                    tier: int, p_hat: float) -> Optional[pd.DataFrame]:
        """Use GPT-5.2 to modify the CSV data."""
        if not self._client:
            return None

        csv_json = self._df_to_json(df)

        # Build context about what's protected
        if tier == 1 and answer_cell:
            cell_info = (f"The answer to the question is at row {answer_cell['row']}, "
                        f"column '{answer_cell['col']}'. You can freely change this value "
                        f"(the new answer will be whatever you put there). "
                        f"Also change OTHER values to create confusion.")
        else:
            cell_info = ("Do NOT change any data values. Only rearrange rows or "
                        "change non-data text if needed. The answer must stay the same.")

        user_prompt = (
            f"CSV Table:\n{csv_json}\n\n"
            f"Question: {question}\n"
            f"Current Solver pass rate: {p_hat:.3f}\n\n"
            f"{cell_info}\n\n"
            f"Rewriter instruction: {edit_instruction[:300]}\n\n"
            f"Output the COMPLETE modified table as JSON."
        )

        try:
            response = self._client.chat.completions.create(
                model=os.environ.get("AMPLIFIER_MODEL", "gpt-5.2"),
                max_completion_tokens=3000,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()
            modified_df = self._json_to_df(raw)
            if modified_df is None:
                return None

            # Clean numeric columns: strip $, %, commas, "M", "B" etc.
            for col in modified_df.columns[1:]:
                modified_df[col] = modified_df[col].astype(str).str.replace(r'[$%,MBKmkb]', '', regex=True).str.strip()

            # Validate: at least one numeric column (otherwise can't render)
            has_numeric = False
            for col in modified_df.columns[1:]:
                numeric_vals = pd.to_numeric(modified_df[col], errors="coerce")
                if numeric_vals.notna().sum() > 0:
                    has_numeric = True
                    break
            if not has_numeric:
                print(f"  [TableEditor] Modified CSV has no numeric columns after cleaning, rejecting")
                for col in modified_df.columns[1:]:
                    print(f"    {col}: {modified_df[col].head(3).tolist()}")
                return None

            # Validate: not empty
            if len(modified_df) < 1:
                print(f"  [TableEditor] Modified CSV is empty, rejecting")
                return None

            print(f"  [TableEditor] GPT-5.2 output: {len(modified_df)} rows, {len(modified_df.columns)} cols")
            return modified_df
        except Exception as e:
            print(f"  [TableEditor] GPT-5.2 error: {e}")
            return None

    def render_chart(self, df: pd.DataFrame, output_path: str,
                     question: str = "", seed: int = None) -> bool:
        """Render chart from DataFrame using matplotlib."""
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
                # Clean numeric values: strip $, %, commas, M/B/K, spaces
                cleaned = df[col].astype(str).str.replace(r'[$%,MBKmbk\s]', '', regex=True).str.strip()
                vals = pd.to_numeric(cleaned, errors="coerce")
                if vals.notna().sum() > 0:
                    series_list.append((str(col), vals.fillna(0).tolist()))
            except:
                pass

        if not series_list:
            print(f"  [Render] FAILED: no plottable numeric columns in modified CSV")
            print(f"  [Render] Columns: {list(df.columns)}")
            for col in y_cols:
                sample = df[col].head(3).tolist()
                print(f"  [Render]   {col}: {sample} (types: {df[col].dtype})")
            return False

        n_series = len(series_list)

        # Detect chart type from question
        q_lower = question.lower()
        if any(kw in q_lower for kw in ["line", "curve", "trend", "intersect"]):
            chart_type = "line"
        elif any(kw in q_lower for kw in ["bar", "column"]):
            chart_type = "bar"
        else:
            chart_type = random.choice(["bar", "line"])

        # Random visual config
        font_scale = random.uniform(0.55, 0.85)
        grid_alpha = random.uniform(0.2, 0.6)
        rotate_labels = random.choice([0, 25, 35, 45])
        bg_color = random.choice(["#ffffff", "#f0f0f0", "#e8e8e8", "#f5f5dc"])

        # Assign colors
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

        plt.tight_layout(pad=random.uniform(0.3, 1.5))

        try:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception as e:
            print(f"  [TableEditor] Render error: {e}")
            plt.close(fig)
            return False

    def generate(self, csv_path: str, question: str, original_answer: str,
                 answer_cell: Optional[Dict], tier: int,
                 edit_instruction: str, output_path: str,
                 p_hat: float = 1.0) -> Dict:
        """
        Full pipeline: load CSV → modify → extract answer → render.

        Returns:
            dict with: success, new_answer, original_answer, error
        """
        result = {
            "success": False,
            "new_answer": None,
            "original_answer": original_answer,
            "n_rows_before": 0,
            "n_rows_after": 0,
            "error": None,
        }

        # Load original CSV
        df = self.load_csv(csv_path)
        if df is None:
            result["error"] = "Cannot load CSV"
            return result

        result["n_rows_before"] = len(df)

        if tier == 2:
            # Tier 2: only change rendering, not data
            # Just render with random style, answer stays same
            success = self.render_chart(df, output_path, question,
                                        seed=random.randint(0, 99999))
            if success:
                result["success"] = True
                result["new_answer"] = original_answer
                result["n_rows_after"] = len(df)
                result["table_path"] = os.path.abspath(csv_path)
            else:
                result["error"] = "Render failed"
            return result

        # Get entity value BEFORE modification (for reliable answer lookup)
        entity_value = None
        if answer_cell:
            entity_value = self.get_entity_value(df, answer_cell)
            print(f"  [TableEditor] Entity at answer row: '{entity_value}'")

        # Tier 1: modify CSV data via GPT-5.2
        modified_df = self._modify_csv(df, question, edit_instruction,
                                        answer_cell, tier, p_hat)
        if modified_df is None:
            result["error"] = "CSV modification failed"
            return result

        result["n_rows_after"] = len(modified_df)

        # Extract new answer from modified CSV using ENTITY MATCHING
        if answer_cell:
            new_answer = self.extract_answer(modified_df, answer_cell,
                                              entity_value=entity_value)
            if new_answer is None:
                # Entity was deleted or renamed — reject this modification
                result["error"] = f"Entity '{entity_value}' not found in modified CSV"
                print(f"  [TableEditor] REJECTED: entity '{entity_value}' lost after modification")
                return result

            # Verify: entity still matches
            if entity_value:
                entity_col = modified_df.columns[0]
                matches = modified_df[modified_df[entity_col].astype(str).str.strip().str.lower()
                                      == str(entity_value).strip().lower()]
                if len(matches) == 0:
                    result["error"] = f"Entity '{entity_value}' deleted by GPT-5.2"
                    return result
                if len(matches) > 1:
                    print(f"  [TableEditor] WARNING: entity '{entity_value}' appears {len(matches)} times, using first")
        else:
            new_answer = original_answer

        result["new_answer"] = new_answer
        print(f"  [TableEditor] Answer: {original_answer} → {new_answer} "
              f"(entity='{entity_value}'), rows: {len(df)} → {len(modified_df)}")

        # Render
        success = self.render_chart(modified_df, output_path, question,
                                     seed=random.randint(0, 99999))
        if not success:
            result["error"] = "Render failed"
            return result

        result["success"] = True
        # Keep the edited table with the image so the next turn uses the same
        # state as the image and answer seen by the Planner and Solver.
        table_path = os.path.splitext(output_path)[0] + ".csv"
        modified_df.to_csv(table_path, index=False)
        result["table_path"] = os.path.abspath(table_path)
        return result
