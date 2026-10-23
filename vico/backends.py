"""Adapters for the included table editor and replaceable component factories."""
import importlib
from pathlib import Path


def load_factory(spec, config=None):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)(**(config or {}))


class TableBackend:
    """Edit CSV-backed figures; the core pipeline does not depend on this class."""
    def __init__(self):
        from table_editor import TableEditor
        self.generator = TableEditor()

    def edit(self, sample, plan, output_path, p_hat):
        if "table" not in sample:
            raise ValueError("The table backend requires sample.table (a CSV path)")
        tier = int(sample.get("tier", 2))
        if tier not in (1, 2):
            raise ValueError("Table tier must be 1 (data edit) or 2 (style only)")
        if tier == 1 and not sample.get("answer_cell"):
            raise ValueError("Data edits require answer_cell to recompute the label")
        result = self.generator.generate(
            csv_path=sample["table"], question=sample["question"],
            original_answer=sample["answer"], answer_cell=sample.get("answer_cell"),
            tier=tier, edit_instruction=plan.get("params", {}).get("edit_instruction", ""),
            output_path=str(output_path), p_hat=p_hat)
        if not result["success"]:
            raise ValueError(result["error"])
        edited = {**sample, "image": str(Path(output_path).resolve()),
                  "answer": str(result["new_answer"])}
        if result.get("table_path"):
            edited["table"] = result["table_path"]
        return edited


class SimplePlanner:
    def get_plan(self, p_hat_original, **context):
        direction = "harder" if p_hat_original > 0.5 else "easier"
        return {"tool": "edit", "params": {"edit_instruction":
                f"Make the visual question {direction}; keep the question answerable."},
                "reasoning": f"Current pass rate is {p_hat_original:.3f}."}
