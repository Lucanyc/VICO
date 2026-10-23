import json
import math
from pathlib import Path
import tempfile
import unittest

from vico.data import load_samples, resolve_image
from vico.metrics import extract_answer, matches
from vico.pipeline import evolve


class DataTests(unittest.TestCase):
    def test_field_mapping_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            row = {"key": 7, "picture": "a.png", "prompt": "How many?", "target": 0}
            path.write_text(json.dumps(row) + "\n")
            mapping = dict(id="key", image="picture", question="prompt", answer="target")
            sample = load_samples(path, mapping)[0]
            self.assertEqual((sample["id"], sample["answer"]), ("7", "0"))
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_samples(path, mapping)

    def test_path_resolution_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "images"
            root.mkdir()
            (Path(tmp) / "outside.png").touch()
            with self.assertRaises(ValueError):
                resolve_image({"image": "../outside.png"}, root)


class MetricTests(unittest.TestCase):
    def test_last_balanced_box_and_tags(self):
        self.assertEqual(extract_answer(r"\boxed{2} then \boxed{\frac{1}{3}}"), r"\frac{1}{3}")
        self.assertIsNone(extract_answer(r"\boxed{unfinished"))
        self.assertEqual(extract_answer("<answer>2</answer><answer>3</answer>", "answer_tag"), "3")

    def test_numeric_tolerance_is_explicit(self):
        self.assertTrue(matches("104", "100", "relaxed", 0.05))
        self.assertFalse(matches("104", "100", "exact"))
        self.assertFalse(matches("10", "10000"))
        self.assertFalse(matches("Italy in 2020", "Italy"))
        self.assertFalse(matches("inf", "1"))
        self.assertFalse(matches(None, "0"))


class PipelineTests(unittest.TestCase):
    def test_failed_edit_and_cumulative_state_preserve_full_question(self):
        class Planner:
            seen = []
            def get_plan(self, **context):
                self.seen.append(context)
                return {"tool": "edit", "params": {"edit_instruction": "change"}}
        class Editor:
            calls = 0
            def edit(self, sample, plan, output, p):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("invalid edit")
                output.touch()
                return {**sample, "image": str(output), "answer": str(self.calls)}
        class Solver:
            values = iter([1.0, 0.9, 0.5])
            def evaluate(self, *args):
                return next(self.values)
        with tempfile.TemporaryDirectory() as tmp:
            planner = Planner()
            sample = dict(id="../unsafe", image="seed.png", question="Q" * 240, answer="1")
            result = evolve(sample, planner, Editor(), Solver(), tmp, max_turns=3)
            self.assertEqual(result["status"], "success")
            self.assertEqual(len(result["question"]), 240)
            self.assertFalse(result["turns"][0]["exec_success"])
            self.assertEqual(planner.seen[-1]["answer"], "2")
            self.assertEqual(result["final_answer"], "3")
            self.assertEqual(Path(result["final_image_path"]).parent, Path(tmp))

    def test_invalid_solver_output_is_not_counted_as_failure(self):
        class Solver:
            def evaluate(self, *args):
                return math.nan
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "pass rate"):
                evolve(dict(id="a", image="x", question="q", answer="a"), None, None, Solver(), tmp)


if __name__ == "__main__":
    unittest.main()
