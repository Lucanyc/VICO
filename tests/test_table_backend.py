from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

from vico.backends import SimplePlanner, TableBackend
from vico.pipeline import evolve
from build_rewriter_data import build_samples


class TableBackendTests(unittest.TestCase):
    def test_real_rendering_and_rewriter_conversion_without_api(self):
        class Solver:
            scores = iter([1.0, 0.5])
            def evaluate(self, *args):
                return next(self.scores)
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            root = Path(tmp)
            table = root / "table.csv"
            table.write_text("Category,Value\nA,10\nB,20\nC,30\n")
            Image.new("RGB", (64, 64), "white").save(root / "seed.png")
            sample = dict(id="synthetic", image=str(root / "seed.png"),
                          table=str(table), question="Value for B?", answer="20", tier=2)
            result = evolve(sample, SimplePlanner(), TableBackend(), Solver(), root / "edits")
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["final_answer"], "20")
            with Image.open(result["final_image_path"]) as image:
                image.verify()
            samples, _ = build_samples([result], beta=4, max_repeats=5, png_dir="")
            self.assertEqual(len(samples), 5)
            self.assertEqual(samples[0]["images"], [sample["image"]])
            self.assertIn("Value for B?", samples[0]["conversations"][1]["value"])

    def test_modified_csv_is_carried_to_next_turn(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            root = Path(tmp)
            table = root / "table.csv"
            table.write_text("Category,Value\nA,10\nB,20\n")
            sample = dict(id="a", image="seed.png", table=str(table), question="Value for B?",
                          answer="20", tier=1, answer_cell={"row": 1, "col": "Value"})
            editor = TableBackend()
            seen_values = []
            def modify(df, *args):
                seen_values.append(int(df.iloc[1, 1]))
                df = df.copy()
                df.iloc[1, 1] += 1
                return df
            # Only the external model response is substituted; rendering, answer
            # extraction, CSV persistence, and the second edit execute normally.
            with patch.object(editor.generator, "_modify_csv", side_effect=modify):
                edited = editor.edit(sample, {"params": {}}, root / "one.png", 1)
                final = editor.edit(edited, {"params": {}}, root / "two.png", 0.8)
            self.assertEqual(seen_values, [20, 21])
            self.assertEqual(float(final["answer"]), 22)
            self.assertEqual(int(pd.read_csv(final["table"]).iloc[1, 1]), 22)


if __name__ == "__main__":
    unittest.main()
