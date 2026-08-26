import json
import tempfile
import unittest
from pathlib import Path

from grpo.prompt import ORDER_TRACE_INSTRUCTION
from rft.prepare_hitom_hf import prepare_hitom_hf
from scripts.add_symbolic_v3_few_shots import FEW_SHOT_MARKER


class PrepareHiToMHuggingFaceTest(unittest.TestCase):
    def test_converts_variants_and_reports_conflicting_labels(self):
        base = {
            "deception": False,
            "story_length": 1,
            "question_order": 4,
            "story": (
                "1 Alice, Bob, Carol, and David entered the room.\n"
                "2 The key is in the blue_box.\n"
            ),
            "question": (
                "Where does Alice think Bob thinks Carol thinks David thinks "
                "the key is?"
            ),
            "choices": "A. red_box, B. blue_box",
        }
        rows = [
            {
                **base,
                "prompting_type": "CoTP",
                "sample_id": 0,
                "answer": "blue_box",
            },
            {
                **base,
                "prompting_type": "VP",
                "sample_id": 1,
                "story": (
                    "Read the following story and answer the multiple-choice "
                    "question. Please provide answer without explanations.\n"
                    + base["story"]
                ),
                "answer": "red_box",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "Hi-ToM_data.json"
            input_path.write_text(json.dumps(rows), encoding="utf-8")

            manifest = prepare_hitom_hf(
                input_path, root, expected_source_count=2
            )
            test_rows = [
                json.loads(line)
                for line in (root / "test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            conflicts = [
                json.loads(line)
                for line in (root / "label_conflicts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(manifest["source_count"], 2)
            self.assertEqual(manifest["test_count"], 1)
            self.assertEqual(manifest["consistent_test_count"], 0)
            self.assertEqual(manifest["order4_test_count"], 1)
            self.assertEqual(manifest["order4_consistent_test_count"], 0)
            self.assertEqual(manifest["order4_source_label_conflict_count"], 1)
            self.assertEqual(manifest["source_label_conflict_count"], 1)
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(len(test_rows), 1)
            row = test_rows[0]
            self.assertEqual(row["source_prompting_type"], "CoTP")
            self.assertTrue(row["source_label_conflict"])
            self.assertEqual(row["answer"], "blue_box")
            self.assertEqual(row["gold_answer"], "blue_box")
            self.assertEqual(
                row["story"],
                "1 Alice, Bob, Carol, and David entered the room.\n"
                "2 The key is in the blue_box.",
            )
            self.assertNotIn("process_target", row)
            self.assertEqual(row["process_prompt"].count(FEW_SHOT_MARKER), 1)
            self.assertEqual(
                row["process_prompt"].count(ORDER_TRACE_INSTRUCTION), 1
            )


if __name__ == "__main__":
    unittest.main()
