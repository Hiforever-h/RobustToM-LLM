import csv
import json
import tempfile
import unittest
from pathlib import Path

from rft.prepare_tomrl_paper_eval import (
    PROMPT_VERSION,
    build_open_ended_process_prompt,
    infer_question_order,
    prepare_tomrl_paper_eval,
)


class PrepareToMRLPaperEvalTest(unittest.TestCase):
    def test_infers_controlled_question_orders(self):
        cases = {
            "Where is the apple really?": 0,
            "Where will Alice look for the apple?": 1,
            "Where does Alice think that Bob searches for the apple?": 2,
            "In which container will Alice search for the key?": 1,
            "In which container does Alice think that Bob will search for the key?": 2,
            "Where does Alice think Bob thinks Carol thinks the key is?": 3,
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(infer_question_order(question), expected)

    def test_prompt_is_open_ended_natural_cot(self):
        prompt = build_open_ended_process_prompt(
            "1 Alice saw the key in the blue_box.\n2 Bob entered the room.",
            "Where does Alice think the key is?",
        )
        self.assertTrue(prompt.startswith("Read the story and answer the question."))
        self.assertIn("Story: Alice saw the key", prompt)
        self.assertIn("Question: Where does Alice think", prompt)
        self.assertEqual(prompt.count("Output requirements:"), 1)
        self.assertNotIn("Choices:", prompt)
        self.assertNotIn("<|im_start|>", prompt)
        self.assertNotIn("<think>", prompt)

    def test_filters_order_zero_and_writes_source_splits(self):
        rows = [
            {
                "data_source": "tomi",
                "story": "The apple is in the box.",
                "question": "Where is the apple really?",
                "prompt": "legacy",
                "answer": "box",
            },
            {
                "data_source": "tomi",
                "story": "The apple is in the box.",
                "question": "Where will Alice look for the apple?",
                "prompt": "legacy",
                "answer": "box",
            },
            {
                "data_source": "explore_tom_structured",
                "story": "Alice moved the key to the drawer.",
                "question": (
                    "In which container does Bob think that Alice will search "
                    "for the key?"
                ),
                "prompt": "legacy",
                "answer": "drawer",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            output = root / "output"
            manifest = prepare_tomrl_paper_eval(
                source, output, strict_expected_counts=False
            )
            test_rows = [
                json.loads(line)
                for line in (output / "test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(manifest["source_count"], 3)
            self.assertEqual(manifest["excluded_order0_count"], 1)
            self.assertEqual(manifest["test_count"], 2)
            self.assertEqual([row["question_order"] for row in test_rows], [1, 2])
            self.assertTrue(all(row["split"] == "test" for row in test_rows))
            self.assertTrue(
                all(row["process_prompt_version"] == PROMPT_VERSION for row in test_rows)
            )
            self.assertTrue(all("process_target" not in row for row in test_rows))
            self.assertEqual(
                len((output / "tomi.jsonl").read_text().splitlines()), 1
            )
            self.assertEqual(
                len(
                    (output / "explore_tom_structured.jsonl")
                    .read_text()
                    .splitlines()
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
