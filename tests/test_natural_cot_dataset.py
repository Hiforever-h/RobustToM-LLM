import json
import re
import unittest
from pathlib import Path

from grpo.build_natural_dataset import (
    build_natural_source_row,
    build_parquet_row,
)
from grpo.prompt import build_natural_cot_prompt, build_natural_problem_prompt
from rft.common import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        rendered = f"<user>{messages[0]['content']}</user><assistant>"
        return [ord(char) for char in rendered] if tokenize else rendered


class NaturalCoTDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = read_jsonl(
            ROOT / "data/counterfactual_process_reward_v3/train.jsonl"
        )[0]

    def test_prompt_removes_json_few_shots_event_numbers_and_newlines(self):
        actor_prompt = build_natural_cot_prompt(self.raw)
        judge_prompt = build_natural_problem_prompt(self.raw)
        self.assertNotIn("\n", actor_prompt)
        self.assertNotIn("\n", judge_prompt)
        self.assertNotIn("belief_trace", actor_prompt)
        self.assertNotIn("Schema:", actor_prompt)
        self.assertNotIn("Demonstration", actor_prompt)
        self.assertNotRegex(actor_prompt, re.compile(r"Story:\s*\d+\s"))
        self.assertIn("Think N:", actor_prompt)
        self.assertIn("State: <location>", actor_prompt)
        self.assertIn("Answer: <location>", actor_prompt)
        self.assertNotIn("Think N:", judge_prompt)

    def test_natural_source_deletes_legacy_response_and_prompt(self):
        row = build_natural_source_row(self.raw)
        self.assertNotIn("process_response", row)
        self.assertNotIn("prompt", row)
        self.assertEqual(row["process_prompt_version"], "natural-cot-think-state-v1")
        self.assertNotIn("\n", row["story"])
        self.assertEqual(row["process_target"], self.raw["process_target"])

    def test_parquet_row_preserves_explicit_judge_prompt_and_target(self):
        source = build_natural_source_row(self.raw)
        row = build_parquet_row(
            source,
            index=7,
            tokenizer=FakeTokenizer(),
            max_prompt_length=100000,
        )
        self.assertEqual(row["data_source"], "robust_tom_natural_cot_v4")
        self.assertEqual(row["judge_prompt"], source["judge_prompt"])
        self.assertEqual(row["reward_model"]["ground_truth"], source["process_target"])
        self.assertEqual(row["reward_model"]["style"], "natural_cot_judge")
        self.assertEqual(row["extra_info"]["index"], 7)

    def test_generated_source_manifest_and_rows_are_auditable(self):
        data_dir = ROOT / "data/counterfactual_process_reward_v4_natural"
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["contains_process_response"])
        self.assertEqual(manifest["splits"]["train"]["count"], 3200)
        for split, expected_count in (("train", 3200), ("val", 400), ("test", 600)):
            rows = read_jsonl(data_dir / f"{split}.jsonl")
            self.assertEqual(len(rows), expected_count)
            self.assertTrue(all("process_response" not in row for row in rows))
            self.assertTrue(all("\n" not in row["process_prompt"] for row in rows))


if __name__ == "__main__":
    unittest.main()
