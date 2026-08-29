import json
import re
import unittest
from copy import deepcopy
from pathlib import Path

from grpo.build_natural_dataset import (
    build_natural_source_row,
    build_parquet_row,
)
from grpo.prompt import build_natural_cot_prompt, build_natural_problem_prompt
from rft.common import read_jsonl
from rft.prompt import (
    COMPACT_NATURAL_COT_PROMPT_VERSION,
    NATURAL_COT_PROMPT_VERSION,
    build_compact_process_prompt,
)

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
        rows = read_jsonl(ROOT / "data/counterfactual_process_reward_v3/train.jsonl")
        cls.raw = rows[0]
        cls.raw_by_order = {
            order: next(row for row in rows if row["question_order"] == order)
            for order in (1, 2, 3)
        }

    def test_prompt_uses_dynamic_exact_order_template(self):
        actor_prompt = build_natural_cot_prompt(self.raw)
        judge_prompt = build_natural_problem_prompt(self.raw)
        self.assertIn("\n", actor_prompt)
        self.assertNotIn("\n", judge_prompt)
        self.assertNotIn("belief_trace", actor_prompt)
        self.assertNotIn("Schema:", actor_prompt)
        self.assertNotIn("Demonstration", actor_prompt)
        self.assertNotIn("Example response format", actor_prompt)
        self.assertNotIn("Sophia", actor_prompt)
        self.assertNotRegex(actor_prompt, re.compile(r"Story:\s*\d+\s"))
        order = self.raw["process_target"]["tom_order"]
        self.assertIn(f"Output exactly {order} Think/State blocks", actor_prompt)
        self.assertIn(f"Answer must repeat the State from Think {order}", actor_prompt)
        self.assertIn("State: <location>", actor_prompt)
        self.assertNotIn("Reasoning rules:", judge_prompt)

    def test_prompt_maps_every_order_from_inner_to_outer(self):
        for order, source in self.raw_by_order.items():
            with self.subTest(order=order):
                prompt = build_natural_cot_prompt(source)
                chain = source["process_target"]["belief_chain"]
                self.assertIn(" -> ".join(chain), prompt)
                self.assertIn(f"Output exactly {order} Think/State", prompt)
                for index in range(1, order + 1):
                    self.assertEqual(prompt.count(f"Think {index}:"), 1)
                self.assertNotIn(f"Think {order + 1}:", prompt)
                self.assertIn(
                    f"Answer: <same location as State from Think {order}>", prompt
                )

    def test_prompt_does_not_use_gold_locations(self):
        source = deepcopy(self.raw)
        sentinel = "secret_gold_location_never_in_problem"
        source["answer"] = sentinel
        source["process_target"]["answer"] = sentinel
        for step in source["process_target"]["belief_trace"]:
            step["location"] = sentinel
        self.assertNotIn(sentinel, build_natural_cot_prompt(source))

    def test_natural_source_deletes_legacy_response_and_prompt(self):
        row = build_natural_source_row(self.raw)
        self.assertNotIn("process_response", row)
        self.assertNotIn("prompt", row)
        self.assertEqual(row["process_prompt_version"], NATURAL_COT_PROMPT_VERSION)
        self.assertNotIn("\n", row["story"])
        self.assertIn("\n", row["process_prompt"])
        self.assertNotIn("\n", row["judge_prompt"])
        self.assertEqual(row["process_target"], self.raw["process_target"])

    def test_compact_source_requires_model_to_infer_order(self):
        row = build_natural_source_row(self.raw, compact_prompt=True)
        prompt = row["process_prompt"]
        instruction = prompt.rsplit("\n\n", 1)[-1]
        self.assertEqual(
            row["process_prompt_version"], COMPACT_NATURAL_COT_PROMPT_VERSION
        )
        self.assertEqual(prompt, build_compact_process_prompt(row))
        self.assertIn("Infer N from the question itself", instruction)
        self.assertNotIn("Reasoning rules:", prompt)
        self.assertNotIn("Required output format:", prompt)
        self.assertNotRegex(instruction, re.compile(r"\bN\s*=\s*\d+\b"))
        self.assertNotIn("process_response", row)

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

    def test_parquet_row_accepts_compact_prompt_version(self):
        source = build_natural_source_row(self.raw, compact_prompt=True)
        row = build_parquet_row(
            source,
            index=8,
            tokenizer=FakeTokenizer(),
            max_prompt_length=100000,
            expected_prompt_version=COMPACT_NATURAL_COT_PROMPT_VERSION,
        )
        self.assertEqual(row["data_source"], "robust_tom_natural_cot_v4_compact")
        self.assertEqual(
            row["process_prompt_version"], COMPACT_NATURAL_COT_PROMPT_VERSION
        )
        self.assertEqual(row["prompt"][0]["content"], source["process_prompt"])
        self.assertEqual(row["judge_prompt"], source["judge_prompt"])
        self.assertEqual(row["reward_model"]["ground_truth"], source["process_target"])

    def test_generated_source_manifest_and_rows_are_auditable(self):
        data_dir = ROOT / "data/counterfactual_process_reward_v4_natural"
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["contains_process_response"])
        self.assertEqual(manifest["prompt_version"], NATURAL_COT_PROMPT_VERSION)
        self.assertTrue(manifest["multiline_actor_prompts"])
        self.assertTrue(manifest["single_line_judge_prompts"])
        self.assertEqual(manifest["splits"]["train"]["count"], 3200)
        for split, expected_count in (("train", 3200), ("val", 400), ("test", 600)):
            rows = read_jsonl(data_dir / f"{split}.jsonl")
            self.assertEqual(len(rows), expected_count)
            self.assertTrue(all("process_response" not in row for row in rows))
            self.assertTrue(all("\n" in row["process_prompt"] for row in rows))
            self.assertTrue(all("\n" not in row["judge_prompt"] for row in rows))
            self.assertTrue(
                all(
                    row["process_prompt_version"] == NATURAL_COT_PROMPT_VERSION
                    for row in rows
                )
            )
            for row in rows:
                prompt = row["process_prompt"]
                order = row["process_target"]["tom_order"]
                self.assertEqual(prompt, build_natural_cot_prompt(row))
                self.assertNotIn("Example response format", prompt)
                for index in range(1, order + 1):
                    self.assertEqual(prompt.count(f"Think {index}:"), 1)
                self.assertNotIn(f"Think {order + 1}:", prompt)

    def test_generated_compact_source_has_all_three_splits(self):
        data_dir = ROOT / "data/counterfactual_process_reward_v4_natural_compact"
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["compact_prompt"])
        self.assertFalse(manifest["numeric_tom_order_exposed_to_actor"])
        self.assertEqual(manifest["prompt_version"], COMPACT_NATURAL_COT_PROMPT_VERSION)
        for split, expected_count in (("train", 3200), ("val", 400), ("test", 600)):
            rows = read_jsonl(data_dir / f"{split}.jsonl")
            self.assertEqual(len(rows), expected_count)
            for row in rows:
                self.assertNotIn("process_response", row)
                self.assertEqual(
                    row["process_prompt_version"],
                    COMPACT_NATURAL_COT_PROMPT_VERSION,
                )
                self.assertEqual(
                    row["process_prompt"], build_compact_process_prompt(row)
                )
                self.assertNotIn("Reasoning rules:", row["process_prompt"])


if __name__ == "__main__":
    unittest.main()
