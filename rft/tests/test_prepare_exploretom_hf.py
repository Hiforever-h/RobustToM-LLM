import csv
import json
import tempfile
import unittest
from pathlib import Path

from grpo.prompt import ORDER_TRACE_INSTRUCTION
from rft.prepare_exploretom_hf import prepare_exploretom_hf
from scripts.add_symbolic_v3_few_shots import FEW_SHOT_MARKER


class PrepareExploreToMHuggingFaceTest(unittest.TestCase):
    def test_builds_recommended_container_benchmark(self):
        fieldnames = [
            "story_structure",
            "infilled_story",
            "question",
            "expected_answer",
            "qprop=params",
            "qprop=nth_order",
            "qprop=non_unique_mental_state",
            "sprop=is_false_belief_story_1st",
            "sprop=is_false_belief_story_1st_and_2nd",
            "sprop=story_accuracy_1st_raw",
            "sprop=story_accuracy_1st_infilled",
            "sprop=global_idx",
            "param=story_type",
            "param=num_stories_total",
            "param=max_sentences",
            "param=num_people",
            "param=num_moves",
            "param=num_rooms",
        ]
        story = (
            "Alice entered the library. Bob entered the library. "
            "Alice moved the key to the red box, which is also located in the "
            "library. Bob left the library. Alice moved the key to the blue "
            "box, which is also located in the library."
        )
        base = {
            "story_structure": story,
            "infilled_story": story,
            "question": "Where does Bob think Alice thinks the key is?",
            "expected_answer": "red box",
            "qprop=params": "(['Bob', 'Alice'], 'key', 'container_location-False')",
            "qprop=nth_order": "2",
            "qprop=non_unique_mental_state": "FALSE",
            "sprop=is_false_belief_story_1st": "TRUE",
            "sprop=is_false_belief_story_1st_and_2nd": "TRUE",
            "sprop=story_accuracy_1st_raw": "0.5",
            "sprop=story_accuracy_1st_infilled": "0.5",
            "sprop=global_idx": "7",
            "param=story_type": "tomi",
            "param=num_stories_total": "10",
            "param=max_sentences": "15",
            "param=num_people": "2",
            "param=num_moves": "2",
            "param=num_rooms": "1",
        }
        rows = [base, dict(base), {**base, "qprop=non_unique_mental_state": "TRUE"}]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "sample.csv"
            with input_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            manifest = prepare_exploretom_hf(
                input_path, root, expected_source_count=3
            )
            test_rows = [
                json.loads(line)
                for line in (root / "test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(manifest["container_tom_question_count"], 3)
            self.assertEqual(manifest["eligible_before_dedup_count"], 2)
            self.assertEqual(manifest["removed_duplicate_count"], 1)
            self.assertEqual(manifest["test_count"], 1)
            self.assertEqual(manifest["test_order_counts"], {"2": 1})
            self.assertEqual(len(test_rows), 1)
            row = test_rows[0]
            self.assertEqual(row["belief_chain"], ["Bob", "Alice"])
            self.assertEqual(row["object"], "key")
            self.assertEqual(row["gold_answer"], "red box")
            self.assertEqual(row["choice_count"], 2)
            self.assertIn("red box", row["choices"])
            self.assertIn("blue box", row["choices"])
            self.assertTrue(row["story"].startswith("1 Alice entered"))
            self.assertIn("\n5 Alice moved", row["story"])
            self.assertNotIn("process_target", row)
            self.assertEqual(row["process_prompt"].count(FEW_SHOT_MARKER), 1)
            self.assertEqual(
                row["process_prompt"].count(ORDER_TRACE_INSTRUCTION), 1
            )


if __name__ == "__main__":
    unittest.main()
