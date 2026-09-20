import unittest
from collections import Counter
from pathlib import Path

from opsd.build_dataset import build_opsd_row
from rft.common import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


class OPSDDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(
            ROOT / "data/counterfactual_process_reward_v4_natural_compact/train.jsonl"
        )

    def test_original_training_split_has_expected_distribution(self):
        self.assertEqual(len(self.rows), 3200)
        self.assertEqual(
            Counter(row["question_order"] for row in self.rows),
            {1: 800, 2: 1504, 3: 896},
        )
        self.assertEqual(len({row["global_pair_id"] for row in self.rows}), 1600)

    def test_opsd_pair_preserves_student_and_augments_teacher(self):
        source = self.rows[0]
        row = build_opsd_row(source)
        self.assertEqual(row["problem"], source["process_prompt"])
        self.assertNotEqual(row["solution"], row["problem"])
        self.assertNotIn("Privileged reference for the teacher", row["problem"])
        self.assertIn("Privileged reference for the teacher", row["solution"])
        self.assertIn(f"verified final answer is {source['answer']}", row["solution"])
        self.assertEqual(
            len(row["privileged_reference"]["support_events"]),
            source["question_order"],
        )

    def test_all_rows_have_one_support_event_per_belief_level(self):
        for source in self.rows:
            with self.subTest(sample=source["global_sample_id"]):
                row = build_opsd_row(source)
                self.assertEqual(
                    len(row["privileged_reference"]["support_events"]),
                    source["question_order"],
                )


if __name__ == "__main__":
    unittest.main()
