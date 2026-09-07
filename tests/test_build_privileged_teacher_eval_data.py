import unittest
from pathlib import Path

from rft.common import read_jsonl
from scripts.build_privileged_teacher_eval_data import (
    PRIVILEGED_PROMPT_VERSION,
    build_privileged_teacher_row,
    derive_support_events,
)


ROOT = Path(__file__).resolve().parents[1]


class PrivilegedTeacherEvalDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(
            ROOT / "data/counterfactual_process_reward_v4_natural_compact/test.jsonl"
        )

    def test_every_order4_step_has_one_distinct_support_event(self):
        self.assertEqual(len(self.rows), 600)
        for source in self.rows:
            with self.subTest(sample=source["global_sample_id"]):
                events = derive_support_events(source)
                self.assertEqual(source["question_order"], 4)
                self.assertEqual(len(events), 4)
                self.assertEqual(len({event["event_id"] for event in events}), 4)

    def test_teacher_prompt_exposes_answer_and_natural_event_evidence(self):
        source = self.rows[0]
        row = build_privileged_teacher_row(source)
        context = row["privileged_context"]
        self.assertEqual(row["process_prompt_version"], PRIVILEGED_PROMPT_VERSION)
        self.assertEqual(row["student_process_prompt"], source["process_prompt"])
        self.assertIn(f"verified final answer is {source['answer']}", context)
        self.assertEqual(context.count("\n- "), 4)
        self.assertIn("jointly observed", context)
        self.assertNotIn("event_id", context)
        self.assertNotIn("answer_event_id", context)
        self.assertNotIn("critical_event", context)
        self.assertLess(
            row["process_prompt"].index("Privileged reference for the teacher"),
            row["process_prompt"].index("Output requirements:"),
        )

    def test_critical_event_is_only_support_on_observed_side(self):
        counts = {"observed": 0, "hidden": 0}
        for source in self.rows:
            support_ids = {
                event["event_id"] for event in derive_support_events(source)
            }
            counts[source["intervention_type"]] += (
                source["critical_event_id"] in support_ids
            )
        self.assertEqual(counts, {"observed": 300, "hidden": 0})

    def test_validation_orders_have_matching_support_event_counts(self):
        rows = read_jsonl(
            ROOT / "data/counterfactual_process_reward_v4_natural_compact/val.jsonl"
        )
        self.assertEqual(len(rows), 400)
        for source in rows:
            with self.subTest(sample=source["global_sample_id"]):
                row = build_privileged_teacher_row(source)
                order = source["question_order"]
                self.assertIn(order, {1, 2, 3})
                self.assertEqual(
                    len(row["privileged_reference"]["support_events"]), order
                )
                self.assertEqual(
                    row["privileged_context"].count("\n- "), order
                )


if __name__ == "__main__":
    unittest.main()
