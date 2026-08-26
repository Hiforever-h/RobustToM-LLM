import json
import unittest

from rft.evaluate import evaluate_answer_predictions, evaluate_predictions


class EvaluateTest(unittest.TestCase):
    def test_answer_only_ignores_trace_and_normalizes_location(self):
        data = [
            {"global_sample_id": "sample-1", "gold_answer": "blue_pantry"},
            {"global_sample_id": "sample-2", "gold_answer": "red_crate"},
        ]
        predictions = [
            {
                "global_sample_id": "sample-1",
                "response": json.dumps(
                    {
                        "tom_order": 99,
                        "belief_trace": [],
                        "answer": "Blue Pantry",
                    }
                ),
            },
            {
                "global_sample_id": "sample-2",
                "response": json.dumps({"answer": "green_box"}),
            },
        ]
        overall = evaluate_answer_predictions(predictions, data)["overall"]
        self.assertEqual(overall["count"], 2)
        self.assertEqual(overall["correct_count"], 1)
        self.assertEqual(overall["answer_accuracy"], 0.5)

    def test_answer_only_counts_malformed_or_missing_answer_as_wrong(self):
        data = [
            {"global_sample_id": "sample-1", "gold_answer": "blue_pantry"},
            {"global_sample_id": "sample-2", "gold_answer": "red_crate"},
        ]
        predictions = [
            {"global_sample_id": "sample-1", "response": "not json"},
            {
                "global_sample_id": "sample-2",
                "response": json.dumps({"value": "red_crate"}),
            },
        ]
        overall = evaluate_answer_predictions(predictions, data)["overall"]
        self.assertEqual(overall["correct_count"], 0)
        self.assertEqual(overall["answer_accuracy"], 0.0)

    def test_answer_only_rejects_id_mismatch_and_duplicates(self):
        data = [{"global_sample_id": "sample-1", "gold_answer": "blue_pantry"}]
        with self.assertRaisesRegex(ValueError, "sample IDs differ"):
            evaluate_answer_predictions(
                [{"global_sample_id": "sample-2", "response": "{}"}], data
            )
        with self.assertRaisesRegex(ValueError, "Duplicate global_sample_id"):
            evaluate_answer_predictions(
                [
                    {"global_sample_id": "sample-1", "response": "{}"},
                    {"global_sample_id": "sample-1", "response": "{}"},
                ],
                data,
            )

    def test_perfect_counterfactual_pair(self):
        rows = []
        for side, observed, answer in (
            ("observed", True, "linen chest"),
            ("hidden", False, "archive drawer"),
        ):
            target = {
                "tom_order": 1,
                "belief_chain": ["Alice"],
                "object": "passport",
                "reasoning_mode": "belief",
                "final_move_observed": observed,
                "nested_belief": answer,
                "answer": answer,
            }
            rows.append(
                {
                    "global_sample_id": f"sample-{side}",
                    "global_pair_id": "pair-1",
                    "source_dataset": "hi-tom",
                    "question_order": 1,
                    "intervention_type": side,
                    "process_target": target,
                    "response": json.dumps(target),
                    "generation_reached_eos": True,
                    "token_count": 20,
                }
            )
        overall = evaluate_predictions(rows)["overall"]
        for key in (
            "parse_rate",
            "strict_format_rate",
            "full_reward_rate",
            "answer_accuracy",
            "core_state_accuracy",
            "pair_accuracy",
            "intervention_sensitivity",
            "answer_state_consistency",
            "eos_rate",
        ):
            self.assertEqual(overall[key], 1.0, key)

    def test_perfect_nested_belief_pair(self):
        rows = []
        for side, inner, answer in (
            ("observed", "linen chest", "archive drawer"),
            ("hidden", "metal trunk", "blue suitcase"),
        ):
            target = {
                "tom_order": 2,
                "belief_chain": ["Alice", "Bob"],
                "object": "passport",
                "reasoning_mode": "nested_belief",
                "belief_trace": [
                    {"belief_chain": ["Bob"], "location": inner},
                    {
                        "belief_chain": ["Alice", "Bob"],
                        "location": answer,
                    },
                ],
                "answer": answer,
            }
            rows.append(
                {
                    "global_sample_id": f"nested-{side}",
                    "global_pair_id": "nested-pair-1",
                    "source_dataset": "symbolic-tom-v3",
                    "question_order": 2,
                    "intervention_type": side,
                    "process_target": target,
                    "response": json.dumps(target),
                }
            )
        overall = evaluate_predictions(rows)["overall"]
        self.assertEqual(overall["core_state_accuracy"], 1.0)
        self.assertEqual(overall["answer_state_consistency"], 1.0)
        self.assertEqual(overall["pair_accuracy"], 1.0)
        self.assertEqual(overall["intervention_sensitivity"], 1.0)


if __name__ == "__main__":
    unittest.main()
