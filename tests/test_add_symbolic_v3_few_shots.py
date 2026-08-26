import json
import unittest

from scripts.add_symbolic_v3_few_shots import (
    FEW_SHOT_COUNT,
    FEW_SHOT_MARKER,
    augment_record,
)
from scripts.generate_symbolic_counterfactual_v3 import TokenCounter, process_prompt


class SymbolicV3FewShotTest(unittest.TestCase):
    def test_adds_three_demonstrations_and_recomputes_lengths(self) -> None:
        target = {
            "tom_order": 2,
            "belief_chain": ["Alice", "Bob"],
            "object": "key",
            "reasoning_mode": "nested_belief",
            "belief_trace": [
                {"belief_chain": ["Bob"], "location": "blue_box"},
                {"belief_chain": ["Alice", "Bob"], "location": "red_box"},
            ],
            "answer": "red_box",
        }
        prompt = process_prompt(
            "1 Alice and Bob jointly watched the key enter the red_box.\n",
            "Where does Alice think Bob thinks the key is?",
            "A. red_box, B. blue_box",
        )
        row = {
            "global_sample_id": "sample-1",
            "process_target_version": "2.0",
            "question_order": 2,
            "global_pair_id": "pair-1",
            "process_target": target,
            "process_response": json.dumps(target, separators=(",", ":")),
            "process_prompt": prompt,
        }

        augmented = augment_record(row, TokenCounter(None), max_tokens=2048)
        self.assertEqual(augmented["few_shot_count"], FEW_SHOT_COUNT)
        self.assertEqual(augmented["process_prompt"].count(FEW_SHOT_MARKER), 1)
        self.assertIn("Demonstration 1", augmented["process_prompt"])
        self.assertIn("Demonstration 2", augmented["process_prompt"])
        self.assertIn("Demonstration 3", augmented["process_prompt"])
        self.assertIn("Where does Alice think Bob thinks the key is?", augmented["process_prompt"])
        self.assertLessEqual(augmented["process_sequence_token_count"], 2048)
        self.assertEqual(augmented["process_response"], row["process_response"])

    def test_rejects_augmenting_a_prompt_twice(self) -> None:
        target = {
            "tom_order": 1,
            "belief_chain": ["Alice"],
            "object": "key",
            "reasoning_mode": "nested_belief",
            "belief_trace": [
                {"belief_chain": ["Alice"], "location": "red_box"}
            ],
            "answer": "red_box",
        }
        row = {
            "global_sample_id": "sample-1",
            "process_target_version": "2.0",
            "process_target": target,
            "process_response": json.dumps(target),
            "process_prompt": (
                f"Return JSON.\n\n{FEW_SHOT_MARKER}\n\nStory:\n1 Example."
            ),
        }
        with self.assertRaises(ValueError):
            augment_record(row, TokenCounter(None), max_tokens=2048)


if __name__ == "__main__":
    unittest.main()
