import json
import unittest

from rft.reward import score_process_output


class RewardTest(unittest.TestCase):
    def setUp(self):
        self.target = {
            "tom_order": 2,
            "belief_chain": ["Alice", "Bob"],
            "object": "passport",
            "reasoning_mode": "belief",
            "final_move_observed": False,
            "nested_belief": "archive drawer",
            "answer": "archive drawer",
        }

    def test_exact_json_gets_full_reward(self):
        self.assertEqual(score_process_output(json.dumps(self.target), self.target)["reward"], 1.0)

    def test_fence_is_not_accepted(self):
        output = "```json\n" + json.dumps(self.target) + "\n```"
        result = score_process_output(output, self.target)
        self.assertEqual(result["reward"], 0.95)
        self.assertFalse(result["checks"]["format"])

    def test_visibility_gates_nested_state(self):
        prediction = dict(self.target, final_move_observed=True)
        result = score_process_output(prediction, self.target)
        self.assertEqual(result["reward"], 0.25)
        self.assertFalse(result["checks"]["final_move_observed"])

    def test_malformed_json_is_zero(self):
        self.assertEqual(score_process_output("not json", self.target)["reward"], 0.0)

    def test_nested_belief_trace_requires_all_steps_for_answer_reward(self):
        target = {
            "tom_order": 2,
            "belief_chain": ["Alice", "Bob"],
            "object": "passport",
            "reasoning_mode": "nested_belief",
            "belief_trace": [
                {"belief_chain": ["Bob"], "location": "linen chest"},
                {
                    "belief_chain": ["Alice", "Bob"],
                    "location": "archive drawer",
                },
            ],
            "answer": "archive drawer",
        }
        self.assertEqual(score_process_output(target, target)["reward"], 1.0)
        prediction = json.loads(json.dumps(target))
        prediction["belief_trace"][1]["location"] = "metal trunk"
        result = score_process_output(prediction, target)
        self.assertEqual(result["checks"]["belief_trace_steps"], [True, False])
        self.assertEqual(result["components"]["answer"], 0.0)
