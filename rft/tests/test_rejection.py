import unittest

from rft.build_dataset import build_dataset
from rft.score_candidates import score_candidates
from scripts.reward import NaturalCoTReward


def make_target(answer: str) -> dict:
    return {
        "tom_order": 1,
        "belief_chain": ["Alice"],
        "object": "passport",
        "reasoning_mode": "nested_belief",
        "belief_trace": [{"belief_chain": ["Alice"], "location": answer}],
        "answer": answer,
    }


def response(answer: str) -> str:
    return (
        "Think 1:\n"
        "Alice saw the relevant move and therefore believes this location.\n"
        f"State: {answer}\n"
        f"Answer: {answer}"
    )


class FakeJudge:
    def __init__(self, reasoning_score: float = 1.0):
        self.reasoning_score = reasoning_score

    def score_group(self, group):
        return {
            "group_id": group.group_id,
            "evaluations": [
                {
                    "candidate_id": candidate_id,
                    "reasoning_scores": [self.reasoning_score]
                    * int(group.target["tom_order"]),
                }
                for candidate_id in group.resolved_candidate_ids()
            ],
            "raw_content": "{}",
        }


class RejectionTest(unittest.TestCase):
    scorer = NaturalCoTReward(FakeJudge())

    @staticmethod
    def record(side: str, answer: str, pair: str = "pair-1") -> dict:
        sample = f"{pair}-{side}"
        return {
            "global_sample_id": sample,
            "global_pair_id": pair,
            "process_prompt": "Natural actor prompt.",
            "judge_prompt": "Story and question.",
            "process_prompt_version": "natural-cot-think-state-v1",
            "process_target": make_target(answer),
            "source_dataset": "symbolic-tom-v3",
            "question_order": 1,
            "intervention_type": side,
        }

    def test_only_full_reward_valid_structure_eos_enters_dataset(self):
        rows = []
        for side, answer in (
            ("observed", "linen chest"),
            ("hidden", "archive drawer"),
        ):
            record = self.record(side, answer)
            rows.append(
                {
                    **record,
                    "candidate_id": record["global_sample_id"] + "-candidate",
                    "raw_response": response(answer),
                    "generation_reached_eos": True,
                    "candidate_index": 0,
                    "token_count": 20,
                }
            )
        rows.append(
            {
                **rows[0],
                "candidate_id": "rejected",
                "raw_response": response("linen chest")
                + "\nThis content appears after Answer.",
            }
        )
        scored, _ = score_candidates(rows, scorer=self.scorer)
        output, manifest = build_dataset(scored, min_samples=2, max_samples=3000)
        self.assertEqual(manifest["final_sample_count"], 2)
        self.assertEqual(len(output), 2)
        self.assertTrue(all(row["process_reward"] == 1.0 for row in output))

    def test_incomplete_pair_is_optional(self):
        record = self.record("observed", "linen chest", pair="pair-only")
        candidate = {
            **record,
            "raw_response": response("linen chest"),
            "generation_reached_eos": True,
        }
        scored, _ = score_candidates([candidate], scorer=self.scorer)
        output, manifest = build_dataset(scored, min_samples=0)
        self.assertEqual(len(output), 1)
        self.assertEqual(manifest["incomplete_pair_count"], 1)

        pair_output, pair_manifest = build_dataset(
            scored, min_samples=0, require_complete_pairs=True
        )
        self.assertEqual(pair_output, [])
        self.assertEqual(pair_manifest["final_sample_count"], 0)

    def test_default_keeps_multiple_full_reward_trajectories(self):
        record = self.record("observed", "linen chest")
        rows = [
            {
                **record,
                "candidate_id": f"candidate-{index}",
                "raw_response": response("linen chest"),
                "generation_reached_eos": True,
                "candidate_index": index,
                "token_count": 20 + index,
            }
            for index in range(3)
        ]
        scored, _ = score_candidates(rows, scorer=self.scorer)
        output, manifest = build_dataset(
            scored, min_samples=3, max_samples=3
        )
        self.assertEqual(len(output), 3)
        self.assertEqual(manifest["selected_prompt_count"], 1)

        deduplicated, dedup_manifest = build_dataset(
            scored,
            min_samples=1,
            max_samples=3,
            deduplicate_semantic=True,
        )
        self.assertEqual(len(deduplicated), 1)
        self.assertEqual(dedup_manifest["duplicate_candidate_count"], 2)

    def test_rule_only_never_accepts(self):
        record = self.record("observed", "linen chest")
        candidate = {
            **record,
            "raw_response": response("linen chest"),
            "generation_reached_eos": True,
        }
        scored, manifest = score_candidates([candidate], rule_only=True)
        self.assertFalse(scored[0]["accepted"])
        self.assertEqual(scored[0]["acceptance_reason"], "judge_disabled")
        self.assertTrue(manifest["rule_only"])

    def test_continuous_good_reasoning_does_not_need_full_reward(self):
        record = self.record("observed", "linen chest")
        candidate = {
            **record,
            "raw_response": response("linen chest"),
            "generation_reached_eos": True,
        }
        scorer = NaturalCoTReward(FakeJudge(reasoning_score=0.75))
        scored, manifest = score_candidates([candidate], scorer=scorer)
        self.assertEqual(scored[0]["score"]["reward"], 0.88)
        self.assertTrue(scored[0]["accepted"])
        self.assertEqual(manifest["acceptance_policy"]["min_reward"], 0.88)
        output, _ = build_dataset(scored, min_samples=1)
        self.assertEqual(output[0]["process_reward"], 0.88)

        stricter, _ = score_candidates(
            [candidate], scorer=scorer, min_reward=0.9
        )
        self.assertFalse(stricter[0]["accepted"])
        self.assertEqual(
            stricter[0]["acceptance_reason"], "reward_below_threshold"
        )


if __name__ == "__main__":
    unittest.main()
