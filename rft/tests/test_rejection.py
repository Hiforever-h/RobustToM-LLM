import unittest

from rft.build_dataset import build_dataset
from rft.prompt import (
    COMPACT_NATURAL_COT_PROMPT_VERSION,
    NATURAL_COT_PROMPT_VERSION,
    compact_process_record,
)
from rft.score_candidates import prepare_scoring_source_rows, score_candidates
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
    @staticmethod
    def record(side: str, answer: str, pair: str = "pair-1") -> dict:
        sample = f"{pair}-{side}"
        return {
            "global_sample_id": sample,
            "global_pair_id": pair,
            "process_prompt": "Natural actor prompt.",
            "judge_prompt": "Story and question.",
            "process_prompt_version": NATURAL_COT_PROMPT_VERSION,
            "process_target": make_target(answer),
            "source_dataset": "symbolic-tom-v3",
            "question_order": 1,
            "intervention_type": side,
        }

    def test_state_answer_correct_valid_structure_eos_enters_dataset(self):
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
        scored, manifest = score_candidates(rows)
        self.assertFalse(manifest["judge_enabled"])
        self.assertEqual(manifest["judge_group_count"], 0)
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
        scored, _ = score_candidates([candidate])
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
        scored, _ = score_candidates(rows)
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

    def test_default_local_scoring_assigns_full_reward_without_judge(self):
        record = self.record("observed", "linen chest")
        candidate = {
            **record,
            "raw_response": response("linen chest"),
            "generation_reached_eos": True,
        }
        scored, manifest = score_candidates([candidate])
        self.assertTrue(scored[0]["accepted"])
        self.assertEqual(scored[0]["score"]["reward"], 1.0)
        self.assertEqual(scored[0]["score"]["scoring_mode"], "state_answer_binary")
        self.assertIsNone(scored[0]["judge_score"])
        self.assertEqual(
            scored[0]["acceptance_reason"],
            "state_answer_correct_valid_structure_eos",
        )
        self.assertFalse(manifest["judge_enabled"])

    def test_default_local_scoring_requires_both_state_and_answer(self):
        record = self.record("observed", "linen chest")
        candidate = {
            **record,
            "raw_response": (
                "Think 1:\n"
                "Alice saw the relevant move and therefore believes this location.\n"
                "State: archive drawer\n"
                "Answer: linen chest"
            ),
            "generation_reached_eos": True,
        }
        scored, _ = score_candidates([candidate])
        self.assertFalse(scored[0]["accepted"])
        self.assertEqual(scored[0]["score"]["reward"], 0.0)
        self.assertFalse(scored[0]["score"]["all_states_correct"])
        self.assertTrue(scored[0]["score"]["answer_correct"])
        self.assertEqual(scored[0]["acceptance_reason"], "state_incorrect")

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

    def test_compact_dataset_replaces_scaffold_but_keeps_sampled_response(self):
        record = self.record("observed", "linen chest")
        record["process_prompt"] = (
            "Story and question.\n\nReasoning rules:\nold detailed rules\n\n"
            "Required output format:\nThink 1:\n<reasoning>\nState: <location>"
        )
        candidate = {
            **record,
            "candidate_id": "compact-candidate",
            "raw_response": response("linen chest"),
            "generation_reached_eos": True,
        }
        scored, _ = score_candidates([candidate])
        output, manifest = build_dataset(
            scored, min_samples=1, compact_prompt=True
        )
        self.assertEqual(output[0]["accepted_response"], candidate["raw_response"])
        self.assertTrue(output[0]["process_prompt"].startswith("Story and question."))
        self.assertNotIn("Reasoning rules:", output[0]["process_prompt"])
        self.assertNotIn("Required output format:", output[0]["process_prompt"])
        self.assertNotIn("<", output[0]["process_prompt"])
        self.assertEqual(
            output[0]["process_prompt_version"],
            COMPACT_NATURAL_COT_PROMPT_VERSION,
        )
        self.assertTrue(manifest["compact_prompt"])
        self.assertEqual(
            manifest["process_prompt_version"],
            COMPACT_NATURAL_COT_PROMPT_VERSION,
        )

    def test_compact_sampling_source_can_be_scored_against_compact_data(self):
        source = self.record("observed", "linen chest")
        compact_candidate = {
            **compact_process_record(source),
            "candidate_id": "compact-sampled-candidate",
            "raw_response": response("linen chest"),
            "generation_reached_eos": True,
        }
        compact_source = prepare_scoring_source_rows([source], compact_prompt=True)
        scored, _ = score_candidates([compact_candidate], compact_source)
        self.assertTrue(scored[0]["accepted"])
        self.assertEqual(
            scored[0]["process_prompt_version"],
            COMPACT_NATURAL_COT_PROMPT_VERSION,
        )

    def test_manifest_reports_group_variance_and_think_step_counts(self):
        record = self.record("observed", "linen chest")
        candidates = [
            {
                **record,
                "candidate_id": "correct",
                "raw_response": response("linen chest"),
                "generation_reached_eos": True,
            },
            {
                **record,
                "candidate_id": "incorrect",
                "raw_response": response("archive drawer"),
                "generation_reached_eos": True,
            },
        ]
        _, manifest = score_candidates(candidates)
        group = manifest["group_reward_diagnostics"]
        self.assertEqual(group["group_count"], 1)
        self.assertEqual(group["group_size_counts"], {"2": 1})
        self.assertEqual(group["group_reward_std_mean"], 0.5)
        self.assertEqual(group["zero_variance_group_rate"], 0.0)
        self.assertEqual(group["all_zero_reward_group_rate"], 0.0)
        think = manifest["think_step_diagnostics"]["1"]
        self.assertEqual(think["actual_step_count_distribution"], {"1": 2})
        self.assertEqual(think["exact_step_count_rate"], 1.0)
        self.assertEqual(think["exact_numbered_sequence_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
