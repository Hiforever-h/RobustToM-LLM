import json
import unittest

from rft.build_dataset import build_dataset
from rft.score_candidates import score_candidates


def make_target(order, chain, answer, observed):
    if order == 0:
        return {
            "tom_order": 0,
            "belief_chain": [],
            "object": "passport",
            "reasoning_mode": "world_state",
            "world_state": answer,
            "answer": answer,
        }
    return {
        "tom_order": order,
        "belief_chain": chain,
        "object": "passport",
        "reasoning_mode": "belief",
        "final_move_observed": observed,
        "nested_belief": answer,
        "answer": answer,
    }


class RejectionTest(unittest.TestCase):
    def test_only_full_reward_complete_pairs_enter_dataset(self):
        rows = []
        for side, answer, observed in (("observed", "linen chest", True), ("hidden", "archive drawer", False)):
            sample = f"pair-1-{side}"
            record = {
                "global_sample_id": sample,
                "global_pair_id": "pair-1",
                "process_prompt": "Return JSON.",
                "process_target": make_target(1, ["Alice"], answer, observed),
                "source_dataset": "hi-tom",
                "question_order": 1,
                "intervention_type": side,
            }
            rows.append({
                **record,
                "candidate_id": sample + "-candidate",
                "raw_response": json.dumps(record["process_target"]),
                "generation_reached_eos": True,
                "candidate_index": 0,
                "token_count": 20,
            })
        rows.append({
            **rows[0],
            "candidate_id": "rejected",
            "raw_response": "```json\n" + json.dumps(rows[0]["process_target"]) + "\n```",
        })
        scored, _ = score_candidates(rows)
        output, manifest = build_dataset(scored, min_samples=2, max_samples=3000)
        self.assertEqual(manifest["final_sample_count"], 2)
        self.assertEqual(len(output), 2)
        self.assertTrue(all(row["process_reward"] == 1.0 for row in output))

    def test_incomplete_pair_is_rejected(self):
        record = {
            "global_sample_id": "pair-only-observed",
            "global_pair_id": "pair-only",
            "process_prompt": "Return JSON.",
            "process_target": make_target(0, [], "linen chest", True),
            "source_dataset": "hi-tom",
            "question_order": 0,
            "intervention_type": "observed",
        }
        candidate = {**record, "raw_response": json.dumps(record["process_target"]), "generation_reached_eos": True}
        scored, _ = score_candidates([candidate])
        output, manifest = build_dataset(scored, min_samples=0)
        self.assertEqual(len(output), 1)
        self.assertEqual(manifest["final_sample_count"], 1)
        self.assertEqual(manifest["incomplete_pair_count"], 1)

        pair_output, pair_manifest = build_dataset(
            scored, min_samples=0, require_complete_pairs=True
        )
        self.assertEqual(pair_output, [])
        self.assertEqual(pair_manifest["final_sample_count"], 0)

    def test_default_keeps_multiple_full_reward_trajectories(self):
        target = make_target(0, [], "linen chest", True)
        rows = []
        for index in range(3):
            rows.append(
                {
                    "global_sample_id": "sample-1",
                    "global_pair_id": "pair-1",
                    "process_prompt": "Return JSON.",
                    "process_target": target,
                    "source_dataset": "hi-tom",
                    "question_order": 0,
                    "intervention_type": "observed",
                    "candidate_id": f"candidate-{index}",
                    "raw_response": json.dumps(target, separators=(",", ":")),
                    "generation_reached_eos": True,
                    "candidate_index": index,
                    "token_count": 20 + index,
                }
            )
        scored, _ = score_candidates(rows)
        output, manifest = build_dataset(scored, min_samples=3, max_samples=3)
        self.assertEqual(len(output), 3)
        self.assertEqual(manifest["selected_prompt_count"], 1)


if __name__ == "__main__":
    unittest.main()
