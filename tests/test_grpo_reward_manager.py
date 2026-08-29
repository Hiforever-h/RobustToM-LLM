import json
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is required")
class ProcessRewardManagerTest(unittest.TestCase):
    @staticmethod
    def target():
        return {
            "tom_order": 1,
            "belief_chain": ["Ada"],
            "object": "key",
            "reasoning_mode": "nested_belief",
            "belief_trace": [
                {"belief_chain": ["Ada"], "location": "blue_box"},
            ],
            "answer": "blue_box",
        }

    def test_decodes_only_response_and_matches_rft_reward(self):
        from grpo.reward_manager import ProcessRewardManager

        target = self.target()
        response = json.dumps(target, separators=(",", ":"))

        class Tokenizer:
            eos_token_id = 99

            def decode(self, token_ids, **kwargs):
                self.decoded = token_ids.tolist()
                return response

        class Item:
            batch = {
                "prompts": torch.tensor([10, 11, 12]),
                "responses": torch.tensor([20, 99, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1, 0]),
            }
            non_tensor_batch = {
                "reward_model": {"ground_truth": response},
                "extra_info": {"global_sample_id": "sample-1"},
            }

        class Data:
            batch = {"responses": torch.tensor([[20, 99, 0]])}

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return Item()

        tokenizer = Tokenizer()
        manager = ProcessRewardManager(tokenizer)
        rewards = manager(Data())
        self.assertEqual(tokenizer.decoded, [20, 99])
        self.assertEqual(rewards.sum().item(), 1.0)
        self.assertEqual(manager.last_records[0]["result"]["reward"], 1.0)
        self.assertTrue(manager.last_records[0]["generation_reached_eos"])

    def test_validation_metrics_flattens_bucketed_and_subset_sections(self):
        from grpo.reward_manager import ProcessRewardManager

        target = self.target()
        response = json.dumps(target)
        record = {
            "global_sample_id": "sample-1",
            "global_pair_id": "pair-1",
            "source_dataset": "unit",
            "question_order": 1,
            "intervention_type": "move",
            "shortcut_conflict": True,
            "last_mention_conflict": True,
            "shortcut_prediction": "red_box",
            "last_mentioned_container": "red_box",
            "response": response,
            "process_target": target,
            "token_count": 20,
            "generation_reached_eos": True,
        }

        metrics = ProcessRewardManager.validation_metrics([record])
        self.assertEqual(metrics["val/overall/full_reward_rate"], 1.0)
        self.assertEqual(metrics["val/source_dataset/unit/full_reward_rate"], 1.0)
        self.assertEqual(metrics["val/shortcut_conflict/full_reward_rate"], 1.0)
        self.assertEqual(metrics["val/last_mention_conflict/full_reward_rate"], 1.0)


@unittest.skipIf(torch is None, "torch is required")
class NaturalCoTRewardManagerTest(unittest.TestCase):
    @staticmethod
    def target():
        return ProcessRewardManagerTest.target()

    @staticmethod
    def response():
        return (
            "Think 1:\nAda retained the last move she observed.\n"
            "State: blue_box\n\nAnswer: blue_box"
        )

    class FakeJudge:
        def score_group(self, group):
            evaluations = []
            for candidate_id in group.resolved_candidate_ids():
                score = 0.0 if candidate_id == "c00" else 1.0
                evaluations.append(
                    {
                        "candidate_id": candidate_id,
                        "reasoning_scores": [score],
                        "raw_step_count": 1,
                        "step_count_normalized": False,
                    }
                )
            return {
                "group_id": group.group_id,
                "evaluations": evaluations,
                "elapsed_seconds": 0.25,
                "attempts": 1,
                "cached": False,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                "raw_content": json.dumps({"evaluations": evaluations}),
            }

    def make_data(self, order):
        response = self.response()

        class Tokenizer:
            eos_token_id = 99

            def decode(self, token_ids, **kwargs):
                return response

        class Item:
            def __init__(self, uid, rollout_index):
                self.batch = {
                    "prompts": torch.tensor([10, 11, 12]),
                    "responses": torch.tensor([20, 99, 0]),
                    "attention_mask": torch.tensor([1, 1, 1, 1, 1, 0]),
                }
                self.non_tensor_batch = {
                    "uid": uid,
                    "rollout_index": rollout_index,
                    "judge_prompt": "A natural ToM story and question.",
                    "reward_model": {
                        "ground_truth": NaturalCoTRewardManagerTest.target()
                    },
                    "extra_info": {"global_sample_id": uid},
                }

        items = [Item(uid, rollout_index) for uid, rollout_index in order]

        class Data:
            batch = {"responses": torch.tensor([[20, 99, 0]] * len(items))}

            def __len__(self):
                return len(items)

            def __getitem__(self, index):
                return items[index]

        return Tokenizer(), Data()

    def test_shuffled_groups_map_rewards_back_by_uid_and_rollout_index(self):
        from grpo.natural_cot_reward_manager import NaturalCoTRewardManager
        from scripts.reward import NaturalCoTReward

        tokenizer, data = self.make_data([("b", 1), ("a", 0), ("b", 0), ("a", 1)])
        manager = NaturalCoTRewardManager(
            tokenizer=tokenizer,
            scorer=NaturalCoTReward(self.FakeJudge()),
            expected_group_size=2,
            max_workers=2,
        )
        rewards = manager(data).sum(-1).tolist()
        for actual, expected in zip(rewards, [1.0, 0.52, 0.52, 1.0]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            [(row["uid"], row["rollout_index"]) for row in manager.last_records],
            [("b", 1), ("a", 0), ("b", 0), ("a", 1)],
        )
        metrics = manager.last_metrics()
        self.assertEqual(metrics["reward/judge/group_count"], 2.0)
        self.assertEqual(metrics["reward/judge/total_tokens"], 24.0)
        self.assertEqual(metrics["reward/format_progress_mean"], 1.0)
        self.assertEqual(metrics["reward/valid_structure_prefix_fraction_mean"], 1.0)
        self.assertEqual(metrics["reward/structure_bonus_mean"], 0.0)

    def test_build_manager_propagates_grpo_structure_weight(self):
        from grpo.reward_manager import build_reward_manager

        manager = build_reward_manager(
            tokenizer=object(),
            reward_config={
                "mode": "natural_cot_judge",
                "process_weight": 0.75,
                "structure_weight": 0.05,
                "answer_weight": 0.2,
                "judge": {"cache_enabled": False},
            },
            rollout_n=16,
        )
        weights = manager.scorer.reward_config
        self.assertEqual(weights.process_weight, 0.75)
        self.assertEqual(weights.structure_weight, 0.05)
        self.assertEqual(weights.answer_weight, 0.2)

    def test_missing_rollout_index_fails_before_judge(self):
        from grpo.natural_cot_reward_manager import NaturalCoTRewardManager
        from scripts.reward import NaturalCoTReward

        tokenizer, data = self.make_data([("a", 0), ("a", 2)])
        manager = NaturalCoTRewardManager(
            tokenizer=tokenizer,
            scorer=NaturalCoTReward(self.FakeJudge()),
            expected_group_size=2,
        )
        with self.assertRaisesRegex(ValueError, "rollout_index mismatch"):
            manager(data)

    def test_arrow_style_nested_arrays_are_normalized_before_scoring(self):
        from grpo.natural_cot_reward_manager import NaturalCoTRewardManager

        class Array:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return self.values

        target = self.target()
        target["belief_chain"] = Array(target["belief_chain"])
        target["belief_trace"] = Array(
            [
                {
                    "belief_chain": Array(step["belief_chain"]),
                    "location": step["location"],
                }
                for step in target["belief_trace"]
            ]
        )
        normalized = NaturalCoTRewardManager._to_builtin(target)
        self.assertIsInstance(normalized["belief_chain"], list)
        self.assertIsInstance(normalized["belief_trace"], list)
        self.assertIsInstance(normalized["belief_trace"][0]["belief_chain"], list)


if __name__ == "__main__":
    unittest.main()
