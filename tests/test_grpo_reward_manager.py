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


if __name__ == "__main__":
    unittest.main()
