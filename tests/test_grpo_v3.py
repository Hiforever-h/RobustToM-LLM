import importlib.util
import json
import math
import sys
import types
import unittest
from pathlib import Path

import yaml

from grpo.build_dataset import build_parquet_row
from grpo.metrics import summarize_group_rewards, summarize_reward_records
from grpo.prompt import ORDER_TRACE_INSTRUCTION, build_grpo_prompt
from rft.common import read_jsonl
from rft.reward import score_process_output


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = f"<user>{messages[0]['content']}</user><assistant>"
        return [ord(char) for char in rendered] if tokenize else rendered

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


class GrpoPromptAndDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = read_jsonl(ROOT / "data/counterfactual_process_reward_v3/train.jsonl")[0]
        cls.derived = read_jsonl(ROOT / "data/rft/derived_v3_fewshot/train.jsonl")
        cls.derived_by_id = {row["global_sample_id"]: row for row in cls.derived}

    def test_prompt_matches_existing_few_shot_augmentation_plus_instruction(self):
        import scripts

        self.assertEqual(
            Path(scripts.__file__).resolve(),
            (ROOT / "scripts/__init__.py").resolve(),
        )
        actual = build_grpo_prompt(self.raw["process_prompt"])
        without_clarification = actual.replace(f" {ORDER_TRACE_INSTRUCTION}", "", 1)
        expected = self.derived_by_id[self.raw["global_sample_id"]]["process_prompt"]
        self.assertEqual(without_clarification, expected)
        self.assertEqual(actual.count(ORDER_TRACE_INSTRUCTION), 1)
        self.assertEqual(actual.count("Nested-belief demonstrations (3-shot):"), 1)

    def test_parquet_row_uses_process_prompt_and_string_ground_truth(self):
        row = build_parquet_row(
            self.raw,
            index=7,
            tokenizer=FakeTokenizer(),
            max_prompt_length=100000,
            max_response_length=100000,
        )
        self.assertEqual(row["data_source"], "robust_tom_process_v3")
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertIn(ORDER_TRACE_INSTRUCTION, row["prompt"][0]["content"])
        self.assertEqual(json.loads(row["reward_model"]["ground_truth"]), self.raw["process_target"])
        self.assertEqual(row["extra_info"]["index"], 7)

    def test_parquet_row_rejects_prompt_overflow(self):
        with self.assertRaises(ValueError):
            build_parquet_row(
                self.raw,
                index=0,
                tokenizer=FakeTokenizer(),
                max_prompt_length=10,
                max_response_length=100000,
            )


class GrpoConfigTest(unittest.TestCase):
    def test_training_shape_and_hyperparameters(self):
        with (ROOT / "verl/trainer/config/robust_tom_grpo.yaml").open(
            encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)

        data = config["data"]
        actor_rollout_ref = config["actor_rollout_ref"]
        actor = actor_rollout_ref["actor"]
        rollout = actor_rollout_ref["rollout"]
        trainer = config["trainer"]

        self.assertEqual(data["train_files"], (
            "data/grpo/counterfactual_process_reward_v3_fewshot/train.parquet"
        ))
        self.assertEqual(actor_rollout_ref["model"]["path"], "runs/final")
        self.assertEqual(actor_rollout_ref["ref"]["model_path"], "runs/final")
        self.assertEqual(data["max_prompt_length"], 2048)
        self.assertEqual(data["max_response_length"], 256)
        self.assertEqual(data["train_batch_size"], 8)
        self.assertEqual(rollout["n"], 16)
        self.assertEqual(rollout["temperature"], 1.0)
        self.assertEqual(actor["ppo_mini_batch_size"], 8 * 16)
        self.assertEqual(actor["optim"]["lr"], 5e-7)
        self.assertEqual(actor["clip_ratio_low"], 0.2)
        self.assertEqual(actor["clip_ratio_high"], 0.3)
        self.assertEqual(trainer["total_epochs"], 2)
        self.assertEqual(trainer["total_training_steps"], 800)
        self.assertEqual(trainer["logger"], ["console", "wandb"])
        self.assertTrue(trainer["progress_bar"])
        self.assertEqual((3200 // data["train_batch_size"]) * trainer["total_epochs"], 800)


class GrpoMetricsTest(unittest.TestCase):
    def test_group_metrics_use_population_std_and_validate_group_size(self):
        metrics = summarize_group_rewards(
            [1.0, 3.0, 2.0, 2.0],
            ["a", "a", "b", "b"],
            expected_group_size=2,
        )
        self.assertAlmostEqual(metrics["grpo/group_reward_std_mean"], 0.5)
        self.assertAlmostEqual(metrics["grpo/zero_variance_group_rate"], 0.5)
        with self.assertRaises(ValueError):
            summarize_group_rewards([1.0], ["a"], expected_group_size=2)

    def test_reward_summary_matches_rft_scorer(self):
        target = {
            "tom_order": 2,
            "belief_chain": ["Alice", "Bob"],
            "object": "key",
            "reasoning_mode": "nested_belief",
            "belief_trace": [
                {"belief_chain": ["Bob"], "location": "red_box"},
                {"belief_chain": ["Alice", "Bob"], "location": "blue_box"},
            ],
            "answer": "blue_box",
        }
        prediction = json.loads(json.dumps(target))
        prediction["belief_trace"][0]["location"] = "green_box"
        result = score_process_output(prediction, target)
        metrics = summarize_reward_records([
            {"result": result, "generation_reached_eos": True},
        ])
        self.assertAlmostEqual(result["components"]["belief_trace"], 0.275)
        self.assertEqual(result["components"]["answer"], 0.0)
        self.assertAlmostEqual(metrics["reward/belief_trace_step_accuracy"], 0.5)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required")
class VerlCoreAlgorithmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        fake_verl = types.ModuleType("verl")
        fake_utils = types.ModuleType("verl.utils")
        fake_functional = types.ModuleType("verl.utils.torch_functional")

        def masked_mean(values, mask, axis=None):
            return (values * mask).sum(dim=axis) / mask.sum(dim=axis)

        fake_functional.masked_mean = masked_mean
        old_modules = {
            name: sys.modules.get(name)
            for name in ("verl", "verl.utils", "verl.utils.torch_functional")
        }
        sys.modules["verl"] = fake_verl
        sys.modules["verl.utils"] = fake_utils
        sys.modules["verl.utils.torch_functional"] = fake_functional
        try:
            spec = importlib.util.spec_from_file_location(
                "isolated_core_algos",
                ROOT / "verl/trainer/ppo/core_algos.py",
            )
            cls.core = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.core)
        finally:
            for name, module in old_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        cls.torch = torch

    def test_asymmetric_policy_clip(self):
        torch = self.torch
        ratio = torch.tensor([[0.7, 1.4]], dtype=torch.float32)
        old_log_prob = torch.zeros_like(ratio)
        log_prob = torch.log(ratio)
        advantages = torch.tensor([[-1.0, 1.0]])
        mask = torch.ones_like(ratio)
        loss, clipped, clipped_low, clipped_high, _ = self.core.compute_policy_loss(
            old_log_prob,
            log_prob,
            advantages,
            mask,
            cliprange_low=0.2,
            cliprange_high=0.3,
        )
        self.assertAlmostEqual(loss.item(), -0.25, places=6)
        self.assertAlmostEqual(clipped.item(), 1.0)
        self.assertAlmostEqual(clipped_low.item(), 0.5)
        self.assertAlmostEqual(clipped_high.item(), 0.5)

    def test_grpo_population_normalization_and_zero_variance(self):
        torch = self.torch
        rewards = torch.tensor([[1.0], [3.0], [2.0], [2.0]])
        mask = torch.ones_like(rewards)
        advantages, _ = self.core.compute_grpo_outcome_advantage(
            rewards,
            mask,
            index=["a", "a", "b", "b"],
        )
        self.assertTrue(torch.allclose(
            advantages.squeeze(-1),
            torch.tensor([-1.0, 1.0, 0.0, 0.0]),
            atol=2e-6,
        ))


if __name__ == "__main__":
    unittest.main()
