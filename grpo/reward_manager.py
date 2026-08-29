"""Function-based v3 process reward manager for verl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from grpo.metrics import summarize_reward_records
from grpo.natural_cot_reward_manager import NaturalCoTRewardManager
from rft.evaluate import evaluate_predictions
from rft.reward import score_process_output
from scripts.reward import (
    DeepSeekJudge,
    JudgeConfig,
    NaturalCoTReward,
    RewardConfig,
)


class ProcessRewardManager:
    """Score decoded response tokens with the unchanged RFT v3 scorer."""

    def __init__(self, tokenizer: Any, num_examine: int = 0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.last_records: list[dict[str, Any]] = []

    def __call__(self, data: Any) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        records: list[dict[str, Any]] = []
        for index in range(len(data)):
            item = data[index]
            prompt_length = item.batch["prompts"].shape[-1]
            response_ids = item.batch["responses"]
            valid_response_length = int(
                item.batch["attention_mask"][prompt_length:].sum().item()
            )
            valid_response_ids = response_ids[:valid_response_length]
            response = self.tokenizer.decode(
                valid_response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            reward_model = item.non_tensor_batch["reward_model"]
            ground_truth = reward_model["ground_truth"]
            target = (
                json.loads(ground_truth)
                if isinstance(ground_truth, str)
                else ground_truth
            )
            if not isinstance(target, dict):
                raise ValueError(
                    "reward_model.ground_truth must decode to a JSON object"
                )
            result = score_process_output(response, target)
            if valid_response_length > 0:
                reward_tensor[index, valid_response_length - 1] = float(
                    result["reward"]
                )

            extra_info = item.non_tensor_batch.get("extra_info", {})
            eos_reached = bool(
                valid_response_length > 0
                and int(valid_response_ids[-1].item()) == self.tokenizer.eos_token_id
            )
            record = {
                **(dict(extra_info) if isinstance(extra_info, dict) else {}),
                "response": response,
                "raw_response": response,
                "process_target": target,
                "token_count": valid_response_length,
                "generation_reached_eos": eos_reached,
                "result": result,
            }
            records.append(record)
            if index < self.num_examine:
                print(f"[RobustToM response]\n{response}\n[reward={result['reward']}]")
        self.last_records = records
        return reward_tensor

    def last_metrics(self, prefix: str = "reward") -> dict[str, float]:
        return summarize_reward_records(self.last_records, prefix=prefix)

    @staticmethod
    def validation_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
        nested = evaluate_predictions(records)
        metrics: dict[str, float] = {}
        for section, section_value in nested.items():
            if section == "overall" or all(
                isinstance(value, (int, float)) for value in section_value.values()
            ):
                for name, value in section_value.items():
                    if isinstance(value, (int, float)):
                        metrics[f"val/{section}/{name}"] = float(value)
                continue
            for bucket, bucket_metrics in section_value.items():
                for name, value in bucket_metrics.items():
                    if isinstance(value, (int, float)):
                        metrics[f"val/{section}/{bucket}/{name}"] = float(value)
        return metrics


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def build_reward_manager(
    tokenizer: Any,
    reward_config: Any,
    rollout_n: int,
    *,
    validation: bool = False,
    num_examine: int = 0,
) -> ProcessRewardManager | NaturalCoTRewardManager:
    """Build the configured legacy JSON or natural-CoT RewardManager."""
    mode = str(_config_get(reward_config, "mode", "json_rule"))
    if mode == "json_rule":
        return ProcessRewardManager(tokenizer=tokenizer, num_examine=num_examine)
    if mode != "natural_cot_judge":
        raise ValueError(f"Unsupported reward.mode: {mode}")

    judge_section = _config_get(reward_config, "judge", {})
    validation_section = _config_get(reward_config, "validation", {})
    judge_enabled = not validation or bool(
        _config_get(validation_section, "judge_enabled", False)
    )
    if validation and judge_enabled:
        raise ValueError(
            "Packed Judge validation is not supported by the deterministic "
            "one-response validation rollout; set reward.validation.judge_enabled=false"
        )

    cache_enabled = bool(_config_get(judge_section, "cache_enabled", True))
    raw_cache_dir = _config_get(judge_section, "cache_dir")
    cache_dir = Path(str(raw_cache_dir)) if cache_enabled and raw_cache_dir else None
    judge_config = JudgeConfig(
        base_url=str(
            _config_get(judge_section, "base_url", "https://api.deepseek.com")
        ),
        model=str(_config_get(judge_section, "model", "deepseek-v4-flash")),
        timeout_seconds=float(_config_get(judge_section, "timeout_seconds", 60.0)),
        max_tokens=int(_config_get(judge_section, "max_tokens", 3000)),
        retries=int(_config_get(judge_section, "retries", 4)),
        retry_backoff_seconds=float(
            _config_get(judge_section, "retry_backoff_seconds", 2.0)
        ),
        thinking=str(_config_get(judge_section, "thinking", "disabled")),
        cache_dir=cache_dir,
    )
    weights = RewardConfig(
        process_weight=float(_config_get(reward_config, "process_weight", 0.8)),
        answer_weight=float(_config_get(reward_config, "answer_weight", 0.2)),
        structure_weight=float(_config_get(reward_config, "structure_weight", 0.0)),
        state_weight_within_step=float(
            _config_get(reward_config, "state_weight_within_step", 0.4)
        ),
        reasoning_weight_within_step=float(
            _config_get(reward_config, "reasoning_weight_within_step", 0.6)
        ),
    )
    scorer = NaturalCoTReward(DeepSeekJudge(judge_config), weights)
    return NaturalCoTRewardManager(
        tokenizer=tokenizer,
        scorer=scorer,
        expected_group_size=1 if validation else int(rollout_n),
        judge_enabled=judge_enabled,
        max_workers=int(_config_get(judge_section, "max_workers", 8)),
        require_rollout_metadata=not validation,
        num_examine=num_examine,
    )


def build_reward_managers(
    tokenizer: Any,
    reward_config: Any,
    rollout_n: int,
) -> tuple[
    ProcessRewardManager | NaturalCoTRewardManager,
    ProcessRewardManager | NaturalCoTRewardManager,
]:
    """Build the paired training and validation RewardManagers."""
    return (
        build_reward_manager(
            tokenizer,
            reward_config,
            rollout_n,
            validation=False,
            num_examine=0,
        ),
        build_reward_manager(
            tokenizer,
            reward_config,
            rollout_n,
            validation=True,
            num_examine=1,
        ),
    )
