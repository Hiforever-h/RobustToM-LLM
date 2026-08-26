"""Function-based v3 process reward manager for verl."""

from __future__ import annotations

import json
from typing import Any

import torch

from grpo.metrics import summarize_reward_records
from rft.evaluate import evaluate_predictions
from rft.reward import score_process_output


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
            valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            response = self.tokenizer.decode(
                valid_response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            reward_model = item.non_tensor_batch["reward_model"]
            ground_truth = reward_model["ground_truth"]
            target = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
            if not isinstance(target, dict):
                raise ValueError("reward_model.ground_truth must decode to a JSON object")
            result = score_process_output(response, target)
            if valid_response_length > 0:
                reward_tensor[index, valid_response_length - 1] = float(result["reward"])

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
