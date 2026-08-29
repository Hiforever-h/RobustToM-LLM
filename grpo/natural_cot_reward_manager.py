"""verl adapter for the natural-language ToM process reward.

Training batches are grouped by the stable ``(uid, rollout_index)`` metadata
inserted by ``RayPPOTrainer``. One packed Judge request is issued per uid, and
the resulting scalar rewards are written back to the final valid response
token at their original (possibly token-balanced) batch indices.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable, Mapping

import torch

from scripts.reward import (
    NaturalCoTReward,
    RewardGroup,
    canonical_json,
)


@dataclass(frozen=True)
class _BatchEntry:
    batch_index: int
    uid: str
    rollout_index: int
    judge_prompt: str
    target: Mapping[str, Any]
    response: str
    valid_response_length: int
    generation_reached_eos: bool
    extra_info: Mapping[str, Any]

    @property
    def candidate_id(self) -> str:
        return f"c{self.rollout_index:02d}"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _natural_metrics(
    records: Iterable[Mapping[str, Any]], prefix: str
) -> dict[str, float]:
    records = list(records)
    if not records:
        return {}

    combined = [record["combined"] for record in records]
    rules = [record["rule"] for record in records]
    parsed = [rule["parsed"] for rule in rules]
    state_flags = [bool(value) for rule in rules for value in rule["state_correct"]]
    reasoning_present = [
        bool(value) for rule in rules for value in rule["reasoning_present"]
    ]
    answer_correct = [bool(rule["answer_correct"]) for rule in rules]
    all_states_correct = [bool(rule["all_states_correct"]) for rule in rules]
    structure_ok = [bool(item["checks"]["structure_ok"]) for item in parsed]
    missing_steps = [bool(item["missing_indices"]) for item in parsed]
    duplicate_steps = [bool(item["duplicate_indices"]) for item in parsed]
    extra_steps = [bool(item["extra_indices"]) for item in parsed]

    metrics = {
        f"{prefix}/mean": _mean(float(result["reward"]) for result in combined),
        f"{prefix}/process_mean": _mean(
            float(result["process_reward"]) for result in combined
        ),
        f"{prefix}/answer_bonus_mean": _mean(
            float(result["answer_bonus"]) for result in combined
        ),
        f"{prefix}/structure_bonus_mean": _mean(
            float(result["structure_bonus"]) for result in combined
        ),
        f"{prefix}/format_progress_mean": _mean(
            float(result["format_progress"]) for result in combined
        ),
        f"{prefix}/valid_structure_prefix_fraction_mean": _mean(
            float(result["valid_structure_prefix_fraction"]) for result in combined
        ),
        f"{prefix}/structure_valid_rate": _mean(structure_ok),
        f"{prefix}/state_step_accuracy": _mean(state_flags),
        f"{prefix}/all_states_correct_rate": _mean(all_states_correct),
        f"{prefix}/answer_accuracy": _mean(answer_correct),
        f"{prefix}/answer_only_rate": _mean(
            answer_ok and not any(rule["state_correct"])
            for answer_ok, rule in zip(answer_correct, rules)
        ),
        f"{prefix}/answer_correct_state_trace_wrong_rate": _mean(
            answer_ok and not states_ok
            for answer_ok, states_ok in zip(answer_correct, all_states_correct)
        ),
        f"{prefix}/reasoning_present_step_rate": _mean(reasoning_present),
        f"{prefix}/missing_step_response_rate": _mean(missing_steps),
        f"{prefix}/duplicate_step_response_rate": _mean(duplicate_steps),
        f"{prefix}/extra_step_response_rate": _mean(extra_steps),
        f"{prefix}/eos_rate": _mean(
            bool(record.get("generation_reached_eos")) for record in records
        ),
    }
    if any(isinstance(record.get("judge"), Mapping) for record in records):
        reasoning_scores = [
            float(value) for result in combined for value in result["reasoning_scores"]
        ]
        effective_reasoning_scores = [
            float(value)
            for result in combined
            for value in result["effective_reasoning_scores"]
        ]
        metrics.update(
            {
                f"{prefix}/judge_reasoning_step_mean": _mean(reasoning_scores),
                f"{prefix}/effective_reasoning_step_mean": _mean(
                    effective_reasoning_scores
                ),
                f"{prefix}/answer_correct_process_imperfect_rate": _mean(
                    answer_ok and float(result["process_reward"]) < 1.0
                    for answer_ok, result in zip(answer_correct, combined)
                ),
            }
        )
    return metrics


def summarize_natural_cot_records(
    records: Iterable[Mapping[str, Any]], prefix: str = "reward"
) -> dict[str, float]:
    """Aggregate rule, Judge and transport metrics for one reward call."""
    records = list(records)
    metrics = _natural_metrics(records, prefix)
    if not records:
        return metrics

    group_metadata: dict[str, Mapping[str, Any]] = {}
    for record in records:
        metadata = record.get("judge_metadata")
        if isinstance(metadata, Mapping):
            group_metadata.setdefault(str(record["uid"]), metadata)
    if not group_metadata:
        return metrics

    metadata_values = list(group_metadata.values())
    usage_values = [
        metadata.get("billable_usage", metadata.get("usage"))
        for metadata in metadata_values
        if isinstance(metadata.get("billable_usage", metadata.get("usage")), Mapping)
    ]
    metrics.update(
        {
            f"{prefix}/judge/group_count": float(len(metadata_values)),
            f"{prefix}/judge/cache_hit_rate": _mean(
                bool(metadata.get("cached")) for metadata in metadata_values
            ),
            f"{prefix}/judge/latency_mean_seconds": _mean(
                float(metadata.get("elapsed_seconds") or 0.0)
                for metadata in metadata_values
            ),
            f"{prefix}/judge/latency_max_seconds": max(
                float(metadata.get("elapsed_seconds") or 0.0)
                for metadata in metadata_values
            ),
            f"{prefix}/judge/attempts_mean": _mean(
                float(metadata.get("attempts") or 0.0) for metadata in metadata_values
            ),
            f"{prefix}/judge/normalized_output_count": float(
                sum(
                    int(metadata.get("normalized_output_count") or 0)
                    for metadata in metadata_values
                )
            ),
            f"{prefix}/judge/prompt_tokens": float(
                sum(int(usage.get("prompt_tokens") or 0) for usage in usage_values)
            ),
            f"{prefix}/judge/completion_tokens": float(
                sum(int(usage.get("completion_tokens") or 0) for usage in usage_values)
            ),
            f"{prefix}/judge/total_tokens": float(
                sum(int(usage.get("total_tokens") or 0) for usage in usage_values)
            ),
        }
    )
    return metrics


class NaturalCoTRewardManager:
    """Bridge packed natural-CoT rewards into verl's token reward tensor."""

    def __init__(
        self,
        tokenizer: Any,
        scorer: NaturalCoTReward,
        expected_group_size: int,
        *,
        judge_enabled: bool = True,
        max_workers: int = 8,
        require_rollout_metadata: bool = True,
        num_examine: int = 0,
    ) -> None:
        if expected_group_size < 1:
            raise ValueError("expected_group_size must be positive")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if judge_enabled and not require_rollout_metadata:
            raise ValueError("Judge training requires stable rollout metadata")
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.expected_group_size = expected_group_size
        self.judge_enabled = judge_enabled
        self.max_workers = max_workers
        self.require_rollout_metadata = require_rollout_metadata
        self.num_examine = num_examine
        self.last_records: list[dict[str, Any]] = []

    @staticmethod
    def _target(item: Any) -> Mapping[str, Any]:
        reward_model = item.non_tensor_batch.get("reward_model")
        if not isinstance(reward_model, Mapping):
            raise TypeError("Every natural-CoT row needs reward_model metadata")
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError as exc:
                raise ValueError("reward_model.ground_truth is invalid JSON") from exc
        if not isinstance(ground_truth, Mapping):
            raise TypeError("reward_model.ground_truth must be a process target")
        return NaturalCoTRewardManager._to_builtin(ground_truth)

    @staticmethod
    def _to_builtin(value: Any) -> Any:
        """Normalize pandas/Arrow ndarray containers into JSON-compatible values."""
        if isinstance(value, Mapping):
            return {
                str(key): NaturalCoTRewardManager._to_builtin(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [NaturalCoTRewardManager._to_builtin(item) for item in value]
        if hasattr(value, "tolist"):
            return NaturalCoTRewardManager._to_builtin(value.tolist())
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        return value

    def _decode_entry(self, data: Any, index: int) -> _BatchEntry:
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

        judge_prompt = item.non_tensor_batch.get("judge_prompt")
        if not isinstance(judge_prompt, str) or not judge_prompt.strip():
            raise ValueError(
                "Every natural-CoT parquet row must preserve a non-empty judge_prompt"
            )
        extra_info = item.non_tensor_batch.get("extra_info", {})
        if not isinstance(extra_info, Mapping):
            extra_info = {}

        if self.require_rollout_metadata:
            uid = item.non_tensor_batch.get("uid")
            rollout_index = item.non_tensor_batch.get("rollout_index")
            if not isinstance(uid, str) or not uid:
                raise ValueError("Natural-CoT training batch is missing uid")
            if not isinstance(rollout_index, Integral):
                raise ValueError("Natural-CoT training batch is missing rollout_index")
            rollout_index = int(rollout_index)
        else:
            sample_id = extra_info.get("global_sample_id", index)
            uid = f"validation:{sample_id}:{index}"
            rollout_index = 0

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        eos_reached = bool(
            valid_response_length > 0
            and eos_token_id is not None
            and int(valid_response_ids[-1].item()) == int(eos_token_id)
        )
        return _BatchEntry(
            batch_index=index,
            uid=uid,
            rollout_index=rollout_index,
            judge_prompt=judge_prompt,
            target=self._target(item),
            response=response,
            valid_response_length=valid_response_length,
            generation_reached_eos=eos_reached,
            extra_info=dict(extra_info),
        )

    def _build_groups(
        self, entries: list[_BatchEntry]
    ) -> tuple[list[RewardGroup], dict[tuple[str, str], _BatchEntry]]:
        by_uid: defaultdict[str, list[_BatchEntry]] = defaultdict(list)
        for entry in entries:
            by_uid[entry.uid].append(entry)

        groups: list[RewardGroup] = []
        entry_by_candidate: dict[tuple[str, str], _BatchEntry] = {}
        for uid, members in by_uid.items():
            if len(members) != self.expected_group_size:
                raise ValueError(
                    f"Expected {self.expected_group_size} rollouts for uid={uid}, "
                    f"got {len(members)}"
                )
            members = sorted(members, key=lambda entry: entry.rollout_index)
            expected_indices = list(range(self.expected_group_size))
            actual_indices = [entry.rollout_index for entry in members]
            if actual_indices != expected_indices:
                raise ValueError(
                    f"uid={uid} rollout_index mismatch: expected={expected_indices}, "
                    f"actual={actual_indices}"
                )
            prompt = members[0].judge_prompt
            target = members[0].target
            if any(entry.judge_prompt != prompt for entry in members):
                raise ValueError(f"uid={uid} contains inconsistent judge_prompt values")
            target_json = canonical_json(target)
            if any(canonical_json(entry.target) != target_json for entry in members):
                raise ValueError(f"uid={uid} contains inconsistent process targets")

            candidate_ids = tuple(entry.candidate_id for entry in members)
            group = RewardGroup(
                group_id=uid,
                process_prompt=prompt,
                responses=tuple(entry.response for entry in members),
                target=target,
                candidate_ids=candidate_ids,
            )
            groups.append(group)
            for entry in members:
                key = (uid, entry.candidate_id)
                if key in entry_by_candidate:
                    raise ValueError(f"Duplicate natural-CoT candidate key: {key}")
                entry_by_candidate[key] = entry
        return groups, entry_by_candidate

    def preflight(self, check_remote: bool = True) -> dict[str, Any]:
        if not self.judge_enabled:
            return {"judge_enabled": False, "remote_checked": False}
        result = self.scorer.judge.preflight(check_remote=check_remote)
        return {"judge_enabled": True, **result}

    def judge_call_stats(self, reset: bool = False) -> dict[str, int | float]:
        return self.scorer.judge.stats_snapshot(reset=reset)

    def __call__(self, data: Any) -> torch.Tensor:
        entries = [self._decode_entry(data, index) for index in range(len(data))]
        groups, entry_by_candidate = self._build_groups(entries)
        if self.judge_enabled:
            group_results = self.scorer.score_groups_concurrently(
                groups, max_workers=min(self.max_workers, len(groups))
            )
        else:
            group_results = [
                self.scorer.score_group_rule_only(group) for group in groups
            ]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        records_by_index: dict[int, dict[str, Any]] = {}
        for result in group_results:
            uid = str(result["group_id"])
            judge_metadata = result.get("judge_metadata")
            for scored in result["records"]:
                candidate_id = str(scored["candidate_id"])
                key = (uid, candidate_id)
                if key not in entry_by_candidate:
                    raise ValueError(f"Judge returned an unknown candidate: {key}")
                entry = entry_by_candidate[key]
                reward = float(scored["combined"]["reward"])
                if entry.valid_response_length > 0:
                    reward_tensor[
                        entry.batch_index, entry.valid_response_length - 1
                    ] = reward
                records_by_index[entry.batch_index] = {
                    **dict(entry.extra_info),
                    "uid": entry.uid,
                    "rollout_index": entry.rollout_index,
                    "candidate_id": candidate_id,
                    "judge_prompt": entry.judge_prompt,
                    "response": entry.response,
                    "raw_response": entry.response,
                    "process_target": dict(entry.target),
                    "token_count": entry.valid_response_length,
                    "generation_reached_eos": entry.generation_reached_eos,
                    "rule": scored["rule"],
                    "judge": scored["judge"],
                    "combined": scored["combined"],
                    "judge_metadata": judge_metadata,
                }

        if len(records_by_index) != len(entries):
            missing = sorted(set(range(len(entries))) - set(records_by_index))
            raise ValueError(f"Natural-CoT rewards missing batch indices: {missing}")
        self.last_records = [records_by_index[index] for index in range(len(entries))]
        for record in self.last_records[: self.num_examine]:
            print(
                "[RobustToM natural-CoT response]\n"
                f"{record['response']}\n"
                f"[reward={record['combined']['reward']}]"
            )
        return reward_tensor

    def last_metrics(self, prefix: str = "reward") -> dict[str, float]:
        return summarize_natural_cot_records(self.last_records, prefix=prefix)

    @staticmethod
    def validation_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
        metrics = _natural_metrics(records, "val/overall")
        bucket_specs = {
            "source_dataset": "source_dataset",
            "question_order": "order",
            "intervention_type": "intervention",
        }
        for field, section in bucket_specs.items():
            values = sorted({str(record.get(field, "unknown")) for record in records})
            for value in values:
                subset = [
                    record
                    for record in records
                    if str(record.get(field, "unknown")) == value
                ]
                metrics.update(_natural_metrics(subset, f"val/{section}/{value}"))
        for field in ("shortcut_conflict", "last_mention_conflict"):
            subset = [record for record in records if bool(record.get(field, False))]
            if subset:
                metrics.update(_natural_metrics(subset, f"val/{field}"))
        return metrics
