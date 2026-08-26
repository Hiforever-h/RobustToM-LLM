"""Metric aggregation for v3 process rewards and GRPO groups."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def summarize_reward_records(
    records: Iterable[dict[str, Any]], prefix: str = "reward"
) -> dict[str, float]:
    records = list(records)
    if not records:
        return {}

    def rate(check: str) -> float:
        return sum(bool(row["result"]["checks"].get(check, False)) for row in records) / len(records)

    metrics = {
        f"{prefix}/mean": sum(float(row["result"]["reward"]) for row in records) / len(records),
        f"{prefix}/parse_rate": rate("parseable_json"),
        f"{prefix}/strict_format_rate": rate("format"),
        f"{prefix}/tom_order_accuracy": rate("tom_order"),
        f"{prefix}/belief_chain_accuracy": rate("belief_chain"),
        f"{prefix}/object_accuracy": rate("object"),
        f"{prefix}/belief_trace_accuracy": rate("belief_trace"),
        f"{prefix}/answer_accuracy": rate("answer"),
        f"{prefix}/full_reward_rate": sum(
            float(row["result"]["reward"]) == 1.0 for row in records
        )
        / len(records),
        f"{prefix}/eos_rate": sum(bool(row.get("generation_reached_eos")) for row in records)
        / len(records),
    }
    trace_steps = [
        bool(step)
        for row in records
        for step in row["result"]["checks"].get("belief_trace_steps", [])
    ]
    metrics[f"{prefix}/belief_trace_step_accuracy"] = (
        sum(trace_steps) / len(trace_steps) if trace_steps else 0.0
    )
    for component in ("format", "tom_order", "belief_chain", "object", "belief_trace", "answer"):
        metrics[f"{prefix}/component/{component}"] = sum(
            float(row["result"]["components"].get(component, 0.0)) for row in records
        ) / len(records)
    return metrics


def summarize_group_rewards(
    rewards: Iterable[float], group_ids: Iterable[str], expected_group_size: int
) -> dict[str, float]:
    groups: defaultdict[str, list[float]] = defaultdict(list)
    for reward, group_id in zip(rewards, group_ids):
        groups[str(group_id)].append(float(reward))
    if not groups:
        return {}
    sizes = [len(values) for values in groups.values()]
    if any(size != expected_group_size for size in sizes):
        raise ValueError(
            f"Expected {expected_group_size} rollouts per prompt, got group sizes {sorted(set(sizes))}"
        )
    stds = []
    for values in groups.values():
        mean = sum(values) / len(values)
        stds.append((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)
    return {
        "grpo/group_count": float(len(groups)),
        "grpo/group_reward_std_mean": sum(stds) / len(stds),
        "grpo/zero_variance_group_rate": sum(std <= 1e-12 for std in stds) / len(stds),
    }

