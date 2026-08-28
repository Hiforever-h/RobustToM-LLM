#!/usr/bin/env python3
"""Evaluate natural-CoT RFT responses, with an optional external Judge pass."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from rft.common import read_jsonl
from rft.prompt import (
    COMPACT_NATURAL_COT_PROMPT_VERSION,
    NATURAL_COT_PROMPT_VERSION,
    compact_process_record,
)
from rft.reward import parse_prediction, score_process_output
from scripts.reward import (
    ANSWER_RE,
    DeepSeekJudge,
    JudgeConfig,
    NaturalCoTReward,
    combine_reward,
    normalize,
    score_rule_components,
)

NATURAL_PROMPT_VERSIONS = frozenset(
    {
        "natural-cot-think-state-v1",
        NATURAL_COT_PROMPT_VERSION,
        COMPACT_NATURAL_COT_PROMPT_VERSION,
    }
)


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _response(row: dict[str, Any]) -> str:
    value = row.get("response") or row.get("raw_response") or row.get(
        "accepted_response"
    )
    return value if isinstance(value, str) else ""


def _answer_from_response(response: str) -> str | None:
    natural_answers = []
    for line in response.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = ANSWER_RE.fullmatch(line)
        if match and match.group(1).strip():
            natural_answers.append(match.group(1).strip())
    if len(natural_answers) == 1:
        return normalize(natural_answers[0])
    prediction, _ = parse_prediction(response)
    if prediction and isinstance(prediction.get("answer"), str):
        return normalize(prediction["answer"])
    return None


def _unique_rows_by_sample(
    rows: Iterable[dict[str, Any]], source_name: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("global_sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Missing global_sample_id in {source_name}")
        if sample_id in indexed:
            raise ValueError(f"Duplicate global_sample_id in {source_name}: {sample_id}")
        indexed[sample_id] = row
    return indexed


def evaluate_answer_predictions(
    prediction_rows: list[dict[str, Any]], data_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score final answers from either natural ``Answer:`` text or legacy JSON."""
    predictions = _unique_rows_by_sample(prediction_rows, "predictions")
    data = _unique_rows_by_sample(data_rows, "data")
    prediction_ids = set(predictions)
    data_ids = set(data)
    if prediction_ids != data_ids:
        missing = sorted(data_ids - prediction_ids)
        unexpected = sorted(prediction_ids - data_ids)
        raise ValueError(
            "Prediction/data sample IDs differ: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    correct_count = 0
    for sample_id, gold_row in data.items():
        gold_answer = gold_row.get("gold_answer") or gold_row.get("answer")
        if not isinstance(gold_answer, str) or not gold_answer.strip():
            raise ValueError(f"Missing gold answer in data: {sample_id}")
        predicted_answer = _answer_from_response(_response(predictions[sample_id]))
        if predicted_answer is not None and predicted_answer == normalize(gold_answer):
            correct_count += 1

    count = len(data)
    return {
        "overall": {
            "count": count,
            "correct_count": correct_count,
            "answer_accuracy": correct_count / count if count else 0.0,
        }
    }


def _is_natural_row(row: dict[str, Any], response: str) -> bool:
    if row.get("process_prompt_version") in NATURAL_PROMPT_VERSIONS:
        return True
    return any(line.lower().startswith("think ") for line in response.splitlines())


def _natural_score(
    row: dict[str, Any], response: str, target: dict[str, Any]
) -> dict[str, Any]:
    rule = row.get("rule_score")
    if not isinstance(rule, dict):
        rule = score_rule_components(response, target)
    stored = row.get("score")
    judged = isinstance(row.get("judge_score"), dict)
    if isinstance(stored, dict) and isinstance(stored.get("reward"), (int, float)):
        combined = stored
    else:
        combined = combine_reward(rule, [0.0] * len(rule["state_correct"]))
    parsed = rule["parsed"]
    checks = parsed["checks"]
    answer = normalize(rule["answer"]) if isinstance(rule.get("answer"), str) else None
    return {
        "reward": float(combined["reward"]),
        "judged": judged,
        "parseable": bool(parsed["steps"] or parsed.get("answer")),
        "format": bool(checks["structure_ok"]),
        "reasoning_present": all(bool(value) for value in rule["reasoning_present"]),
        "state_accuracy": float(rule["state_accuracy"]),
        "core_state_correct": bool(rule["all_states_correct"]),
        "answer": answer,
        "answer_correct": bool(rule["answer_correct"]),
        "consistent": bool(rule["final_state_answer_consistent"]),
    }


def _legacy_score(response: str, target: dict[str, Any]) -> dict[str, Any]:
    result = score_process_output(response, target)
    prediction, _ = parse_prediction(response)
    checks = result["checks"]
    mode = target.get("reasoning_mode")
    if mode == "world_state":
        core_state = checks.get("world_state", False)
        state_value = normalize(prediction.get("world_state", "")) if prediction else None
    elif mode == "nested_belief":
        core_state = checks.get("belief_trace", False)
        trace = prediction.get("belief_trace") if prediction else None
        state_value = (
            normalize(trace[-1]["location"])
            if isinstance(trace, list)
            and trace
            and isinstance(trace[-1], dict)
            and isinstance(trace[-1].get("location"), str)
            else None
        )
    else:
        core_state = checks.get("final_move_observed", False) and checks.get(
            "nested_belief", False
        )
        state_value = (
            normalize(prediction.get("nested_belief", "")) if prediction else None
        )
    answer = (
        normalize(prediction["answer"])
        if prediction and isinstance(prediction.get("answer"), str)
        else None
    )
    return {
        "reward": float(result["reward"]),
        "judged": True,
        "parseable": bool(checks.get("parseable_json", False)),
        "format": bool(checks.get("format", False)),
        "reasoning_present": True,
        "state_accuracy": float(bool(core_state)),
        "core_state_correct": bool(core_state),
        "answer": answer,
        "answer_correct": bool(checks.get("answer", False)),
        "consistent": bool(answer is not None and state_value == answer),
    }


def _metric_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    lengths: list[int] = []
    for row in rows:
        response = _response(row)
        target = row.get("process_target")
        if isinstance(target, str):
            target = json.loads(target)
        if not isinstance(target, dict):
            raise ValueError(
                f"Missing process_target: {row.get('global_sample_id', '<unknown>')}"
            )
        if isinstance(row.get("token_count"), int):
            lengths.append(row["token_count"])
        item = (
            _natural_score(row, response, target)
            if _is_natural_row(row, response)
            else _legacy_score(response, target)
        )
        item["row"] = row
        item["target_answer"] = normalize(target.get("answer", ""))
        scored.append(item)

    def rate(predicate: Any) -> float:
        return (
            sum(bool(predicate(item)) for item in scored) / len(scored)
            if scored
            else 0.0
        )

    pair_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        pair_groups[str(item["row"].get("global_pair_id"))].append(item)
    complete_pairs = [group for group in pair_groups.values() if len(group) == 2]
    pair_accuracy = (
        sum(all(item["answer_correct"] for item in group) for group in complete_pairs)
        / len(complete_pairs)
        if complete_pairs
        else 0.0
    )
    sensitivity_groups = [
        group
        for group in complete_pairs
        if len({item["target_answer"] for item in group}) > 1
    ]
    intervention_sensitivity = (
        sum(
            len({item["answer"] for item in group}) == 2
            and all(item["answer"] is not None for item in group)
            for group in sensitivity_groups
        )
        / len(sensitivity_groups)
        if sensitivity_groups
        else 0.0
    )

    shortcut_rows = [
        item
        for item in scored
        if item["row"].get("shortcut_conflict")
        and item["row"].get("shortcut_prediction") is not None
    ]
    shortcut_copy_rate = (
        sum(
            item["answer"] == normalize(item["row"]["shortcut_prediction"])
            for item in shortcut_rows
        )
        / len(shortcut_rows)
        if shortcut_rows
        else 0.0
    )
    last_rows = [
        item
        for item in scored
        if item["row"].get("last_mention_conflict")
        and item["row"].get("last_mentioned_container") is not None
    ]
    last_copy_rate = (
        sum(
            item["answer"] == normalize(item["row"]["last_mentioned_container"])
            for item in last_rows
        )
        / len(last_rows)
        if last_rows
        else 0.0
    )
    return {
        "count": len(scored),
        "parse_rate": rate(lambda item: item["parseable"]),
        "strict_format_rate": rate(lambda item: item["format"]),
        "reasoning_present_rate": rate(lambda item: item["reasoning_present"]),
        "mean_process_reward": (
            sum(item["reward"] for item in scored) / len(scored) if scored else 0.0
        ),
        "full_reward_rate": rate(lambda item: item["reward"] == 1.0),
        "judge_scored_count": sum(bool(item["judged"]) for item in scored),
        "answer_accuracy": rate(lambda item: item["answer_correct"]),
        "core_state_accuracy": rate(lambda item: item["core_state_correct"]),
        "mean_state_accuracy": (
            sum(item["state_accuracy"] for item in scored) / len(scored)
            if scored
            else 0.0
        ),
        "pair_accuracy": pair_accuracy,
        "intervention_sensitivity": intervention_sensitivity,
        "shortcut_copy_rate": shortcut_copy_rate,
        "last_mention_copy_rate": last_copy_rate,
        "answer_state_consistency": rate(lambda item: item["consistent"]),
        "eos_rate": rate(
            lambda item: item["row"].get("generation_reached_eos", False) is True
        ),
        "length_p95": _percentile(lengths, 0.95),
        "group_count": len(complete_pairs),
        "sensitivity_pair_count": len(sensitivity_groups),
    }


def evaluate_predictions(
    prediction_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if data_rows:
        by_sample = {row.get("global_sample_id"): row for row in data_rows}
        merged = []
        for prediction in prediction_rows:
            base = dict(by_sample.get(prediction.get("global_sample_id"), {}))
            base.update(prediction)
            merged.append(base)
    else:
        merged = prediction_rows
    metrics = {"overall": _metric_rows(merged)}
    dimensions = {
        "source_dataset": lambda row: row.get("source_dataset", "unknown"),
        "question_order": lambda row: str(row.get("question_order", "unknown")),
        "intervention_type": lambda row: row.get("intervention_type", "unknown"),
    }
    for name, key_fn in dimensions.items():
        buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in merged:
            buckets[str(key_fn(row))].append(row)
        metrics[name] = {
            key: _metric_rows(rows) for key, rows in sorted(buckets.items())
        }
    for name, flag in (
        ("shortcut_conflict", "shortcut_conflict"),
        ("last_mention_conflict", "last_mention_conflict"),
    ):
        metrics[name] = _metric_rows([row for row in merged if row.get(flag)])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Score only the final Answer marker (or legacy JSON answer)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Call scripts/reward.py's external Judge before computing metrics",
    )
    parser.add_argument(
        "--compact-prompt",
        action="store_true",
        help=(
            "Rebuild --data prompts with the compact protocol so optional Judge "
            "scoring matches compact-prompt predictions"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument("--judge-max-tokens", type=int, default=3000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    data = read_jsonl(args.data) if args.data else None
    if args.compact_prompt:
        if data is None:
            raise ValueError("--compact-prompt requires --data")
        data = [compact_process_record(row) for row in data]
    if args.answer_only:
        if data is None:
            raise ValueError("--answer-only requires --data with answer fields")
        if args.judge:
            raise ValueError("--answer-only and --judge are mutually exclusive")
        metrics = evaluate_answer_predictions(predictions, data)
    else:
        if args.judge:
            if data is None:
                raise ValueError("--judge requires --data with judge_prompt fields")
            from rft.score_candidates import score_candidates

            scorer = NaturalCoTReward(
                DeepSeekJudge(
                    JudgeConfig(
                        base_url=args.base_url,
                        model=args.judge_model,
                        timeout_seconds=args.timeout,
                        max_tokens=args.judge_max_tokens,
                        retries=args.retries,
                        thinking=args.thinking,
                        cache_dir=args.cache_dir,
                    )
                )
            )
            predictions, _ = score_candidates(
                predictions,
                data,
                scorer=scorer,
                max_workers=args.max_workers,
            )
        metrics = evaluate_predictions(predictions, data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["overall"], indent=2))


if __name__ == "__main__":
    main()
