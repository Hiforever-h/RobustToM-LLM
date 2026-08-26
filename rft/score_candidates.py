#!/usr/bin/env python3
"""Score natural-CoT candidates with the shared deterministic rules and Judge."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rft.common import canonical_json, read_jsonl, sha256_file, write_jsonl
from scripts.reward import (
    DeepSeekJudge,
    JudgeConfig,
    NaturalCoTReward,
    RewardGroup,
    score_rule_only_group,
)

DEFAULT_MIN_REWARD = 0.88
DEFAULT_MIN_REASONING_SCORE = 0.5


def _response(candidate: dict[str, Any]) -> str:
    for key in ("raw_response", "response", "accepted_response"):
        value = candidate.get(key)
        if isinstance(value, str):
            return value
    raise ValueError(
        "Candidate has no response field: "
        f"{candidate.get('candidate_id', '<unknown>')}"
    )


def _as_target(value: Any, sample: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"Missing process_target for candidate {sample}")
    return value


def score_candidates(
    candidate_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]] | None = None,
    *,
    scorer: NaturalCoTReward | Any | None = None,
    rule_only: bool = False,
    max_workers: int = 8,
    min_reward: float = DEFAULT_MIN_REWARD,
    min_reasoning_score: float = DEFAULT_MIN_REASONING_SCORE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pack candidates by prompt, score them, and apply the RFT acceptance rule.

    Deterministic correctness and EOS are hard gates. Judge-backed reward and
    every effective per-step reasoning score must meet configurable thresholds.
    ``rule_only`` is diagnostic and therefore cannot accept a trajectory.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if not 0.0 <= min_reward <= 1.0:
        raise ValueError("min_reward must be within [0, 1]")
    if not 0.0 <= min_reasoning_score <= 1.0:
        raise ValueError("min_reasoning_score must be within [0, 1]")
    if scorer is not None and rule_only:
        raise ValueError("scorer and rule_only are mutually exclusive")

    source_by_sample: dict[str, dict[str, Any]] = {}
    for row in source_rows or []:
        sample = row.get("global_sample_id")
        if not isinstance(sample, str) or not sample:
            raise ValueError("Every source row must contain global_sample_id")
        if sample in source_by_sample:
            raise ValueError(f"Duplicate source sample: {sample}")
        source_by_sample[sample] = row

    prepared: list[dict[str, Any]] = []
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, candidate in enumerate(candidate_rows):
        sample = candidate.get("global_sample_id")
        if not isinstance(sample, str) or not sample:
            raise ValueError("Every candidate must contain global_sample_id")
        if source_rows is not None and sample not in source_by_sample:
            raise ValueError(
                f"Candidate sample is absent from the supplied data split: {sample}"
            )
        source = source_by_sample.get(sample, {})
        candidate_target = candidate.get("process_target")
        target = _as_target(source.get("process_target") or candidate_target, sample)
        if candidate_target is not None and canonical_json(
            _as_target(candidate_target, sample)
        ) != canonical_json(target):
            raise ValueError(f"Candidate target disagrees with source data: {sample}")

        actor_prompt = source.get("process_prompt") or candidate.get("process_prompt")
        judge_prompt = source.get("judge_prompt") or candidate.get("judge_prompt")
        if not isinstance(actor_prompt, str) or not actor_prompt.strip():
            raise ValueError(f"Missing process_prompt for candidate {sample}")
        if not isinstance(judge_prompt, str) or not judge_prompt.strip():
            raise ValueError(f"Missing judge_prompt for candidate {sample}")
        if isinstance(candidate.get("process_prompt"), str) and source:
            if candidate["process_prompt"] != actor_prompt:
                raise ValueError(f"Candidate process_prompt disagrees with data: {sample}")
        if isinstance(candidate.get("judge_prompt"), str) and source:
            if candidate["judge_prompt"] != judge_prompt:
                raise ValueError(f"Candidate judge_prompt disagrees with data: {sample}")

        item = {
            "index": index,
            "candidate": candidate,
            "source": source,
            "sample": sample,
            "target": target,
            "actor_prompt": actor_prompt,
            "judge_prompt": judge_prompt,
            "response": _response(candidate),
            # Judge IDs only need to be stable and unique within a packed group.
            "judge_candidate_id": f"c{index:06d}",
        }
        prepared.append(item)
        grouped[sample].append(item)

    groups: list[RewardGroup] = []
    for sample, members in grouped.items():
        first_target = canonical_json(members[0]["target"])
        if any(canonical_json(member["target"]) != first_target for member in members):
            raise ValueError(f"Packed candidates disagree on process_target: {sample}")
        if any(member["judge_prompt"] != members[0]["judge_prompt"] for member in members):
            raise ValueError(f"Packed candidates disagree on judge_prompt: {sample}")
        groups.append(
            RewardGroup(
                group_id=sample,
                process_prompt=members[0]["judge_prompt"],
                responses=tuple(member["response"] for member in members),
                target=members[0]["target"],
                candidate_ids=tuple(
                    member["judge_candidate_id"] for member in members
                ),
            )
        )
    if rule_only:
        group_results = [score_rule_only_group(group) for group in groups]
    else:
        scorer = scorer or NaturalCoTReward(DeepSeekJudge())
        group_results = scorer.score_groups_concurrently(
            groups, max_workers=min(max_workers, len(groups)) if groups else 1
        )

    score_by_candidate: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for group_result in group_results:
        for record in group_result["records"]:
            score_by_candidate[record["candidate_id"]] = (record, group_result)

    scored: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for item in prepared:
        record, group_result = score_by_candidate[item["judge_candidate_id"]]
        combined = record["combined"]
        rule = record["rule"]
        structure_ok = bool(rule["parsed"]["checks"]["structure_ok"])
        all_states_correct = bool(rule["all_states_correct"])
        answer_correct = bool(rule["answer_correct"])
        effective_reasoning_scores = [
            float(value) for value in combined["effective_reasoning_scores"]
        ]
        reasoning_threshold_ok = bool(effective_reasoning_scores) and all(
            score >= min_reasoning_score for score in effective_reasoning_scores
        )
        reward_threshold_ok = float(combined["reward"]) >= min_reward
        reached_eos = item["candidate"].get("generation_reached_eos", False)
        accepted = (
            structure_ok
            and all_states_correct
            and answer_correct
            and reasoning_threshold_ok
            and reward_threshold_ok
            and reached_eos is True
            and not rule_only
        )
        counters["full_reward"] += int(combined["reward"] == 1.0)
        counters["reward_threshold"] += int(reward_threshold_ok)
        counters["reasoning_threshold"] += int(reasoning_threshold_ok)
        counters["valid_structure"] += int(structure_ok)
        counters["all_states_correct"] += int(all_states_correct)
        counters["answer_correct"] += int(answer_correct)
        counters["eos"] += int(reached_eos is True)
        counters["accepted"] += int(accepted)

        enriched = dict(item["candidate"])
        enriched.update(
            {
                "raw_response": item["response"],
                "process_prompt": item["actor_prompt"],
                "judge_prompt": item["judge_prompt"],
                "process_target": item["target"],
                "score": combined,
                "rule_score": rule,
                "judge_score": record["judge"],
                "judge_metadata": group_result.get("judge_metadata"),
                "accepted": accepted,
            }
        )
        if accepted:
            enriched["acceptance_reason"] = "quality_thresholds_and_hard_gates_met"
        elif rule_only:
            enriched["acceptance_reason"] = "judge_disabled"
        elif not structure_ok:
            enriched["acceptance_reason"] = "invalid_response_structure"
        elif not all_states_correct:
            enriched["acceptance_reason"] = "state_incorrect"
        elif not answer_correct:
            enriched["acceptance_reason"] = "answer_incorrect"
        elif not reasoning_threshold_ok:
            enriched["acceptance_reason"] = "reasoning_below_threshold"
        elif not reward_threshold_ok:
            enriched["acceptance_reason"] = "reward_below_threshold"
        else:
            enriched["acceptance_reason"] = "generation_not_stopped"

        source = item["source"]
        if source:
            for key in (
                "source_dataset",
                "question_order",
                "intervention_type",
                "shortcut_conflict",
                "last_mention_conflict",
                "shortcut_prediction",
                "last_mentioned_container",
                "global_pair_id",
                "process_prompt_version",
            ):
                enriched[key] = source.get(key)
        scored.append(enriched)

    accepted_samples = {row["global_sample_id"] for row in scored if row["accepted"]}
    coverage_rows = source_rows if source_rows is not None else list(
        {row["global_sample_id"]: row for row in scored}.values()
    )
    accepted_pair_sides: defaultdict[str, set[str]] = defaultdict(set)
    for row in scored:
        if row["accepted"] and isinstance(row.get("global_pair_id"), str):
            accepted_pair_sides[row["global_pair_id"]].add(
                str(row.get("intervention_type"))
            )
    source_pairs = {
        row.get("global_pair_id")
        for row in coverage_rows
        if isinstance(row.get("global_pair_id"), str)
    }
    complete_pairs = sum(
        sides == {"observed", "hidden"}
        for sides in accepted_pair_sides.values()
    )

    bucket_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in coverage_rows:
        key = "|".join(
            (
                str(row.get("source_dataset", "unknown")),
                f"order={row.get('question_order', 'unknown')}",
                str(row.get("intervention_type", "unknown")),
            )
        )
        bucket_counts[key]["prompts"] += 1
        bucket_counts[key]["accepted_prompts"] += int(
            row.get("global_sample_id") in accepted_samples
        )

    manifest = {
        "reward_backend": "scripts.reward.NaturalCoTReward",
        "rule_only": rule_only,
        "acceptance_policy": {
            "min_reward": min_reward,
            "min_reasoning_score": min_reasoning_score,
            "require_valid_structure": True,
            "require_all_states_correct": True,
            "require_answer_correct": True,
            "require_eos": True,
        },
        "candidate_count": len(scored),
        "judge_group_count": len(groups) if not rule_only else 0,
        "accepted_count": counters["accepted"],
        "full_reward_count": counters["full_reward"],
        "reward_threshold_count": counters["reward_threshold"],
        "reasoning_threshold_count": counters["reasoning_threshold"],
        "valid_structure_count": counters["valid_structure"],
        "all_states_correct_count": counters["all_states_correct"],
        "answer_correct_count": counters["answer_correct"],
        "eos_count": counters["eos"],
        "acceptance_rate": counters["accepted"] / len(scored) if scored else 0.0,
        "prompt_count": len(coverage_rows),
        "accepted_prompt_count": len(accepted_samples),
        "prompt_coverage": (
            len(accepted_samples) / len(coverage_rows) if coverage_rows else 0.0
        ),
        "pair_count": len(source_pairs),
        "complete_pair_count": complete_pairs,
        "complete_pair_coverage": (
            complete_pairs / len(source_pairs) if source_pairs else 0.0
        ),
        "coverage_by_source_order_intervention": {
            key: {
                "prompts": counts["prompts"],
                "accepted_prompts": counts["accepted_prompts"],
                "coverage": counts["accepted_prompts"] / counts["prompts"],
            }
            for key, counts in sorted(bucket_counts.items())
        },
        "source_candidate_sha256": None,
    }
    return scored, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, help="JSONL source data used to resolve prompt metadata"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--rule-only",
        action="store_true",
        help="Run deterministic diagnostics without Judge calls; accepts nothing",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--min-reward", type=float, default=DEFAULT_MIN_REWARD)
    parser.add_argument(
        "--min-reasoning-score",
        type=float,
        default=DEFAULT_MIN_REASONING_SCORE,
        help="Minimum effective Judge score required for every Think step",
    )
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
    scorer = None
    if not args.rule_only:
        judge = DeepSeekJudge(
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
        scorer = NaturalCoTReward(judge)
    rows, manifest = score_candidates(
        read_jsonl(args.candidates),
        read_jsonl(args.data) if args.data else None,
        scorer=scorer,
        rule_only=args.rule_only,
        max_workers=args.max_workers,
        min_reward=args.min_reward,
        min_reasoning_score=args.min_reasoning_score,
    )
    manifest["source_candidate_sha256"] = sha256_file(args.candidates)
    if scorer is not None and hasattr(scorer.judge, "stats_snapshot"):
        manifest["judge_call_statistics"] = scorer.judge.stats_snapshot()
    write_jsonl(args.output, rows)
    manifest_path = args.manifest or args.output.with_name("acceptance_metrics.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
