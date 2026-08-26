#!/usr/bin/env python3
"""Score raw model candidates and apply the first-version RFT acceptance rule."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rft.common import canonical_json, read_jsonl, sha256_file, write_jsonl
from rft.reward import score_process_output


def _response(candidate: dict[str, Any]) -> str:
    for key in ("raw_response", "response", "accepted_response"):
        value = candidate.get(key)
        if isinstance(value, str):
            return value
    raise ValueError(f"Candidate has no response field: {candidate.get('candidate_id', '<unknown>')}")


def score_candidates(
    candidate_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_by_sample = {row.get("global_sample_id"): row for row in source_rows or []}
    scored: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for candidate in candidate_rows:
        sample = candidate.get("global_sample_id")
        if not isinstance(sample, str) or not sample:
            raise ValueError("Every candidate must contain global_sample_id")
        if source_rows is not None and sample not in source_by_sample:
            raise ValueError(f"Candidate sample is absent from the supplied data split: {sample}")
        source = source_by_sample.get(sample, {})
        candidate_target = candidate.get("process_target")
        target = source.get("process_target") or candidate_target
        if isinstance(target, str):
            target = json.loads(target)
        if not isinstance(target, dict):
            raise ValueError(f"Missing process_target for candidate {candidate.get('candidate_id', sample)}")
        if candidate_target is not None:
            if isinstance(candidate_target, str):
                candidate_target = json.loads(candidate_target)
            if canonical_json(candidate_target) != canonical_json(target):
                raise ValueError(f"Candidate target disagrees with source data: {sample}")
        response = _response(candidate)
        result = score_process_output(response, target)
        reached_eos = candidate.get("generation_reached_eos", False)
        strict = bool(result.get("checks", {}).get("format", False))
        accepted = result["reward"] == 1.0 and strict and reached_eos is True
        if result["reward"] == 1.0:
            counters["full_reward"] += 1
        if strict:
            counters["strict_format"] += 1
        if reached_eos is True:
            counters["eos"] += 1
        counters["accepted"] += int(accepted)
        enriched = dict(candidate)
        enriched["raw_response"] = response
        enriched["process_target"] = target
        enriched["score"] = result
        enriched["accepted"] = accepted
        enriched["acceptance_reason"] = (
            "full_reward_strict_format_eos"
            if accepted
            else "full_reward_missing"
            if result["reward"] != 1.0
            else "strict_format_missing"
            if not strict
            else "generation_not_stopped"
        )
        if source:
            for key in (
                "process_prompt",
                "source_dataset",
                "question_order",
                "intervention_type",
                "shortcut_conflict",
                "last_mention_conflict",
                "shortcut_prediction",
                "last_mentioned_container",
                "global_pair_id",
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
            accepted_pair_sides[row["global_pair_id"]].add(str(row.get("intervention_type")))
    source_pairs = {row.get("global_pair_id") for row in coverage_rows}
    complete_pairs = sum(sides == {"observed", "hidden"} for sides in accepted_pair_sides.values())

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
        bucket_counts[key]["accepted_prompts"] += int(row.get("global_sample_id") in accepted_samples)

    manifest = {
        "candidate_count": len(scored),
        "accepted_count": counters["accepted"],
        "full_reward_count": counters["full_reward"],
        "strict_format_count": counters["strict_format"],
        "eos_count": counters["eos"],
        "acceptance_rate": counters["accepted"] / len(scored) if scored else 0.0,
        "prompt_count": len(coverage_rows),
        "accepted_prompt_count": len(accepted_samples),
        "prompt_coverage": len(accepted_samples) / len(coverage_rows) if coverage_rows else 0.0,
        "pair_count": len(source_pairs),
        "complete_pair_count": complete_pairs,
        "complete_pair_coverage": complete_pairs / len(source_pairs) if source_pairs else 0.0,
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
    parser.add_argument("--data", type=Path, help="Optional JSONL source data used to fill candidate metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, manifest = score_candidates(
        read_jsonl(args.candidates), read_jsonl(args.data) if args.data else None
    )
    manifest["source_candidate_sha256"] = sha256_file(args.candidates)
    write_jsonl(args.output, rows)
    manifest_path = args.manifest or args.output.with_name("acceptance_metrics.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
