#!/usr/bin/env python3
"""Create the fixed RFT train/dev/test split without modifying source JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rft.common import pair_id, read_jsonl, sha256_file, stable_hash, write_jsonl
from rft.reward import score_process_output


def validate_pairs(records_by_split: dict[str, list[dict[str, Any]]]) -> None:
    seen_samples: set[str] = set()
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, rows in records_by_split.items():
        for row in rows:
            sample = row.get("global_sample_id")
            if not isinstance(sample, str) or not sample:
                raise ValueError(f"Missing global_sample_id in {split}")
            if sample in seen_samples:
                raise ValueError(f"Duplicate global_sample_id: {sample}")
            seen_samples.add(sample)
            if row.get("split") != split:
                raise ValueError(f"Incorrect split field for {sample}: {row.get('split')!r}")
            target = row.get("process_target")
            if not isinstance(target, dict):
                raise ValueError(f"Missing process_target: {sample}")
            canonical = row.get("process_response")
            if not isinstance(canonical, str) or json.loads(canonical) != target:
                raise ValueError(f"process_response/target mismatch: {sample}")
            if score_process_output(canonical, target)["reward"] != 1.0:
                raise ValueError(f"Canonical target does not receive full reward: {sample}")
            groups[pair_id(row)].append(row)

    for current_pair, rows in groups.items():
        if len(rows) != 2 or {row.get("intervention_type") for row in rows} != {"observed", "hidden"}:
            raise ValueError(f"Incomplete pair: {current_pair}")
        if len({row.get("split") for row in rows}) != 1:
            raise ValueError(f"Pair crosses splits: {current_pair}")
        observed = next(row for row in rows if row["intervention_type"] == "observed")
        hidden = next(row for row in rows if row["intervention_type"] == "hidden")
        for key in ("tom_order", "belief_chain", "object", "reasoning_mode"):
            if observed["process_target"][key] != hidden["process_target"][key]:
                raise ValueError(f"Pair target mismatch for {current_pair}: {key}")

    scenario_splits: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for split, rows in records_by_split.items():
        for row in rows:
            source = str(row.get("source_dataset"))
            if source == "exploretom":
                scenario = row.get("base_scenario_id")
            else:
                scenario = row.get("source_group_id")
            if scenario is None:
                raise ValueError(f"Missing source scenario ID: {row.get('global_sample_id')}")
            scenario_splits[(source, str(scenario))].add(split)
    leaked = [key for key, splits in scenario_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Source scenarios cross splits, first violations: {leaked[:5]}")


def derive_split(input_dir: Path, output_dir: Path, seed: int, holdout_pairs: int) -> dict[str, Any]:
    if holdout_pairs < 0:
        raise ValueError("holdout_pairs must be non-negative")
    source = {split: read_jsonl(input_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
    validate_pairs(source)

    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source["train"]:
        if row.get("source_dataset") == "exploretom":
            eligible[pair_id(row)].append(row)
    if holdout_pairs > len(eligible):
        raise ValueError(f"Requested {holdout_pairs} ExploreToM pairs, only {len(eligible)} available")
    ordered_pairs = sorted(
        eligible,
        key=lambda value: stable_hash(seed, value, eligible[value][0].get("base_scenario_id")),
    )
    holdout = set(ordered_pairs[:holdout_pairs])

    derived_train = [dict(row, split="train") for row in source["train"] if pair_id(row) not in holdout]
    derived_dev = [dict(row, split="dev") for row in source["val"]]
    derived_dev.extend(dict(row, split="dev") for row in source["train"] if pair_id(row) in holdout)
    derived_test = [dict(row, split="test") for row in source["test"]]
    derived = {"train": derived_train, "dev": derived_dev, "test": derived_test}
    validate_pairs(derived)

    output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in derived.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    all_rows = [row for rows in derived.values() for row in rows]
    process_target_versions = {
        str(row.get("process_target_version", "1.0")) for row in all_rows
    }
    if len(process_target_versions) != 1:
        raise ValueError(
            f"Mixed process target versions: {sorted(process_target_versions)}"
        )
    process_target_version = next(iter(process_target_versions))
    manifest = {
        "name": "RobustToM-RL RFT derived split",
        "seed": seed,
        "holdout_exploretom_pairs": sorted(holdout),
        "input_files": {
            split: {"path": str(input_dir / f"{split}.jsonl"), "sha256": sha256_file(input_dir / f"{split}.jsonl")}
            for split in source
        },
        "split_counts": {split: len(rows) for split, rows in derived.items()},
        "pair_counts": {split: len({pair_id(row) for row in rows}) for split, rows in derived.items()},
        "source_counts": dict(Counter(row.get("source_dataset") for row in all_rows)),
        "order_counts": dict(Counter(str(row.get("question_order")) for row in all_rows)),
        "intervention_counts": dict(Counter(row.get("intervention_type") for row in all_rows)),
        "output_sha256": {
            split: sha256_file(output_dir / f"{split}.jsonl") for split in derived
        },
        "process_target_version": process_target_version,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/counterfactual_process_reward"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/rft/derived"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--holdout-pairs", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(derive_split(args.input_dir, args.output_dir, args.seed, args.holdout_pairs), indent=2))


if __name__ == "__main__":
    main()
