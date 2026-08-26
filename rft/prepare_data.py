#!/usr/bin/env python3
"""Create the fixed natural-CoT RFT train/dev/test split.

The source dataset already has stable train/val/test membership. RFT keeps that
membership and only renames ``val`` to ``dev``. No canonical
``process_response`` is copied or fabricated.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rft.common import pair_id, read_jsonl, sha256_file, write_jsonl
from scripts.reward import normalize, score_rule_components

PROMPT_VERSION = "natural-cot-think-state-v1"
SOURCE_TO_RFT_SPLIT = {"train": "train", "val": "dev", "test": "test"}


def _validate_row(row: dict[str, Any], split: str) -> None:
    sample = row.get("global_sample_id")
    if not isinstance(sample, str) or not sample:
        raise ValueError(f"Missing global_sample_id in {split}")
    if row.get("split") != split:
        raise ValueError(f"Incorrect split field for {sample}: {row.get('split')!r}")
    if "process_response" in row:
        raise ValueError(f"Natural-CoT row contains process_response: {sample}")
    if row.get("process_prompt_version") != PROMPT_VERSION:
        raise ValueError(f"Unexpected process_prompt_version for {sample}")
    if row.get("process_target_version") != "2.0":
        raise ValueError(f"Unexpected process_target_version for {sample}")
    for field in ("process_prompt", "judge_prompt", "answer"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing {field}: {sample}")
    target = row.get("process_target")
    if not isinstance(target, dict):
        raise ValueError(f"Missing process_target: {sample}")
    # This public deterministic entry point performs the target validation used
    # by scripts/reward.py without requiring a Judge request.
    score_rule_components("", target)
    if target.get("reasoning_mode") != "nested_belief":
        raise ValueError(f"Natural-CoT requires nested_belief targets: {sample}")
    if target.get("tom_order") != row.get("question_order"):
        raise ValueError(f"question_order/process_target mismatch: {sample}")
    if normalize(target.get("answer")) != normalize(row["answer"]):
        raise ValueError(f"answer/process_target mismatch: {sample}")


def validate_pairs(records_by_split: dict[str, list[dict[str, Any]]]) -> None:
    """Validate IDs, counterfactual pairs, and split isolation."""
    seen_samples: set[str] = set()
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    scenario_splits: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for split, rows in records_by_split.items():
        for row in rows:
            _validate_row(row, split)
            sample = str(row["global_sample_id"])
            if sample in seen_samples:
                raise ValueError(f"Duplicate global_sample_id: {sample}")
            seen_samples.add(sample)
            groups[pair_id(row)].append(row)
            scenario = row.get("source_group_id")
            if scenario is None:
                raise ValueError(f"Missing source_group_id: {sample}")
            scenario_splits[(str(row.get("source_dataset")), str(scenario))].add(
                split
            )

    for current_pair, rows in groups.items():
        if len(rows) != 2 or {
            row.get("intervention_type") for row in rows
        } != {"observed", "hidden"}:
            raise ValueError(f"Incomplete pair: {current_pair}")
        if len({row.get("split") for row in rows}) != 1:
            raise ValueError(f"Pair crosses splits: {current_pair}")
        observed = next(row for row in rows if row["intervention_type"] == "observed")
        hidden = next(row for row in rows if row["intervention_type"] == "hidden")
        for key in ("tom_order", "belief_chain", "object", "reasoning_mode"):
            if observed["process_target"][key] != hidden["process_target"][key]:
                raise ValueError(f"Pair target mismatch for {current_pair}: {key}")

    leaked = [key for key, splits in scenario_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Source scenarios cross splits: {leaked[:5]}")


def derive_split(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Copy natural train/val/test into a fresh RFT train/dev/test directory."""
    source = {
        split: read_jsonl(input_dir / f"{split}.jsonl")
        for split in SOURCE_TO_RFT_SPLIT
    }
    validate_pairs(source)

    derived: dict[str, list[dict[str, Any]]] = {}
    for source_split, rft_split in SOURCE_TO_RFT_SPLIT.items():
        derived[rft_split] = [
            dict(row, split=rft_split) for row in source[source_split]
        ]
    validate_pairs(derived)

    output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in derived.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    all_rows = [row for rows in derived.values() for row in rows]
    manifest = {
        "name": "RobustToM natural-CoT v4 RFT split",
        "source_dataset_dir": str(input_dir),
        "prompt_version": PROMPT_VERSION,
        "contains_process_response": False,
        "split_mapping": SOURCE_TO_RFT_SPLIT,
        "input_files": {
            split: {
                "path": str(input_dir / f"{split}.jsonl"),
                "sha256": sha256_file(input_dir / f"{split}.jsonl"),
            }
            for split in source
        },
        "split_counts": {split: len(rows) for split, rows in derived.items()},
        "pair_counts": {
            split: len({pair_id(row) for row in rows})
            for split, rows in derived.items()
        },
        "source_counts": dict(
            Counter(str(row.get("source_dataset")) for row in all_rows)
        ),
        "order_counts": dict(
            Counter(str(row.get("question_order")) for row in all_rows)
        ),
        "intervention_counts": dict(
            Counter(str(row.get("intervention_type")) for row in all_rows)
        ),
        "output_sha256": {
            split: sha256_file(output_dir / f"{split}.jsonl") for split in derived
        },
        "process_target_version": "2.0",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/counterfactual_process_reward_v4_natural"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rft/derived_v3_fewshot"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(derive_split(args.input_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
