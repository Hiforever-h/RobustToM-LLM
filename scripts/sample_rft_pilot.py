#!/usr/bin/env python3
"""Select a deterministic, complete-pair RFT pilot split by ToM order."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rft.common import pair_id, read_jsonl, sample_id, sha256_file, stable_hash, write_jsonl
from rft.prompt import NATURAL_COT_PROMPT_VERSION

DEFAULT_INPUT = Path("data/rft/derived_v3_fewshot/train.jsonl")
DEFAULT_OUTPUT = Path("data/rft/pilot_v2/train.jsonl")
SELECTION_VERSION = "rft-pilot-complete-pairs-v1"
INTERVENTION_ORDER = {"observed": 0, "hidden": 1}


def _validated_order(row: dict[str, Any]) -> int:
    order = row.get("question_order")
    target = row.get("process_target")
    if type(order) is not int or order < 1:
        raise ValueError(f"Invalid question_order: {sample_id(row)}")
    if not isinstance(target, dict) or target.get("tom_order") != order:
        raise ValueError(f"question_order/process_target mismatch: {sample_id(row)}")
    if row.get("process_prompt_version") != NATURAL_COT_PROMPT_VERSION:
        raise ValueError(
            f"Expected {NATURAL_COT_PROMPT_VERSION}: {sample_id(row)}"
        )
    intervention = row.get("intervention_type")
    if intervention not in INTERVENTION_ORDER:
        raise ValueError(f"Invalid intervention_type: {sample_id(row)}")
    return order


def select_pilot_rows(
    rows: Iterable[dict[str, Any]],
    *,
    orders: tuple[int, ...] = (1, 2, 3),
    pairs_per_order: int = 50,
    seed: int = 2026,
    num_samples_per_prompt: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose complete observed/hidden pairs with stable hash-based ranking."""
    if not orders or len(set(orders)) != len(orders) or any(order < 1 for order in orders):
        raise ValueError("orders must contain unique positive integers")
    if pairs_per_order < 1:
        raise ValueError("pairs_per_order must be positive")
    if num_samples_per_prompt < 1:
        raise ValueError("num_samples_per_prompt must be positive")

    source_rows = list(rows)
    seen_samples: set[str] = set()
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    order_by_pair: dict[str, int] = {}
    for row in source_rows:
        current_sample = sample_id(row)
        if current_sample in seen_samples:
            raise ValueError(f"Duplicate global_sample_id: {current_sample}")
        seen_samples.add(current_sample)
        order = _validated_order(row)
        current_pair = pair_id(row)
        previous_order = order_by_pair.setdefault(current_pair, order)
        if previous_order != order:
            raise ValueError(f"Pair crosses question orders: {current_pair}")
        grouped[current_pair].append(row)

    eligible_by_order: defaultdict[int, list[str]] = defaultdict(list)
    for current_pair, pair_rows in grouped.items():
        sides = {row["intervention_type"] for row in pair_rows}
        if len(pair_rows) != 2 or sides != set(INTERVENTION_ORDER):
            raise ValueError(f"Incomplete observed/hidden pair: {current_pair}")
        eligible_by_order[order_by_pair[current_pair]].append(current_pair)

    selected_pair_ids: dict[int, list[str]] = {}
    for order in orders:
        eligible = eligible_by_order[order]
        if len(eligible) < pairs_per_order:
            raise ValueError(
                f"Order {order} has only {len(eligible)} eligible pairs; "
                f"requested {pairs_per_order}"
            )
        ranked = sorted(
            eligible,
            key=lambda current_pair: (
                stable_hash(SELECTION_VERSION, seed, order, current_pair),
                current_pair,
            ),
        )
        selected_pair_ids[order] = ranked[:pairs_per_order]

    selected: list[dict[str, Any]] = []
    for order in orders:
        for current_pair in selected_pair_ids[order]:
            selected.extend(
                sorted(
                    grouped[current_pair],
                    key=lambda row: INTERVENTION_ORDER[row["intervention_type"]],
                )
            )

    order_counts = Counter(str(row["question_order"]) for row in selected)
    intervention_counts = Counter(row["intervention_type"] for row in selected)
    bucket_counts = Counter(
        f"order={row['question_order']}|{row['intervention_type']}"
        for row in selected
    )
    manifest = {
        "name": "RobustToM natural-CoT v2 stratified RFT pilot",
        "selection_version": SELECTION_VERSION,
        "process_prompt_version": NATURAL_COT_PROMPT_VERSION,
        "seed": seed,
        "orders": list(orders),
        "pairs_per_order": pairs_per_order,
        "prompt_count": len(selected),
        "pair_count": len(selected_pair_ids) * pairs_per_order,
        "num_samples_per_prompt": num_samples_per_prompt,
        "expected_candidate_count": len(selected) * num_samples_per_prompt,
        "source_prompt_count": len(source_rows),
        "eligible_pair_counts": {
            str(order): len(eligible_by_order[order]) for order in orders
        },
        "order_counts": dict(sorted(order_counts.items())),
        "intervention_counts": dict(sorted(intervention_counts.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "selected_pair_ids": {
            str(order): selected_pair_ids[order] for order in orders
        },
    }
    return selected, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--pairs-per-order", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-samples-per-prompt", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or args.output.with_name("manifest.json")
    existing = [path for path in (args.output, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing pilot artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    selected, manifest = select_pilot_rows(
        read_jsonl(args.input),
        orders=tuple(args.orders),
        pairs_per_order=args.pairs_per_order,
        seed=args.seed,
        num_samples_per_prompt=args.num_samples_per_prompt,
    )
    write_jsonl(args.output, selected)
    manifest.update(
        {
            "source_file": str(args.input),
            "source_sha256": sha256_file(args.input),
            "output_file": str(args.output),
            "output_sha256": sha256_file(args.output),
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
