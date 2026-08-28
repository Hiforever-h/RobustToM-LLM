#!/usr/bin/env python3
"""Build the natural-CoT source JSONL and verl parquet dataset.

The actor sees a dynamic exact-order Think/State/Answer protocol. The Judge
receives ``judge_prompt`` as a separate one-line parquet column.
Neither artifact contains the legacy canonical ``process_response``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from grpo.prompt import (
    build_natural_cot_prompt,
    build_natural_problem_prompt,
    chat_prompt_token_ids,
    clean_numbered_story,
)
from rft.common import read_jsonl
from rft.prompt import NATURAL_COT_PROMPT_VERSION

DATA_SOURCE = "robust_tom_natural_cot_v4"
PROMPT_VERSION = NATURAL_COT_PROMPT_VERSION
EXPECTED_SPLIT_COUNTS = {"train": 3200, "val": 400, "test": 600}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _validate_target(source: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = source.get("global_sample_id")
    if source.get("process_target_version") != "2.0":
        raise ValueError(f"Expected process target version 2.0: {sample_id}")
    target = source.get("process_target")
    if not isinstance(target, dict) or target.get("reasoning_mode") != "nested_belief":
        raise ValueError(f"Expected a nested_belief process target: {sample_id}")
    trace = target.get("belief_trace")
    if not isinstance(trace, list) or len(trace) != target.get("tom_order"):
        raise ValueError(f"Invalid process target trace: {sample_id}")
    if target.get("answer") != source.get("answer"):
        raise ValueError(f"Process target answer mismatch: {sample_id}")
    return target


def build_natural_source_row(source: Mapping[str, Any]) -> dict[str, Any]:
    """Remove legacy response/schema fields and create explicit actor/Judge prompts."""
    _validate_target(source)
    row = dict(source)
    row["story"] = clean_numbered_story(str(source.get("story", "")))
    row["process_prompt"] = build_natural_cot_prompt(source)
    row["judge_prompt"] = build_natural_problem_prompt(source)
    row["process_prompt_version"] = PROMPT_VERSION
    for key in (
        "process_response",
        "process_prompt_token_count",
        "process_sequence_token_count",
        "prompt",
    ):
        row.pop(key, None)
    if "\n" in row["story"] or "\n" in row["judge_prompt"]:
        raise AssertionError("Natural-CoT stories and Judge prompts must be single-line")
    if "\n" not in row["process_prompt"]:
        raise AssertionError("Natural-CoT actor prompts must expose a multiline template")
    return row


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare_source_dataset(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Create the standalone natural-CoT JSONL dataset using only stdlib I/O."""
    split_metrics: dict[str, Any] = {}
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        input_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}.jsonl"
        rows = [build_natural_source_row(row) for row in read_jsonl(input_path)]
        if len(rows) != expected_count:
            raise ValueError(
                f"Unexpected {split} count: {len(rows)} != {expected_count}"
            )
        _write_jsonl(output_path, rows)
        split_metrics[split] = {
            "count": len(rows),
            "pair_count": len({row["global_pair_id"] for row in rows}),
            "order_counts": dict(Counter(str(row["question_order"]) for row in rows)),
            "intervention_counts": dict(
                Counter(str(row["intervention_type"]) for row in rows)
            ),
            "input_sha256": _sha256_file(input_path),
            "output_sha256": _sha256_file(output_path),
        }
    manifest = {
        "name": "RobustToM natural-CoT v4 source data",
        "prompt_version": PROMPT_VERSION,
        "contains_process_response": False,
        "event_numbers_removed": True,
        "multiline_actor_prompts": True,
        "single_line_judge_prompts": True,
        "splits": split_metrics,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parquet_row(
    source: Mapping[str, Any],
    index: int,
    tokenizer: Any,
    max_prompt_length: int,
) -> dict[str, Any]:
    """Convert one natural source row while preserving Judge and scoring fields."""
    target = _validate_target(source)
    actor_prompt = source.get("process_prompt")
    judge_prompt = source.get("judge_prompt")
    if not isinstance(actor_prompt, str) or not actor_prompt.strip():
        raise ValueError("Natural source row is missing process_prompt")
    if not isinstance(judge_prompt, str) or not judge_prompt.strip():
        raise ValueError("Natural source row is missing judge_prompt")
    if source.get("process_response") is not None:
        raise ValueError("Natural source rows must not contain process_response")
    if source.get("process_prompt_version") != PROMPT_VERSION:
        raise ValueError(
            "Natural source row has an unexpected process_prompt_version: "
            f"{source.get('process_prompt_version')!r}"
        )

    prompt_length = len(chat_prompt_token_ids(tokenizer, actor_prompt))
    if prompt_length > max_prompt_length:
        raise ValueError(
            f"Prompt {source.get('global_sample_id')} has {prompt_length} tokens, "
            f"exceeding {max_prompt_length}"
        )
    extra_info = {
        "index": index,
        "global_sample_id": str(source.get("global_sample_id")),
        "global_pair_id": str(source.get("global_pair_id")),
        "source_dataset": str(source.get("source_dataset", "symbolic-tom-v3")),
        "question_order": int(source["question_order"]),
        "intervention_type": str(source["intervention_type"]),
        "shortcut_conflict": bool(source.get("shortcut_conflict", False)),
        "shortcut_prediction": str(source.get("shortcut_prediction", "")),
        "last_mention_conflict": bool(source.get("last_mention_conflict", False)),
        "last_mentioned_container": str(source.get("last_mentioned_container", "")),
    }
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": actor_prompt}],
        "judge_prompt": judge_prompt,
        "reward_model": {
            "style": "natural_cot_judge",
            "ground_truth": target,
        },
        "extra_info": extra_info,
        "prompt_token_count": prompt_length,
        "target_step_count": len(target["belief_trace"]),
    }


def convert_parquet_split(
    input_path: Path,
    output_path: Path,
    tokenizer: Any,
    max_prompt_length: int,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - training host dependency
        raise RuntimeError(
            "pandas and pyarrow are required to build parquet data"
        ) from exc

    source_rows = read_jsonl(input_path)
    rows = [
        build_parquet_row(row, index, tokenizer, max_prompt_length)
        for index, row in enumerate(source_rows)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(rows).to_parquet(output_path, index=False, engine="pyarrow")
    except ImportError as exc:  # pragma: no cover - training host dependency
        raise RuntimeError(
            "pandas and pyarrow are required to build parquet data"
        ) from exc
    prompt_lengths = [row["prompt_token_count"] for row in rows]
    return {
        "count": len(rows),
        "pair_count": len({row["extra_info"]["global_pair_id"] for row in rows}),
        "prompt_tokens": {
            "max": max(prompt_lengths),
            "p95": _percentile(prompt_lengths, 0.95),
            "p99": _percentile(prompt_lengths, 0.99),
        },
        "output_sha256": _sha256_file(output_path),
    }


def build_parquet_dataset(
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str,
    max_prompt_length: int = 2048,
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - training host dependency
        raise RuntimeError("transformers is required to audit prompt lengths") from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    splits: dict[str, Any] = {}
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        splits[split] = convert_parquet_split(
            input_dir / f"{split}.jsonl",
            output_dir / f"{split}.parquet",
            tokenizer,
            max_prompt_length,
        )
        if splits[split]["count"] != expected_count:
            actual_count = splits[split]["count"]
            raise ValueError(
                f"Unexpected {split} count: {actual_count} != {expected_count}"
            )
    manifest = {
        "name": "RobustToM natural-CoT v4 verl parquet",
        "prompt_version": PROMPT_VERSION,
        "tokenizer": tokenizer_name,
        "max_prompt_length": max_prompt_length,
        "contains_judge_prompt": True,
        "ground_truth_is_process_target": True,
        "splits": splits,
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
        default=Path("data/counterfactual_process_reward_v3"),
    )
    parser.add_argument(
        "--source-output-dir",
        type=Path,
        default=Path("data/counterfactual_process_reward_v4_natural"),
    )
    parser.add_argument(
        "--parquet-output-dir",
        type=Path,
        default=Path("data/grpo/counterfactual_process_reward_v4_natural"),
    )
    parser.add_argument("--tokenizer", default="runs/final")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Create natural JSONL without requiring transformer/parquet dependencies",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest = prepare_source_dataset(args.input_dir, args.source_output_dir)
    result: dict[str, Any] = {"source": source_manifest}
    if not args.source_only:
        result["parquet"] = build_parquet_dataset(
            args.source_output_dir,
            args.parquet_output_dir,
            args.tokenizer,
            args.max_prompt_length,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
