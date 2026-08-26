#!/usr/bin/env python3
"""Build audited verl parquet files from the symbolic v3 process dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from grpo.prompt import ORDER_TRACE_INSTRUCTION, build_grpo_prompt, chat_prompt_token_ids
from rft.common import read_jsonl


DATA_SOURCE = "robust_tom_process_v3"
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


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def build_parquet_row(
    source: dict[str, Any],
    index: int,
    tokenizer: Any,
    max_prompt_length: int,
    max_response_length: int,
) -> dict[str, Any]:
    """Convert one raw v3 row while preserving all scoring metadata."""
    sample_id = source.get("global_sample_id")
    if source.get("process_target_version") != "2.0":
        raise ValueError(f"Expected process target version 2.0: {sample_id}")
    target = source.get("process_target")
    if not isinstance(target, dict) or target.get("reasoning_mode") != "nested_belief":
        raise ValueError(f"Expected a nested_belief target: {sample_id}")
    response = source.get("process_response")
    if not isinstance(response, str) or json.loads(response) != target:
        raise ValueError(f"process_response/target mismatch: {sample_id}")

    prompt = build_grpo_prompt(source.get("process_prompt", ""))
    prompt_length = len(chat_prompt_token_ids(tokenizer, prompt))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    response_length = len(_encode(tokenizer, response)) + 1
    if prompt_length > max_prompt_length:
        raise ValueError(
            f"Prompt {sample_id} has {prompt_length} tokens, exceeding {max_prompt_length}"
        )
    if response_length > max_response_length:
        raise ValueError(
            f"Canonical response {sample_id} has {response_length} tokens, "
            f"exceeding {max_response_length}"
        )

    extra_info = {
        "index": index,
        "global_sample_id": str(sample_id),
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
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {"style": "rule", "ground_truth": response},
        "extra_info": extra_info,
        "prompt_token_count": prompt_length,
        "canonical_response_token_count": response_length,
    }


def convert_split(
    input_path: Path,
    output_path: Path,
    tokenizer: Any,
    max_prompt_length: int,
    max_response_length: int,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised on the training host
        raise RuntimeError("pandas and pyarrow are required to build parquet data") from exc

    source_rows = read_jsonl(input_path)
    rows = [
        build_parquet_row(
            source,
            index=index,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
        )
        for index, source in enumerate(source_rows)
    ]
    if len({row["extra_info"]["index"] for row in rows}) != len(rows):
        raise AssertionError("Every prompt index must be unique within its split")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False, engine="pyarrow")
    prompt_lengths = [row["prompt_token_count"] for row in rows]
    response_lengths = [row["canonical_response_token_count"] for row in rows]
    return {
        "count": len(rows),
        "pair_count": len({row["extra_info"]["global_pair_id"] for row in rows}),
        "order_counts": dict(Counter(str(row["extra_info"]["question_order"]) for row in rows)),
        "intervention_counts": dict(
            Counter(row["extra_info"]["intervention_type"] for row in rows)
        ),
        "prompt_tokens": {
            "max": max(prompt_lengths),
            "p95": _percentile(prompt_lengths, 0.95),
            "p99": _percentile(prompt_lengths, 0.99),
        },
        "canonical_response_tokens": {
            "max": max(response_lengths),
            "p95": _percentile(response_lengths, 0.95),
            "p99": _percentile(response_lengths, 0.99),
        },
        "input_sha256": _sha256_file(input_path),
        "output_sha256": _sha256_file(output_path),
    }


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str,
    max_prompt_length: int = 2048,
    max_response_length: int = 256,
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised on the training host
        raise RuntimeError("transformers is required to audit prompt lengths") from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    split_metrics: dict[str, Any] = {}
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        input_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}.parquet"
        split_metrics[split] = convert_split(
            input_path,
            output_path,
            tokenizer,
            max_prompt_length,
            max_response_length,
        )
        if split_metrics[split]["count"] != expected_count:
            raise ValueError(
                f"Unexpected {split} count: {split_metrics[split]['count']} != {expected_count}"
            )

    manifest = {
        "name": "RobustToM symbolic v3 GRPO data with fixed 3-shot prompts",
        "data_source": DATA_SOURCE,
        "input_dir": str(input_dir),
        "tokenizer": tokenizer_name,
        "max_prompt_length": max_prompt_length,
        "max_response_length": max_response_length,
        "order_trace_instruction": ORDER_TRACE_INSTRUCTION,
        "few_shot_count": 3,
        "splits": split_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
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
        "--output-dir",
        type=Path,
        default=Path("data/grpo/counterfactual_process_reward_v3_fewshot"),
    )
    parser.add_argument("--tokenizer", default="runs/final")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.input_dir,
        args.output_dir,
        args.tokenizer,
        args.max_prompt_length,
        args.max_response_length,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

