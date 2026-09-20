#!/usr/bin/env python3
"""Build paired student/privileged-teacher prompts for RobustToM OPSD."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from rft.common import read_jsonl, sha256_file, write_jsonl
from scripts.build_privileged_teacher_eval_data import (
    PRIVILEGED_PROMPT_VERSION,
    PRIVILEGED_REFERENCE_VERSION,
    build_privileged_teacher_row,
)


DEFAULT_INPUT = Path(
    "data/counterfactual_process_reward_v4_natural_compact/train.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/counterfactual_process_reward_v4_natural_compact_opsd"
)


def _require_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        sample = row.get("global_sample_id", "<unknown>")
        raise ValueError(f"Missing {key} for {sample}")
    return value


def build_opsd_row(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return one GOLD-compatible record with explicitly paired prompts."""
    teacher = build_privileged_teacher_row(source)
    student_prompt = _require_string(source, "process_prompt")
    teacher_prompt = _require_string(teacher, "process_prompt")
    order = source.get("question_order")
    support_events = teacher["privileged_reference"]["support_events"]
    if type(order) is not int or order < 1 or len(support_events) != order:
        raise ValueError(
            "Each OPSD row must expose exactly one support event per belief level: "
            f"{source.get('global_sample_id', '<unknown>')}"
        )
    return {
        # The upstream GOLD/OPSD trainer retains these two column names.  Here
        # they contain complete prompts rather than math problem/solution text.
        "problem": student_prompt,
        "solution": teacher_prompt,
        "global_sample_id": _require_string(source, "global_sample_id"),
        "global_pair_id": _require_string(source, "global_pair_id"),
        "question_order": order,
        "intervention_type": _require_string(source, "intervention_type"),
        "student_process_prompt_version": _require_string(
            teacher, "student_process_prompt_version"
        ),
        "teacher_process_prompt_version": PRIVILEGED_PROMPT_VERSION,
        "privileged_reference_version": PRIVILEGED_REFERENCE_VERSION,
        "privileged_reference": teacher["privileged_reference"],
    }


def build_dataset(input_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    source_rows = read_jsonl(input_path)
    rows = [build_opsd_row(row) for row in source_rows]
    if not rows:
        raise ValueError("Input dataset is empty")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "train.jsonl"
    write_jsonl(output_path, rows)

    manifest = {
        "name": "RobustToM privileged-context OPSD train data",
        "count": len(rows),
        "pair_count": len({row["global_pair_id"] for row in rows}),
        "order_counts": dict(
            sorted(Counter(str(row["question_order"]) for row in rows).items())
        ),
        "intervention_counts": dict(
            sorted(Counter(row["intervention_type"] for row in rows).items())
        ),
        "support_event_count_distribution": dict(
            sorted(
                Counter(
                    str(len(row["privileged_reference"]["support_events"]))
                    for row in rows
                ).items()
            )
        ),
        "source_file": str(input_path),
        "output_file": str(output_path),
        "student_column": "problem",
        "teacher_column": "solution",
        "teacher_prompt_version": PRIVILEGED_PROMPT_VERSION,
        "privileged_reference_version": PRIVILEGED_REFERENCE_VERSION,
        "gold_answer_exposed_to_teacher_only": True,
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args.input, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
