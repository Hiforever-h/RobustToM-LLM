#!/usr/bin/env python3
"""Build a fixed compact-prompt RFT JSONL from an existing natural split."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rft.common import read_jsonl, sha256_file, write_jsonl
from rft.prompt import (
    COMPACT_NATURAL_COT_PROMPT_VERSION,
    compact_process_record,
)


def build_compact_data(
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy rows and replace only their model-facing prompt and version."""
    compact_rows = [compact_process_record(row) for row in source_rows]
    for row in compact_rows:
        prompt = row["process_prompt"]
        format_instruction = prompt.rsplit("\n\n", 1)[-1]
        if "Infer N from the question itself" not in prompt:
            raise AssertionError("Compact prompt is missing the inferred-order rule")
        if "Reasoning rules:" in prompt or "Required output format:" in prompt:
            raise AssertionError("Compact prompt retained the v2 actor scaffold")
        if re.search(r"\bN\s*=\s*\d+\b", format_instruction):
            raise AssertionError("Compact prompt leaked the numeric ToM order")
    return compact_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output and manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or args.output.with_name(
        f"{args.output.stem}_manifest.json"
    )
    existing = [path for path in (args.output, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing compact data: "
            + ", ".join(str(path) for path in existing)
        )

    source_rows = read_jsonl(args.input)
    compact_rows = build_compact_data(source_rows)
    write_jsonl(args.output, compact_rows)
    manifest = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "row_count": len(compact_rows),
        "process_prompt_version": COMPACT_NATURAL_COT_PROMPT_VERSION,
        "split_counts": dict(
            Counter(str(row.get("split", "unknown")) for row in compact_rows)
        ),
        "order_counts": dict(
            Counter(str(row.get("question_order", "unknown")) for row in compact_rows)
        ),
        "contains_process_response": any(
            "process_response" in row for row in compact_rows
        ),
        "numeric_order_injected_into_format_instruction": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
