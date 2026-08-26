#!/usr/bin/env python3
"""Add compact nested-belief demonstrations to a derived symbolic v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from generate_symbolic_counterfactual_v3 import TokenCounter, percentile
except ModuleNotFoundError:  # Allow importing as scripts.<module> in tests.
    from scripts.generate_symbolic_counterfactual_v3 import TokenCounter, percentile


FEW_SHOT_VERSION = "symbolic-v3-3shot-1.0"
FEW_SHOT_COUNT = 3
STORY_MARKER = "\n\nStory:\n"
FEW_SHOT_MARKER = "Nested-belief demonstrations (3-shot):"
FEW_SHOT_BLOCK = r'''Nested-belief demonstrations (3-shot):
These examples demonstrate the update rules and exact JSON format. Solve the actual story independently. Locations, not choice letters, belong in belief_trace and answer.

Demonstration 1: first-order private observation
Story:
1 Ada and Bruno jointly watched the key being placed in the red_box.
2 Ada alone received a private feed showing the key move from the red_box to the blue_box.
3 Unseen by everyone, the key moved from the blue_box to the green_box.
Question: Where does Ada think the key is?
Choices: A. red_box, B. blue_box, C. green_box
Output:
{"tom_order":1,"belief_chain":["Ada"],"object":"key","reasoning_mode":"nested_belief","belief_trace":[{"belief_chain":["Ada"],"location":"blue_box"}],"answer":"blue_box"}

Demonstration 2: second-order suffix separation
Story:
1 Alice, Bob, and Carol jointly watched the coin being placed in the red_drawer.
2 Alice and Bob watched together as the coin moved from the red_drawer to the blue_drawer; each saw the other watching.
3 Bob alone received a private feed showing the coin move from the blue_drawer to the green_drawer.
4 Unseen by everyone, the coin moved from the green_drawer to the grey_drawer.
Question: Where does Alice think Bob thinks the coin is?
Choices: A. red_drawer, B. blue_drawer, C. green_drawer, D. grey_drawer
Output:
{"tom_order":2,"belief_chain":["Alice","Bob"],"object":"coin","reasoning_mode":"nested_belief","belief_trace":[{"belief_chain":["Bob"],"location":"green_drawer"},{"belief_chain":["Alice","Bob"],"location":"blue_drawer"}],"answer":"blue_drawer"}

Demonstration 3: third-order suffix expansion
Story:
1 Ava, Ben, Cora, and Dan jointly watched the map being placed in the red_case.
2 Ava, Ben, and Cora watched together as the map moved from the red_case to the blue_case; each saw the full group watching.
3 Ben and Cora watched together as the map moved from the blue_case to the green_case; each saw the other watching.
4 Cora alone received a private feed showing the map move from the green_case to the yellow_case.
5 Unseen by everyone, the map moved from the yellow_case to the grey_case.
Question: Where does Ava think Ben thinks Cora thinks the map is?
Choices: A. red_case, B. blue_case, C. green_case, D. yellow_case, E. grey_case
Output:
{"tom_order":3,"belief_chain":["Ava","Ben","Cora"],"object":"map","reasoning_mode":"nested_belief","belief_trace":[{"belief_chain":["Cora"],"location":"yellow_case"},{"belief_chain":["Ben","Cora"],"location":"green_case"},{"belief_chain":["Ava","Ben","Cora"],"location":"blue_case"}],"answer":"blue_case"}'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def add_few_shots(prompt: str) -> str:
    if FEW_SHOT_MARKER in prompt:
        raise ValueError("Prompt already contains symbolic v3 few-shot demonstrations")
    if prompt.count(STORY_MARKER) != 1:
        raise ValueError("Expected exactly one Story marker in the process prompt")
    instructions, task = prompt.split(STORY_MARKER, maxsplit=1)
    return f"{instructions}\n\n{FEW_SHOT_BLOCK}{STORY_MARKER}{task}"


def augment_record(
    source: dict[str, Any], counter: TokenCounter, max_tokens: int
) -> dict[str, Any]:
    if source.get("process_target_version") != "2.0":
        raise ValueError(f"Expected process target version 2.0: {source.get('global_sample_id')}")
    target = source.get("process_target")
    if not isinstance(target, dict) or target.get("reasoning_mode") != "nested_belief":
        raise ValueError(f"Expected a nested-belief target: {source.get('global_sample_id')}")
    prompt = source.get("process_prompt")
    response = source.get("process_response")
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValueError(f"Missing process prompt/response: {source.get('global_sample_id')}")

    augmented = dict(source)
    augmented["base_process_prompt_sha256"] = sha256_text(prompt)
    augmented["process_prompt"] = add_few_shots(prompt)
    augmented["few_shot_version"] = FEW_SHOT_VERSION
    augmented["few_shot_count"] = FEW_SHOT_COUNT
    prompt_tokens = counter.raw_count(augmented["process_prompt"])
    sequence_tokens = counter.sequence_count(augmented["process_prompt"], response)
    if prompt_tokens > max_tokens or sequence_tokens > max_tokens:
        raise ValueError(
            f"Token limit exceeded for {source.get('global_sample_id')}: "
            f"prompt={prompt_tokens}, sequence={sequence_tokens}"
        )
    augmented["process_prompt_token_count"] = prompt_tokens
    augmented["process_sequence_token_count"] = sequence_tokens
    return augmented


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str | None,
    max_tokens: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    counter = TokenCounter(tokenizer_name)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        rows_by_split[split] = [
            augment_record(row, counter, max_tokens)
            for row in read_jsonl(input_dir / f"{split}.jsonl")
        ]

    rows = [row for split_rows in rows_by_split.values() for row in split_rows]
    prompt_lengths = [row["process_prompt_token_count"] for row in rows]
    sequence_lengths = [row["process_sequence_token_count"] for row in rows]
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, split_rows in rows_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_rows)

    input_manifest = input_dir / "manifest.json"
    manifest = {
        "name": "RobustToM-RL symbolic v3 RFT split with 3-shot prompts",
        "few_shot_version": FEW_SHOT_VERSION,
        "few_shot_count": FEW_SHOT_COUNT,
        "demonstration_orders": [1, 2, 3],
        "process_target_version": "2.0",
        "input_dir": str(input_dir),
        "input_manifest_sha256": sha256_file(input_manifest),
        "tokenizer": counter.name,
        "max_process_and_sequence_tokens": max_tokens,
        "split_counts": {split: len(split_rows) for split, split_rows in rows_by_split.items()},
        "pair_counts": {
            split: len({row["global_pair_id"] for row in split_rows})
            for split, split_rows in rows_by_split.items()
        },
        "order_counts": dict(Counter(str(row["question_order"]) for row in rows)),
        "process_prompt_tokens": {
            "max": max(prompt_lengths),
            "p95": percentile(prompt_lengths, 0.95),
            "mean": round(sum(prompt_lengths) / len(prompt_lengths), 3),
        },
        "process_sequence_tokens": {
            "max": max(sequence_lengths),
            "p95": percentile(sequence_lengths, 0.95),
            "mean": round(sum(sequence_lengths) / len(sequence_lengths), 3),
        },
        "output_sha256": {
            split: sha256_file(output_dir / f"{split}.jsonl")
            for split in rows_by_split
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Symbolic v3 RFT data with 3-shot prompts\n\n"
        "This is a non-destructive prompt augmentation of `data/rft/derived_v3`. "
        "Each process prompt includes fixed order-1, order-2, and order-3 "
        "nested-belief demonstrations. Labels and split membership are unchanged.\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/rft/derived_v3"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/rft/derived_v3_fewshot")
    )
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = convert_dataset(
        args.input_dir, args.output_dir, args.tokenizer, args.max_tokens
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
