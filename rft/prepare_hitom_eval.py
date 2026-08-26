#!/usr/bin/env python3
"""Build the fixed Hi-ToM order-4 answer-only evaluation dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from grpo.prompt import ORDER_TRACE_INSTRUCTION, build_grpo_prompt
from rft.common import sha256_file, sha256_text, write_jsonl
from scripts.add_symbolic_v3_few_shots import (
    FEW_SHOT_COUNT,
    FEW_SHOT_MARKER,
    FEW_SHOT_VERSION,
)
from scripts.build_hi_tom_counterfactual import (
    parse_choices,
    question_agents,
    question_object,
)


SOURCE_DATASET = "hi-tom-order4"
DEFAULT_INPUT = Path("data/cleaned_tom/raw/hi_tom_3000.csv")
DEFAULT_OUTPUT_DIR = Path("data/rft/hitom_order4")
EXPECTED_ORDER4_COUNT = 600

HITOM_ASSUMPTIONS = (
    "Hi-ToM assumptions:\n"
    "(1) An agent witnesses every event and every movement before exiting a room.\n"
    "(2) An agent A can infer another agent B's mental state only if A and B "
    "have been in the same location, or have private or public interactions.\n"
    "(3) Every agent tends to lie. What an agent A tells others does not affect "
    "A's actual belief. An agent tends to trust an agent that exited the room "
    "later than themself. The exit order is known to all agents.\n"
    "(4) Agents in private communications know that others will not hear them, "
    "while everyone knows that public claims can be heard."
)

NESTED_BELIEF_SCHEMA = (
    '{"tom_order":2,"belief_chain":["outer","inner"],"object":"...",'
    '"reasoning_mode":"nested_belief","belief_trace":['
    '{"belief_chain":["inner"],"location":"..."},'
    '{"belief_chain":["outer","inner"],"location":"..."}],'
    '"answer":"..."}'
)

STORY_EVENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
NUMBERED_STORY_EVENT = re.compile(r"^\d+\s+(.+)$")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "deception",
            "story_length",
            "question_order",
            "sample_id",
            "story",
            "question",
            "choices",
            "answer",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing Hi-ToM CSV columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def parse_boolean(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid {field} boolean: {value!r}")


def number_story_events(story: str) -> str:
    """Format each sentence-like Hi-ToM event as a one-based numbered line."""
    source_lines = [line.strip() for line in story.strip().splitlines() if line.strip()]
    numbered_events = [
        match.group(1).strip()
        for line in source_lines
        if (match := NUMBERED_STORY_EVENT.fullmatch(line))
    ]
    if numbered_events:
        non_event_lines = [
            line
            for line in source_lines
            if not NUMBERED_STORY_EVENT.fullmatch(line) and line != "***"
        ]
        if non_event_lines and non_event_lines != [
            "Read the following story and answer the multiple-choice question. "
            "Please provide answer without explanations."
        ]:
            raise ValueError(f"Unexpected text around numbered Hi-ToM events: {non_event_lines}")
        events = numbered_events
    else:
        events = [
            event.strip()
            for event in STORY_EVENT_BOUNDARY.split(story.strip())
            if event.strip()
        ]
    if not events:
        raise ValueError("Hi-ToM story must contain at least one event")
    return "\n".join(
        f"{event_number} {event}"
        for event_number, event in enumerate(events, start=1)
    )


def build_base_prompt(story: str, question: str, choices: str) -> str:
    return (
        "Return exactly one JSON object and no markdown or extra text. Copy "
        "names exactly. belief_chain runs from the outermost thinker to the "
        "innermost thinker. belief_trace must list suffix chains from the "
        "innermost belief through the full queried chain. Locations, not "
        "choice letters, belong in belief_trace and answer.\n\n"
        f"{HITOM_ASSUMPTIONS}\n\n"
        f"Schema:\n{NESTED_BELIEF_SCHEMA}\n\n"
        f"Story:\n{story.rstrip()}\n\n"
        f"Question: {question.strip()}\n"
        f"Choices: {choices.strip()}\n"
    )


def stable_sample_id(row: dict[str, str]) -> str:
    deception = str(parse_boolean(row["deception"], "deception")).lower()
    story_length = int(row["story_length"])
    sample = int(row["sample_id"])
    return (
        f"hi-tom-order4:deception={deception}:"
        f"length={story_length}:sample={sample:03d}"
    )


def convert_row(row: dict[str, str], order: int) -> dict[str, Any]:
    row_order = int(row["question_order"])
    if row_order != order:
        raise ValueError(f"Expected order {order}, got {row_order}")

    raw_story = row["story"].strip()
    question = row["question"].strip()
    choices = row["choices"].strip()
    answer = row["answer"].strip()
    if not raw_story or not question or not choices or not answer:
        raise ValueError("Hi-ToM story, question, choices, and answer must be non-empty")

    story = number_story_events(raw_story)

    belief_chain = question_agents(question)
    if len(belief_chain) != order:
        raise ValueError(
            f"Belief-chain length {len(belief_chain)} != order {order}: {question!r}"
        )
    obj = question_object(question)
    choice_values = {value for _, value in parse_choices(choices)}
    if answer not in choice_values:
        raise ValueError(f"Gold answer {answer!r} is not in choices: {question!r}")

    base_prompt = build_base_prompt(story, question, choices)
    process_prompt = build_grpo_prompt(base_prompt)
    if process_prompt.count(FEW_SHOT_MARKER) != 1:
        raise AssertionError("Hi-ToM prompt must contain exactly one few-shot block")
    if process_prompt.count(ORDER_TRACE_INSTRUCTION) != 1:
        raise AssertionError("Hi-ToM prompt must contain exactly one order instruction")

    return {
        "global_sample_id": stable_sample_id(row),
        "source_dataset": SOURCE_DATASET,
        "source_split": "test",
        "question_order": order,
        "belief_chain": belief_chain,
        "object": obj,
        "story": story,
        "question": question,
        "choices": choices,
        "gold_answer": answer,
        "deception": parse_boolean(row["deception"], "deception"),
        "story_length": int(row["story_length"]),
        "source_sample_id": int(row["sample_id"]),
        "process_prompt": process_prompt,
        "base_process_prompt_sha256": sha256_text(base_prompt),
        "process_prompt_sha256": sha256_text(process_prompt),
        "few_shot_version": FEW_SHOT_VERSION,
        "few_shot_count": FEW_SHOT_COUNT,
    }


def prepare_hitom_eval(
    input_path: Path,
    output_dir: Path,
    order: int = 4,
    expected_count: int | None = EXPECTED_ORDER4_COUNT,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    source_rows = read_csv_rows(input_path)
    selected = [row for row in source_rows if int(row["question_order"]) == order]
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(
            f"Expected {expected_count} order-{order} rows, found {len(selected)}"
        )
    rows = [convert_row(row, order) for row in selected]
    sample_ids = [row["global_sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        duplicates = [
            sample_id
            for sample_id, count in Counter(sample_ids).items()
            if count > 1
        ]
        raise ValueError(f"Duplicate global_sample_id values: {duplicates[:5]}")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "test.jsonl"
    write_jsonl(output_path, rows)
    manifest = {
        "name": "Hi-ToM order-4 answer-only benchmark with symbolic-v3 few-shot prompts",
        "source_dataset": SOURCE_DATASET,
        "source_file": str(input_path),
        "source_file_sha256": sha256_file(input_path),
        "filter": {"question_order": order},
        "count": len(rows),
        "deception_counts": dict(Counter(str(row["deception"]).lower() for row in rows)),
        "story_length_counts": dict(Counter(str(row["story_length"]) for row in rows)),
        "few_shot_version": FEW_SHOT_VERSION,
        "few_shot_count": FEW_SHOT_COUNT,
        "story_event_numbering": "one-based sentence order",
        "order_trace_instruction": ORDER_TRACE_INSTRUCTION,
        "output_file": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Hi-ToM order-4 answer-only benchmark\n\n"
        "This directory is derived from `data/cleaned_tom/raw/hi_tom_3000.csv` "
        "by selecting the 600 fourth-order questions. Each prompt uses the "
        "same one-based event numbering as the symbolic-v3 data, plus the "
        "symbolic-v3 three-shot block and the explicit `tom_order` / "
        "`belief_trace` cardinality instruction through `build_grpo_prompt`. "
        f"The exact clarification is: `{ORDER_TRACE_INSTRUCTION}` "
        "Only the final JSON `answer` is scored against `gold_answer`; Hi-ToM "
        "does not provide gold intermediate belief traces.\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_ORDER4_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_hitom_eval(
        args.input,
        args.output_dir,
        order=args.order,
        expected_count=args.expected_count,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
