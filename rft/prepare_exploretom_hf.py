#!/usr/bin/env python3
"""Convert the official ExploreToM Hugging Face sample to RFT JSONL."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from grpo.prompt import ORDER_TRACE_INSTRUCTION, build_grpo_prompt
from rft.common import sha256_file, sha256_text, write_jsonl
from rft.prepare_hitom_eval import NESTED_BELIEF_SCHEMA, number_story_events
from scripts.add_symbolic_v3_few_shots import (
    FEW_SHOT_COUNT,
    FEW_SHOT_MARKER,
    FEW_SHOT_VERSION,
)


DATASET_NAME = "ExploreToM-HF-sample"
SOURCE_DATASET = "facebook/ExploreToM"
SOURCE_URL = "https://huggingface.co/datasets/facebook/ExploreToM"
LICENSE = "CC-BY-NC-4.0"
DEFAULT_INPUT = Path("data/ExploreToM/ExploreToM-data-sample.csv")
DEFAULT_OUTPUT_DIR = Path("data/ExploreToM")
EXPECTED_SOURCE_COUNT = 13309

REQUIRED_FIELDS = {
    "story_structure",
    "infilled_story",
    "question",
    "expected_answer",
    "qprop=params",
    "qprop=nth_order",
    "qprop=non_unique_mental_state",
    "sprop=is_false_belief_story_1st",
    "sprop=is_false_belief_story_1st_and_2nd",
    "sprop=story_accuracy_1st_raw",
    "sprop=story_accuracy_1st_infilled",
    "sprop=global_idx",
    "param=story_type",
    "param=num_stories_total",
    "param=max_sentences",
    "param=num_people",
    "param=num_moves",
    "param=num_rooms",
}

TARGET_MOVE_RE = re.compile(
    r"\bmoved the (.*?) to the (.*?), which is also located in", re.IGNORECASE
)
LINE_NUMBER_RE = re.compile(r"^\d+\s+")

EXPLORETOM_ASSUMPTIONS = (
    "ExploreToM assumptions:\n"
    "(1) Events occur in the numbered chronological order.\n"
    "(2) Track room entry and exit, object moves, observations, private "
    "communications, public communications, and secret witnessing literally "
    "as stated.\n"
    "(3) Private or secret information updates only the agents explicitly "
    "described as receiving or witnessing it. Do not give an agent knowledge "
    "of an event the story says they did not observe.\n"
    "(4) Answer with a container name rather than a choice letter."
)


def read_source_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing ExploreToM CSV columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def parse_source_boolean(value: str, field: str) -> bool:
    normalized = value.strip().upper()
    if normalized == "TRUE":
        return True
    if normalized == "FALSE":
        return False
    raise ValueError(f"Invalid {field} boolean: {value!r}")


def parse_question_properties(row: dict[str, str]) -> tuple[list[str], str, str]:
    try:
        value = ast.literal_eval(row["qprop=params"])
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid qprop=params: {row['qprop=params']!r}") from exc
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"Unexpected qprop=params: {value!r}")
    agents, obj, relation = value
    if not isinstance(agents, list) or not all(
        isinstance(agent, str) and agent for agent in agents
    ):
        raise ValueError(f"Invalid belief chain in qprop=params: {value!r}")
    if not isinstance(obj, str) or not obj:
        raise ValueError(f"Invalid object in qprop=params: {value!r}")
    if not isinstance(relation, str) or not relation:
        raise ValueError(f"Invalid relation in qprop=params: {value!r}")
    return agents, obj, relation


def target_container_candidates(numbered_story: str, obj: str) -> list[str]:
    candidates: list[str] = []
    for line in numbered_story.splitlines():
        event = LINE_NUMBER_RE.sub("", line, count=1)
        for moved_object, container in TARGET_MOVE_RE.findall(event):
            normalized_container = container.strip()
            if (
                moved_object.strip().lower() == obj.strip().lower()
                and normalized_container not in candidates
            ):
                candidates.append(normalized_container)
    if not candidates:
        raise ValueError(f"No target-object container moves found for {obj!r}")
    return candidates


def build_choices(
    candidates: list[str], answer: str, sample_id: str
) -> tuple[str, str]:
    if answer not in candidates:
        raise ValueError(f"Gold answer {answer!r} is absent from {candidates!r}")
    shuffled = list(candidates)
    random.Random(sample_id).shuffle(shuffled)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = ", ".join(
        f"{letters[index]}. {candidate}"
        for index, candidate in enumerate(shuffled)
    )
    return choices, letters[shuffled.index(answer)]


def build_answer_prompt(story: str, question: str, choices: str) -> str:
    return (
        "Read the story and answer the question.\n"
        f"Story:\n{story.rstrip()}\n\n"
        f"Question: {question.strip()}\n"
        f"Choices: {choices.strip()}\n"
    )


def build_base_prompt(story: str, question: str, choices: str) -> str:
    return (
        "Return exactly one JSON object and no markdown or extra text. Copy "
        "names exactly. belief_chain runs from the outermost thinker to the "
        "innermost thinker. belief_trace must list suffix chains from the "
        "innermost belief through the full queried chain. Locations, not "
        "choice letters, belong in belief_trace and answer.\n\n"
        f"{EXPLORETOM_ASSUMPTIONS}\n\n"
        f"Schema:\n{NESTED_BELIEF_SCHEMA}\n\n"
        f"Story:\n{story.rstrip()}\n\n"
        f"Question: {question.strip()}\n"
        f"Choices: {choices.strip()}\n"
    )


def source_task_sha256(row: dict[str, str]) -> str:
    task = {
        "story_structure": row["story_structure"].strip(),
        "question": row["question"].strip(),
        "expected_answer": row["expected_answer"].strip(),
    }
    return sha256_text(
        json.dumps(task, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )


def is_container_tom_row(row: dict[str, str]) -> bool:
    order = int(row["qprop=nth_order"])
    if order not in {1, 2}:
        return False
    agents, _, relation = parse_question_properties(row)
    if len(agents) != order:
        raise ValueError(
            f"Belief-chain length {len(agents)} != order {order}: {row['question']!r}"
        )
    return relation.startswith("container_location")


def convert_row(row: dict[str, str], source_row_index: int) -> dict[str, Any]:
    order = int(row["qprop=nth_order"])
    belief_chain, obj, relation = parse_question_properties(row)
    if order not in {1, 2} or len(belief_chain) != order:
        raise ValueError(f"Expected a first- or second-order ToM row: {source_row_index}")
    if not relation.startswith("container_location"):
        raise ValueError(f"Expected container_location relation: {relation!r}")

    story = number_story_events(row["story_structure"])
    question = row["question"].strip()
    answer = row["expected_answer"].strip()
    candidates = target_container_candidates(story, obj)
    sample_id = f"exploretom-hf:row={source_row_index:05d}"
    choices, answer_letter = build_choices(candidates, answer, sample_id)

    base_prompt = build_base_prompt(story, question, choices)
    process_prompt = build_grpo_prompt(base_prompt)
    if process_prompt.count(FEW_SHOT_MARKER) != 1:
        raise AssertionError("ExploreToM prompt must contain one few-shot block")
    if process_prompt.count(ORDER_TRACE_INSTRUCTION) != 1:
        raise AssertionError("ExploreToM prompt must contain one order instruction")

    non_unique = parse_source_boolean(
        row["qprop=non_unique_mental_state"], "qprop=non_unique_mental_state"
    )
    return {
        "dataset": DATASET_NAME,
        "source_dataset": SOURCE_DATASET,
        "source_split": "train",
        "split": "test",
        "sample_id": sample_id,
        "global_sample_id": sample_id,
        "source_row_index": source_row_index,
        "source_story_id": int(row["sprop=global_idx"]),
        "source_task_sha256": source_task_sha256(row),
        "question_order": order,
        "belief_chain": belief_chain,
        "object": obj,
        "relation": relation,
        "non_unique_mental_state": non_unique,
        "story": story,
        "story_structure": story,
        "story_variant": "story_structure",
        "story_event_count": len(story.splitlines()),
        "question": question,
        "choices": choices,
        "choice_count": len(candidates),
        "answer": answer,
        "gold_answer": answer,
        "answer_letter": answer_letter,
        "prompt": build_answer_prompt(story, question, choices),
        "process_prompt": process_prompt,
        "base_process_prompt_sha256": sha256_text(base_prompt),
        "process_prompt_sha256": sha256_text(process_prompt),
        "few_shot_version": FEW_SHOT_VERSION,
        "few_shot_count": FEW_SHOT_COUNT,
        "is_false_belief_story_1st": parse_source_boolean(
            row["sprop=is_false_belief_story_1st"],
            "sprop=is_false_belief_story_1st",
        ),
        "is_false_belief_story_1st_and_2nd": parse_source_boolean(
            row["sprop=is_false_belief_story_1st_and_2nd"],
            "sprop=is_false_belief_story_1st_and_2nd",
        ),
        "source_story_accuracy_1st_raw": float(
            row["sprop=story_accuracy_1st_raw"]
        ),
        "source_story_accuracy_1st_infilled": float(
            row["sprop=story_accuracy_1st_infilled"]
        ),
        "source_story_type": row["param=story_type"],
        "source_num_people": int(row["param=num_people"]),
        "source_num_moves": int(row["param=num_moves"]),
        "source_num_rooms": int(row["param=num_rooms"]),
    }


def prepare_exploretom_hf(
    input_path: Path,
    output_dir: Path,
    expected_source_count: int | None = EXPECTED_SOURCE_COUNT,
) -> dict[str, Any]:
    source_rows = read_source_rows(input_path)
    if expected_source_count is not None and len(source_rows) != expected_source_count:
        raise ValueError(
            f"Expected {expected_source_count} source rows, found {len(source_rows)}"
        )

    container_rows = [
        convert_row(row, index)
        for index, row in enumerate(source_rows)
        if is_container_tom_row(row)
    ]
    eligible = [
        row
        for row in container_rows
        if not row["non_unique_mental_state"] and row["choice_count"] >= 2
    ]
    seen_tasks: set[str] = set()
    test_rows: list[dict[str, Any]] = []
    duplicate_count = 0
    for row in eligible:
        task_key = row["source_task_sha256"]
        if task_key in seen_tasks:
            duplicate_count += 1
            continue
        seen_tasks.add(task_key)
        test_rows.append(row)
    order1 = [row for row in test_rows if row["question_order"] == 1]
    order2 = [row for row in test_rows if row["question_order"] == 2]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "all_container_questions": output_dir / "all_container_questions.jsonl",
        "test": output_dir / "test.jsonl",
        "order1_test": output_dir / "order1_test.jsonl",
        "order2_test": output_dir / "order2_test.jsonl",
        "manifest": output_dir / "manifest.json",
        "processed_readme": output_dir / "PROCESSED_README.md",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite processed files: {existing}")

    write_jsonl(output_paths["all_container_questions"], container_rows)
    write_jsonl(output_paths["test"], test_rows)
    write_jsonl(output_paths["order1_test"], order1)
    write_jsonl(output_paths["order2_test"], order2)

    manifest = {
        "name": "ExploreToM official sample answer-only benchmark",
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_URL,
        "source_split": "train",
        "license": LICENSE,
        "canonical_test_warning": (
            "The publisher says this sample is not the canonical ExploreToM test set."
        ),
        "source_file": str(input_path),
        "source_file_sha256": sha256_file(input_path),
        "source_count": len(source_rows),
        "container_tom_question_count": len(container_rows),
        "eligible_before_dedup_count": len(eligible),
        "removed_duplicate_count": duplicate_count,
        "test_count": len(test_rows),
        "test_order_counts": dict(
            sorted(Counter(str(row["question_order"]) for row in test_rows).items())
        ),
        "test_story_type_counts": dict(
            sorted(Counter(row["source_story_type"] for row in test_rows).items())
        ),
        "test_choice_count_counts": dict(
            sorted(Counter(str(row["choice_count"]) for row in test_rows).items())
        ),
        "selection": {
            "question_order": [1, 2],
            "relation_prefix": "container_location",
            "non_unique_mental_state": False,
            "minimum_choice_count": 2,
            "deduplicate_by": "story_structure + question + expected_answer",
            "story_variant": "story_structure",
        },
        "few_shot_version": FEW_SHOT_VERSION,
        "few_shot_count": FEW_SHOT_COUNT,
        "story_event_numbering": "one-based sentence order",
        "choice_generation": (
            "Distinct target-object containers mentioned in story order, then "
            "deterministically shuffled by sample ID."
        ),
        "order_trace_instruction": ORDER_TRACE_INSTRUCTION,
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in output_paths.items()
            if name not in {"manifest", "processed_readme"}
        },
    }
    output_paths["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    output_paths["processed_readme"].write_text(
        "# Processed ExploreToM files\n\n"
        "The upstream Hugging Face CSV and README are preserved in this "
        "directory. The publisher explicitly says this Llama-3.1-70B-targeted "
        "sample is not the canonical ExploreToM test set.\n\n"
        "`all_container_questions.jsonl` contains all first- and second-order "
        "container-location ToM rows. `test.jsonl` additionally requires a "
        "unique mental state, at least two target-container candidates, and "
        "removes exact duplicate tasks. `order1_test.jsonl` and "
        "`order2_test.jsonl` split that recommended set by ToM order. The "
        "official sample has no third- or fourth-order questions.\n\n"
        "The structured `story_structure` field is used as `story`, with one "
        "numbered event per line. Candidate choices are derived only from "
        "containers used for the queried object. Every `process_prompt` has "
        "the symbolic-v3 three-shot block and exactly one copy of: "
        f"`{ORDER_TRACE_INSTRUCTION}`\n\n"
        "Only final answers are supplied upstream, so no intermediate "
        "`process_target` is fabricated. Evaluate using "
        "`python -m rft.evaluate --answer-only`.\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-source-count", type=int, default=EXPECTED_SOURCE_COUNT
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_exploretom_hf(
        args.input,
        args.output_dir,
        expected_source_count=args.expected_source_count,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
