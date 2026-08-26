#!/usr/bin/env python3
"""Convert the official Hugging Face Hi-ToM data to RFT answer-only JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from grpo.prompt import ORDER_TRACE_INSTRUCTION, build_grpo_prompt
from rft.common import sha256_file, sha256_text, write_jsonl
from rft.prepare_hitom_eval import build_base_prompt, number_story_events
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


DATASET_NAME = "Hi-ToM"
SOURCE_DATASET = "Hi-ToM/Hi-ToM_Dataset"
SOURCE_URL = "https://huggingface.co/datasets/Hi-ToM/Hi-ToM_Dataset"
DEFAULT_INPUT = Path("data/Hi-ToM/Hi-ToM_data.json")
DEFAULT_OUTPUT_DIR = Path("data/Hi-ToM")
EXPECTED_SOURCE_COUNT = 1200
CANONICAL_PROMPTING_TYPE = "CoTP"

REQUIRED_FIELDS = {
    "prompting_type",
    "deception",
    "story_length",
    "question_order",
    "sample_id",
    "story",
    "question",
    "choices",
    "answer",
}


def read_source_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data")
    if not isinstance(payload, list):
        raise ValueError("Hi-ToM JSON must be a list or an object containing data")

    rows: list[dict[str, Any]] = []
    for index, value in enumerate(payload):
        if not isinstance(value, dict):
            raise ValueError(f"Hi-ToM row {index} is not an object")
        missing = REQUIRED_FIELDS - set(value)
        if missing:
            raise ValueError(f"Hi-ToM row {index} is missing: {sorted(missing)}")
        rows.append(value)
    return rows


def source_task_key(row: dict[str, Any]) -> str:
    canonical = {
        "story": number_story_events(str(row["story"])),
        "question": str(row["question"]).strip(),
        "choices": str(row["choices"]).strip(),
    }
    return sha256_text(
        json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )


def find_label_conflicts(
    rows: list[dict[str, Any]],
) -> tuple[set[str], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[source_task_key(row)].append(row)

    conflict_keys: set[str] = set()
    conflict_rows: list[dict[str, Any]] = []
    for task_key, group in groups.items():
        prompting_types = {str(row["prompting_type"]) for row in group}
        if len(group) != 2 or prompting_types != {"CoTP", "VP"}:
            raise ValueError(
                "Every canonical Hi-ToM task must have one CoTP and one VP row: "
                f"task={task_key}, rows={len(group)}, types={sorted(prompting_types)}"
            )
        answers = {str(row["answer"]).strip() for row in group}
        if len(answers) == 1:
            continue
        conflict_keys.add(task_key)
        by_type = {str(row["prompting_type"]): row for row in group}
        conflict_rows.append(
            {
                "source_task_sha256": task_key,
                "question_order": int(group[0]["question_order"]),
                "question": str(group[0]["question"]).strip(),
                "cotp_source_sample_id": int(by_type["CoTP"]["sample_id"]),
                "cotp_answer": str(by_type["CoTP"]["answer"]).strip(),
                "vp_source_sample_id": int(by_type["VP"]["sample_id"]),
                "vp_answer": str(by_type["VP"]["answer"]).strip(),
            }
        )
    return conflict_keys, conflict_rows


def build_answer_prompt(story: str, question: str, choices: str) -> str:
    return (
        "Read the story and answer the question.\n"
        f"Story:\n{story.rstrip()}\n\n"
        f"Question: {question.strip()}\n"
        f"Choices: {choices.strip()}\n"
    )


def stable_sample_id(row: dict[str, Any]) -> str:
    prompting_type = str(row["prompting_type"]).strip().lower()
    source_sample_id = int(row["sample_id"])
    return f"hi-tom-hf:{prompting_type}:sample={source_sample_id:04d}"


def convert_row(
    row: dict[str, Any], conflict_keys: set[str]
) -> dict[str, Any]:
    prompting_type = str(row["prompting_type"]).strip()
    if prompting_type not in {"CoTP", "VP"}:
        raise ValueError(f"Unknown Hi-ToM prompting_type: {prompting_type!r}")

    order = int(row["question_order"])
    if order not in {0, 1, 2, 3, 4}:
        raise ValueError(f"Unsupported Hi-ToM question order: {order}")

    story = number_story_events(str(row["story"]))
    question = str(row["question"]).strip()
    choices = str(row["choices"]).strip()
    answer = str(row["answer"]).strip()
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

    task_key = source_task_key(row)
    sample_id = stable_sample_id(row)
    return {
        "dataset": DATASET_NAME,
        "source_dataset": SOURCE_DATASET,
        "source_split": "test",
        "split": "test",
        "sample_id": sample_id,
        "global_sample_id": sample_id,
        "source_task_sha256": task_key,
        "source_sample_id": int(row["sample_id"]),
        "source_prompting_type": prompting_type,
        "source_label_conflict": task_key in conflict_keys,
        "deception": bool(row["deception"]),
        "story_length": int(row["story_length"]),
        "question_order": order,
        "belief_chain": belief_chain,
        "object": obj,
        "story": story,
        "story_event_count": len(story.splitlines()),
        "question": question,
        "choices": choices,
        "answer": answer,
        "gold_answer": answer,
        "prompt": build_answer_prompt(story, question, choices),
        "process_prompt": process_prompt,
        "base_process_prompt_sha256": sha256_text(base_prompt),
        "process_prompt_sha256": sha256_text(process_prompt),
        "few_shot_version": FEW_SHOT_VERSION,
        "few_shot_count": FEW_SHOT_COUNT,
    }


def prepare_hitom_hf(
    input_path: Path,
    output_dir: Path,
    expected_source_count: int | None = EXPECTED_SOURCE_COUNT,
) -> dict[str, Any]:
    source_rows = read_source_rows(input_path)
    if expected_source_count is not None and len(source_rows) != expected_source_count:
        raise ValueError(
            f"Expected {expected_source_count} source rows, found {len(source_rows)}"
        )
    source_ids = [int(row["sample_id"]) for row in source_rows]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Hi-ToM source sample_id values must be unique")

    conflict_keys, conflicts = find_label_conflicts(source_rows)
    converted = [convert_row(row, conflict_keys) for row in source_rows]
    canonical = [
        row
        for row in converted
        if row["source_prompting_type"] == CANONICAL_PROMPTING_TYPE
    ]
    consistent = [row for row in canonical if not row["source_label_conflict"]]
    order4 = [row for row in canonical if row["question_order"] == 4]
    order4_consistent = [
        row for row in order4 if not row["source_label_conflict"]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "all_prompt_variants": output_dir / "all_prompt_variants.jsonl",
        "test": output_dir / "test.jsonl",
        "consistent_test": output_dir / "consistent_test.jsonl",
        "order4_test": output_dir / "order4_test.jsonl",
        "order4_consistent_test": output_dir / "order4_consistent_test.jsonl",
        "label_conflicts": output_dir / "label_conflicts.jsonl",
        "manifest": output_dir / "manifest.json",
        "processed_readme": output_dir / "PROCESSED_README.md",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite processed files: {existing}")

    write_jsonl(output_paths["all_prompt_variants"], converted)
    write_jsonl(output_paths["test"], canonical)
    write_jsonl(output_paths["consistent_test"], consistent)
    write_jsonl(output_paths["order4_test"], order4)
    write_jsonl(output_paths["order4_consistent_test"], order4_consistent)
    write_jsonl(output_paths["label_conflicts"], conflicts)

    manifest = {
        "name": "Official Hi-ToM answer-only benchmark with symbolic-v3 prompts",
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_URL,
        "source_file": str(input_path),
        "source_file_sha256": sha256_file(input_path),
        "source_count": len(source_rows),
        "all_prompt_variants_count": len(converted),
        "canonical_prompting_type": CANONICAL_PROMPTING_TYPE,
        "test_count": len(canonical),
        "consistent_test_count": len(consistent),
        "order4_test_count": len(order4),
        "order4_consistent_test_count": len(order4_consistent),
        "order4_source_label_conflict_count": len(order4) - len(order4_consistent),
        "source_label_conflict_count": len(conflicts),
        "test_order_counts": dict(
            sorted(Counter(str(row["question_order"]) for row in canonical).items())
        ),
        "order4_deception_counts": dict(
            sorted(Counter(str(row["deception"]).lower() for row in order4).items())
        ),
        "order4_story_length_counts": dict(
            sorted(Counter(str(row["story_length"]) for row in order4).items())
        ),
        "few_shot_version": FEW_SHOT_VERSION,
        "few_shot_count": FEW_SHOT_COUNT,
        "story_event_numbering": "one-based source event order",
        "order_trace_instruction": ORDER_TRACE_INSTRUCTION,
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
            if name not in {"manifest", "processed_readme"}
        },
    }
    output_paths["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    output_paths["processed_readme"].write_text(
        "# Processed Hi-ToM files\n\n"
        "The upstream Hugging Face files in this directory are preserved. "
        "`all_prompt_variants.jsonl` converts all 1,200 source rows. "
        "`test.jsonl` selects the 600 CoTP rows so the duplicated VP prompt "
        "variant does not double-weight the same tasks. `order4_test.jsonl` "
        "contains the 120 fourth-order rows from that canonical selection. "
        "`consistent_test.jsonl` and `order4_consistent_test.jsonl` remove "
        "tasks whose two upstream answer labels disagree.\n\n"
        "The source contains 138 CoTP/VP task pairs with conflicting answer "
        "labels; `label_conflicts.jsonl` records them without changing either "
        "upstream label. The recommended files retain the CoTP label and expose "
        "`source_label_conflict` on every record. Hi-ToM supplies only final "
        "answers, so no intermediate `process_target` is fabricated and "
        "evaluation must use `python -m rft.evaluate --answer-only`.\n\n"
        "Every story event has a one-based line number. Every `process_prompt` "
        "contains the symbolic-v3 three-shot block and exactly one copy of: "
        f"`{ORDER_TRACE_INSTRUCTION}`\n",
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
    manifest = prepare_hitom_hf(
        args.input,
        args.output_dir,
        expected_source_count=args.expected_source_count,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
