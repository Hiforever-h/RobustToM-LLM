#!/usr/bin/env python3
"""Build the ToM-RL paper evaluation set for open-ended natural-CoT testing.

The upstream paper CSV contains final answers but no process targets or answer
choices.  This converter keeps only positive-order Theory-of-Mind questions,
rebuilds a prompt compatible with the project's Think/State/Answer protocol,
and deliberately supports answer-only evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from grpo.prompt import clean_numbered_story
from rft.common import sha256_file, sha256_text, write_jsonl


DATASET_NAME = "ToM-RL-paper-eval-open-ended"
PROMPT_VERSION = "natural-cot-think-state-v3-compact-open-ended"
DEFAULT_INPUT = Path("eval_tom/tom_eval_datasets.csv")
DEFAULT_OUTPUT_DIR = Path("data/tomrl_paper_eval_open_ended")

SOURCE_OUTPUT_NAMES = {
    "hi_tom": "hi_tom.jsonl",
    "4th-order-ToM": "order4.jsonl",
    "tomi": "tomi.jsonl",
    "explore_tom_structured": "explore_tom_structured.jsonl",
    "explore_tom_infilled": "explore_tom_infilled.jsonl",
}

# These distributions are a guardrail for the checked-in upstream paper CSV.
EXPECTED_ORDER_COUNTS = {
    "hi_tom": {0: 120, 1: 120, 2: 120, 3: 120, 4: 120},
    "4th-order-ToM": {4: 600},
    "tomi": {0: 1998, 1: 1998, 2: 1998},
    "explore_tom_structured": {0: 178, 1: 118, 2: 770},
    "explore_tom_infilled": {0: 178, 1: 118, 2: 770},
}

NAME = r"[A-Z][A-Za-z-]*"
THINK_RE = re.compile(r"\bthink(?:s)?\b", re.IGNORECASE)
TERMINAL_SEARCH_RES = (
    re.compile(rf"\b(?:will|would)\s+{NAME}\s+(?:look|search)\b"),
    re.compile(rf"\b{NAME}\s+(?:will|would)\s+(?:look|search)\b"),
    re.compile(rf"\b{NAME}\s+(?:looks|searches)\b"),
)

OPEN_ENDED_FORMAT_INSTRUCTION = (
    "Output requirements: In these instructions, N means the theory-of-mind (ToM) "
    "order implied by the question—that is, the number of nested belief levels in "
    "the question. Infer N from the question itself. Give a natural-language "
    "chain-of-thought answer with exactly N numbered reasoning blocks. Number them "
    "consecutively starting at 1, using `Think 1:`, `Think 2:`, and so on; use actual "
    "integers in the output, never the letter N. Each block must begin with its Think "
    "marker at the beginning of a new line, contain non-empty reasoning, and end on "
    "a new line with exactly one `State:` followed by the exact location supported "
    "by the story. After the final block, output exactly one `Answer:` on a new line, "
    "repeating the final State. Put every marker at the beginning of a new line. Do "
    "not output JSON, Markdown, choice letters, extra blocks, or text after Answer."
)


def read_source_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"data_source", "story", "question", "answer"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing ToM-RL CSV columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def infer_question_order(question: str) -> int:
    """Infer nesting depth from the controlled question grammar.

    Hi-ToM expresses every belief level with ``think(s)``.  ToMi and
    ExploreToM may express the innermost level as an agent looking/searching,
    so one terminal search clause contributes one additional level.
    """
    normalized = " ".join(question.split())
    if not normalized:
        raise ValueError("Question must be non-empty")
    think_count = len(THINK_RE.findall(normalized))
    has_terminal_search = any(
        pattern.search(normalized) for pattern in TERMINAL_SEARCH_RES
    )
    order = think_count + int(has_terminal_search)
    if order > 4:
        raise ValueError(f"Unsupported inferred question order {order}: {question!r}")
    return order


def build_open_ended_problem_prompt(story: str, question: str) -> str:
    clean_story = clean_numbered_story(story)
    clean_question = " ".join(question.split())
    if not clean_question:
        raise ValueError("Question must be non-empty")
    prompt = (
        f"Read the story and answer the question. Story: {clean_story} "
        f"Question: {clean_question}"
    )
    if "\n" in prompt or "\r" in prompt:
        raise AssertionError("Open-ended problem prompt must be a single line")
    return prompt


def build_open_ended_process_prompt(story: str, question: str) -> str:
    return (
        f"{build_open_ended_problem_prompt(story, question)}\n\n"
        f"{OPEN_ENDED_FORMAT_INSTRUCTION}"
    )


def stable_sample_id(source_dataset: str, source_index: int) -> str:
    source = source_dataset.lower().replace("_", "-").replace(" ", "-")
    return f"tomrl-paper:{source}:{source_index:06d}"


def convert_row(
    row: dict[str, str], csv_row_index: int, source_index: int
) -> dict[str, Any]:
    source_dataset = row["data_source"].strip()
    if source_dataset not in SOURCE_OUTPUT_NAMES:
        raise ValueError(f"Unknown ToM-RL data_source: {source_dataset!r}")
    story = row["story"].strip()
    question = row["question"].strip()
    answer = row["answer"].strip()
    if not story or not question or not answer:
        raise ValueError(f"Empty field in source CSV row {csv_row_index}")

    question_order = infer_question_order(question)
    if source_dataset == "4th-order-ToM" and question_order != 4:
        raise ValueError(
            "A 4th-order-ToM row did not parse as order 4: "
            f"row={csv_row_index}, question={question!r}"
        )
    sample_id = stable_sample_id(source_dataset, source_index)
    process_prompt = build_open_ended_process_prompt(story, question)
    if "Choices:" in process_prompt:
        raise AssertionError("Open-ended prompt unexpectedly contains Choices")
    if "<|im_start|>" in process_prompt or "<think>" in process_prompt:
        raise AssertionError("Open-ended prompt contains a legacy chat protocol")

    return {
        "dataset": DATASET_NAME,
        "source_dataset": source_dataset,
        "source_split": "test",
        "split": "test",
        "global_sample_id": sample_id,
        "sample_id": sample_id,
        "source_csv_row_index": csv_row_index,
        "source_dataset_index": source_index,
        "question_order": question_order,
        "story": story,
        "question": question,
        "answer": answer,
        "gold_answer": answer,
        "process_prompt_version": PROMPT_VERSION,
        "process_prompt": process_prompt,
        "process_prompt_sha256": sha256_text(process_prompt),
        "evaluation_mode": "answer-only",
    }


def _nested_counts(
    rows: list[dict[str, Any]], include_zero: bool = True
) -> dict[str, dict[int, int]]:
    counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        order = int(row["question_order"])
        if include_zero or order >= 1:
            counts[str(row["source_dataset"])][order] += 1
    return {
        source: dict(sorted(source_counts.items()))
        for source, source_counts in sorted(counts.items())
    }


def prepare_tomrl_paper_eval(
    input_path: Path,
    output_dir: Path,
    *,
    strict_expected_counts: bool = True,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    source_rows = read_source_rows(input_path)
    source_indices: Counter[str] = Counter()
    converted: list[dict[str, Any]] = []
    for csv_row_index, source_row in enumerate(source_rows):
        source = source_row["data_source"].strip()
        source_index = source_indices[source]
        source_indices[source] += 1
        converted.append(
            convert_row(source_row, csv_row_index, source_index)
        )

    source_order_counts = _nested_counts(converted)
    if strict_expected_counts and source_order_counts != EXPECTED_ORDER_COUNTS:
        raise ValueError(
            "Unexpected source/order distribution; the input may not be the "
            "checked-in ToM-RL paper CSV. "
            f"expected={EXPECTED_ORDER_COUNTS}, actual={source_order_counts}"
        )

    filtered = [row for row in converted if row["question_order"] >= 1]
    sample_ids = [row["global_sample_id"] for row in filtered]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Generated global_sample_id values must be unique")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_paths: dict[str, Path] = {"test": output_dir / "test.jsonl"}
    write_jsonl(output_paths["test"], filtered)
    for source, filename in SOURCE_OUTPUT_NAMES.items():
        path = output_dir / filename
        output_paths[source] = path
        write_jsonl(
            path, [row for row in filtered if row["source_dataset"] == source]
        )

    manifest = {
        "name": DATASET_NAME,
        "source_file": str(input_path),
        "source_file_sha256": sha256_file(input_path),
        "source_count": len(converted),
        "filter": {"question_order": ">= 1"},
        "excluded_order0_count": len(converted) - len(filtered),
        "test_count": len(filtered),
        "prompt_version": PROMPT_VERSION,
        "choices_in_prompt": False,
        "process_targets_available": False,
        "evaluation_mode": "answer-only",
        "source_order_counts_before_filter": source_order_counts,
        "source_order_counts_after_filter": _nested_counts(filtered),
        "outputs": {
            name: {
                "path": str(path),
                "count": (
                    len(filtered)
                    if name == "test"
                    else sum(
                        row["source_dataset"] == name for row in filtered
                    )
                ),
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# ToM-RL paper evaluation: open-ended positive-order split\n\n"
        "This directory is generated from `eval_tom/tom_eval_datasets.csv`. "
        "It contains only test rows whose inferred `question_order >= 1`; "
        "no training rows or synthetic validation rows are included.\n\n"
        "Prompts use the project's natural `Think/State/Answer` protocol but "
        "omit `Choices`. The corresponding State rule asks for the exact "
        "location supported by the story. The upstream data supplies final "
        "answers but no intermediate belief-state targets, so evaluation must "
        "use `python -m rft.evaluate --answer-only`.\n\n"
        "`test.jsonl` contains all retained rows. The other JSONL files split "
        "the same rows by the five source benchmarks reported by ToM-RL.\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-strict-expected-counts",
        action="store_true",
        help="Allow an input whose source/order counts differ from the paper CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_tomrl_paper_eval(
        args.input,
        args.output_dir,
        strict_expected_counts=not args.no_strict_expected_counts,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
