#!/usr/bin/env python3
"""Build answer-and-support-event prompts for privileged-teacher evaluation.

The resulting JSONL remains compatible with ``python -m rft.generate``.  Its
``process_prompt`` is the teacher prompt, while ``student_process_prompt`` keeps
the original compact prompt for audit and later OPSD work.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from rft.common import read_jsonl, sha256_file, write_jsonl
from rft.prompt import (
    COMPACT_NATURAL_COT_PROMPT_VERSION,
    PRIVILEGED_TEACHER_PROMPT_VERSION,
    build_compact_process_prompt,
)


PRIVILEGED_PROMPT_VERSION = PRIVILEGED_TEACHER_PROMPT_VERSION
PRIVILEGED_REFERENCE_VERSION = "answer-support-events-v1"
DEFAULT_INPUT = Path(
    "data/counterfactual_process_reward_v4_natural_compact/test.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/counterfactual_process_reward_v4_natural_compact_teacher_order4_ood"
)
OUTPUT_REQUIREMENTS_MARKER = "\n\nOutput requirements:"


def _sample_name(record: Mapping[str, Any]) -> str:
    return str(record.get("global_sample_id", "<unknown>"))


def _target(record: Mapping[str, Any]) -> Mapping[str, Any]:
    target = record.get("process_target")
    if not isinstance(target, Mapping):
        raise ValueError(f"Missing process_target: {_sample_name(record)}")
    trace = target.get("belief_trace")
    order = target.get("tom_order")
    if (
        target.get("reasoning_mode") != "nested_belief"
        or type(order) is not int
        or order < 1
        or not isinstance(trace, list)
        or len(trace) != order
    ):
        raise ValueError(f"Invalid nested-belief process target: {_sample_name(record)}")
    if target.get("answer") != record.get("answer"):
        raise ValueError(f"Answer/target mismatch: {_sample_name(record)}")
    return target


def _event_updates_chain(event: Mapping[str, Any], chain: Sequence[str]) -> bool:
    visibility = event.get("visibility")
    observers = event.get("observers")
    if not isinstance(observers, list):
        return False
    if visibility == "joint":
        return set(chain).issubset(set(observers))
    if visibility == "private":
        return len(chain) == 1 and chain[0] in observers
    return False


def derive_support_events(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the last event that updated each queried suffix belief chain.

    Events are deduplicated and returned in story chronology.  Gold locations
    are used only to validate the symbolic derivation, not to select by target
    location.
    """
    target = _target(record)
    events = record.get("latent_events")
    if not isinstance(events, list) or not events:
        raise ValueError(f"Missing latent_events: {_sample_name(record)}")
    if not all(isinstance(event, Mapping) for event in events):
        raise ValueError(f"Invalid latent_events: {_sample_name(record)}")

    selected_ids: set[str] = set()
    for step in target["belief_trace"]:
        if not isinstance(step, Mapping):
            raise ValueError(f"Invalid belief-trace step: {_sample_name(record)}")
        chain = step.get("belief_chain")
        if not isinstance(chain, list) or not chain:
            raise ValueError(f"Invalid belief chain: {_sample_name(record)}")
        support = next(
            (
                event
                for event in reversed(events)
                if _event_updates_chain(event, chain)
            ),
            None,
        )
        if support is None:
            raise ValueError(
                f"No support event for chain {chain}: {_sample_name(record)}"
            )
        if support.get("to_location") != step.get("location"):
            raise ValueError(
                f"Support event/target location mismatch for chain {chain}: "
                f"{_sample_name(record)}"
            )
        event_id = support.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"Support event has no ID: {_sample_name(record)}")
        selected_ids.add(event_id)

    selected = [dict(event) for event in events if event.get("event_id") in selected_ids]
    if len(selected) != len(target["belief_trace"]):
        raise ValueError(
            "Expected one distinct support event per belief level: "
            f"{_sample_name(record)}"
        )
    answer_event_id = record.get("answer_event_id")
    if answer_event_id is not None:
        final_chain = target["belief_trace"][-1]["belief_chain"]
        final_support = next(
            event
            for event in reversed(events)
            if _event_updates_chain(event, final_chain)
        )
        if final_support.get("event_id") != answer_event_id:
            raise ValueError(f"answer_event_id mismatch: {_sample_name(record)}")
    return selected


def _join_names(names: Sequence[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def render_support_event(event: Mapping[str, Any]) -> str:
    """Render symbolic evidence without exposing generator-only labels or IDs."""
    object_name = str(event.get("object", "the object"))
    source = event.get("from_location")
    destination = event.get("to_location")
    if not isinstance(destination, str) or not destination:
        raise ValueError("Support event has no destination")
    movement = (
        f"the initial placement of the {object_name} in {destination}"
        if source is None
        else f"the {object_name} move from {source} to {destination}"
    )
    observers = event.get("observers")
    if not isinstance(observers, list):
        raise ValueError("Support event has invalid observers")
    visibility = event.get("visibility")
    if visibility == "joint":
        return (
            f"{_join_names([str(name) for name in observers])} jointly observed "
            f"{movement}; every named observer knew the full audience."
        )
    if visibility == "private":
        return (
            f"{_join_names([str(name) for name in observers])} privately observed "
            f"{movement}; nobody else was informed of this private observation."
        )
    if visibility == "hidden":
        return f"Nobody observed {movement}; it was unseen and unreported."
    raise ValueError(f"Unsupported event visibility: {visibility!r}")


def build_privileged_context(
    record: Mapping[str, Any], support_events: Sequence[Mapping[str, Any]]
) -> str:
    answer = record.get("answer")
    if not isinstance(answer, str) or not answer:
        raise ValueError(f"Missing answer: {_sample_name(record)}")
    bullets = "\n".join(f"- {render_support_event(event)}" for event in support_events)
    return (
        "Privileged reference for the teacher:\n"
        f"The verified final answer is {answer}.\n\n"
        "The following verified events are sufficient for solving the nested-belief "
        "question. They are listed in chronological order:\n\n"
        f"{bullets}\n\n"
        "Use the verified answer and events to determine why the answer is correct. "
        "Reason through the nested beliefs from the innermost thinker to the "
        "outermost thinker. Do not refer to these facts as hints or privileged "
        "information in the response."
    )


def build_privileged_teacher_prompt(
    record: Mapping[str, Any], support_events: Sequence[Mapping[str, Any]]
) -> str:
    compact_prompt = build_compact_process_prompt(record)
    if compact_prompt.count(OUTPUT_REQUIREMENTS_MARKER) != 1:
        raise ValueError(
            f"Expected one output-requirements marker: {_sample_name(record)}"
        )
    problem, requirements = compact_prompt.split(
        OUTPUT_REQUIREMENTS_MARKER, maxsplit=1
    )
    context = build_privileged_context(record, support_events)
    return f"{problem}\n\n{context}{OUTPUT_REQUIREMENTS_MARKER}{requirements}"


def build_privileged_teacher_row(source: Mapping[str, Any]) -> dict[str, Any]:
    if source.get("process_prompt_version") != COMPACT_NATURAL_COT_PROMPT_VERSION:
        raise ValueError(
            "Expected compact natural-CoT source prompt: "
            f"{_sample_name(source)}"
        )
    support_events = derive_support_events(source)
    original_prompt = source.get("process_prompt")
    if not isinstance(original_prompt, str) or not original_prompt:
        raise ValueError(f"Missing process_prompt: {_sample_name(source)}")
    context = build_privileged_context(source, support_events)
    row = dict(source)
    row["student_process_prompt"] = original_prompt
    row["student_process_prompt_version"] = source["process_prompt_version"]
    row["process_prompt"] = build_privileged_teacher_prompt(source, support_events)
    row["process_prompt_version"] = PRIVILEGED_PROMPT_VERSION
    row["privileged_reference_version"] = PRIVILEGED_REFERENCE_VERSION
    row["privileged_context"] = context
    row["privileged_reference"] = {
        "answer": source["answer"],
        "support_event_ids": [event["event_id"] for event in support_events],
        "support_events": support_events,
    }
    return row


def build_dataset(input_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    source_rows = read_jsonl(input_path)
    rows = [build_privileged_teacher_row(row) for row in source_rows]
    if not rows:
        raise ValueError("Input dataset is empty")
    if any(row.get("question_order") != 4 for row in rows):
        raise ValueError("This builder expects a pure order-4 OOD dataset")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "test.jsonl"
    write_jsonl(output_path, rows)
    support_counts = Counter(
        len(row["privileged_reference"]["support_events"]) for row in rows
    )
    critical_in_support = Counter()
    for row in rows:
        support_ids = set(row["privileged_reference"]["support_event_ids"])
        critical_in_support[
            str(row.get("intervention_type"))
        ] += row.get("critical_event_id") in support_ids
    manifest = {
        "name": "RobustToM order-4 OOD privileged-teacher evaluation data",
        "prompt_version": PRIVILEGED_PROMPT_VERSION,
        "privileged_reference_version": PRIVILEGED_REFERENCE_VERSION,
        "source_file": str(input_path),
        "output_file": str(output_path),
        "count": len(rows),
        "pair_count": len({row["global_pair_id"] for row in rows}),
        "order_counts": dict(Counter(str(row["question_order"]) for row in rows)),
        "intervention_counts": dict(
            Counter(str(row["intervention_type"]) for row in rows)
        ),
        "support_event_count_distribution": {
            str(key): value for key, value in sorted(support_counts.items())
        },
        "critical_event_is_support_count": dict(critical_in_support),
        "gold_answer_exposed_to_teacher": True,
        "student_prompt_preserved": True,
        "generator_labels_exposed_in_prompt": False,
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
