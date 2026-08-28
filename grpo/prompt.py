"""Prompt construction shared by GRPO data preparation and audits."""

from __future__ import annotations

import re
from typing import Any, Mapping

from rft.prompt import NATURAL_COT_PROMPT_VERSION, format_chat_prompt
from scripts.add_symbolic_v3_few_shots import FEW_SHOT_MARKER, add_few_shots

ORDER_TRACE_INSTRUCTION = (
    "tom_order is exactly the number of names in belief_chain, not the number "
    "of story events. belief_trace contains exactly tom_order entries."
)
SCHEMA_MARKER = "\n\nSchema:\n"
def build_grpo_prompt(process_prompt: str) -> str:
    """Add the v3 demonstrations and explicit order/trace cardinality rule."""
    if not isinstance(process_prompt, str) or not process_prompt.strip():
        raise ValueError("process_prompt must be a non-empty string")
    if FEW_SHOT_MARKER in process_prompt:
        raise ValueError("Expected the raw v3 prompt without few-shot demonstrations")
    if ORDER_TRACE_INSTRUCTION in process_prompt:
        raise ValueError(
            "Expected the raw v3 prompt without the GRPO order instruction"
        )
    if process_prompt.count(SCHEMA_MARKER) != 1:
        raise ValueError("Expected exactly one schema marker in the v3 process prompt")

    clarified = process_prompt.replace(
        SCHEMA_MARKER,
        f" {ORDER_TRACE_INSTRUCTION}{SCHEMA_MARKER}",
        1,
    )
    augmented = add_few_shots(clarified)
    if augmented.count(ORDER_TRACE_INSTRUCTION) != 1:
        raise AssertionError("The order/trace instruction must appear exactly once")
    if augmented.count(FEW_SHOT_MARKER) != 1:
        raise AssertionError("The few-shot block must appear exactly once")
    return augmented


def clean_numbered_story(story: str) -> str:
    """Remove event-number prefixes and collapse a story to one continuous line."""
    if not isinstance(story, str) or not story.strip():
        raise ValueError("story must be a non-empty string")
    without_numbers = re.sub(r"(?m)^\s*\d+\s+", "", story)
    return re.sub(r"\s+", " ", without_numbers).strip()


def build_natural_problem_prompt(source: Mapping[str, Any]) -> str:
    """Build the Judge task text without actor output-format instructions."""
    story = clean_numbered_story(source.get("story", ""))
    question = source.get("question")
    choices = source.get("choices")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(choices, str) or not choices.strip():
        raise ValueError("choices must be a non-empty string")
    prompt = (
        f"Read the story and answer the question. Story: {story} "
        f"Question: {question.strip()} Choices: {choices.strip()}"
    )
    if "\n" in prompt or "\r" in prompt:
        raise AssertionError("Natural problem prompt must be a single line")
    return prompt


def _natural_order_and_chain(source: Mapping[str, Any]) -> tuple[int, list[str]]:
    target = source.get("process_target")
    if not isinstance(target, Mapping):
        raise ValueError("process_target must be a mapping")
    order = target.get("tom_order")
    chain = target.get("belief_chain")
    if type(order) is not int or order < 1:
        raise ValueError("process_target.tom_order must be a positive integer")
    if (
        not isinstance(chain, list)
        or len(chain) != order
        or not all(isinstance(name, str) and name.strip() for name in chain)
    ):
        raise ValueError("belief_chain must contain exactly tom_order names")
    question_order = source.get("question_order")
    if question_order is not None and question_order != order:
        raise ValueError("question_order and process_target.tom_order disagree")
    return order, [name.strip() for name in chain]


def _possessive(name: str) -> str:
    return f"{name}'" if name.lower().endswith("s") else f"{name}'s"


def _belief_description(chain: list[str]) -> str:
    description = f"{_possessive(chain[-1])} own belief"
    for outer_name in reversed(chain[:-1]):
        description = f"{_possessive(outer_name)} belief about {description}"
    return description


def build_natural_cot_prompt(source: Mapping[str, Any]) -> str:
    """Build an exact-order actor prompt without leaking gold locations."""
    order, belief_chain = _natural_order_and_chain(source)
    level_chains = [belief_chain[-index:] for index in range(1, order + 1)]
    level_rules = [
        f"- Think {index} must reason about {_belief_description(level_chain)}."
        for index, level_chain in enumerate(level_chains, start=1)
    ]
    output_lines: list[str] = []
    for index, level_chain in enumerate(level_chains, start=1):
        output_lines.extend(
            (
                f"Think {index}:",
                f"<reasoning for {_belief_description(level_chain)}>",
                "State: <location>",
            )
        )
    output_lines.append(f"Answer: <same location as State from Think {order}>")
    block_word = "block" if order == 1 else "blocks"
    level_word = "level" if order == 1 else "levels"
    prompt = "\n".join(
        (
            build_natural_problem_prompt(source),
            "",
            "Reasoning rules:",
            f"- This question has exactly {order} nested-belief {level_word}.",
            "- The belief chain from outermost to innermost is: "
            + " -> ".join(belief_chain)
            + ".",
            f"- Output exactly {order} Think/State {block_word}. Do not create a "
            "block for each story event or for people outside the belief chain.",
            *level_rules,
            "- A private observation updates only the named observer's own belief.",
            "- An unseen move updates reality but does not update any person's belief.",
            "- A jointly observed move is common knowledge only within the named group.",
            "- Each State must be one location from Choices; output the location, not its letter.",
            f"- Answer must repeat the State from Think {order}.",
            f"- Stop immediately after Answer. Do not output JSON, markdown, extra text, or Think {order + 1}.",
            "",
            "Required output format:",
            *output_lines,
        )
    )
    if SCHEMA_MARKER.strip() in prompt or FEW_SHOT_MARKER in prompt:
        raise AssertionError("Natural-CoT prompt contains a legacy JSON instruction")
    return prompt


def chat_prompt_token_ids(tokenizer: Any, process_prompt: str) -> list[int]:
    """Tokenize the exact single-user chat prompt consumed by verl."""
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": process_prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def formatted_chat_prompt(tokenizer: Any, process_prompt: str) -> str:
    """Expose the RFT formatting path for prompt-parity tests."""
    return format_chat_prompt(tokenizer, process_prompt)
