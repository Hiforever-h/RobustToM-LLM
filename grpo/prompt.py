"""Prompt construction shared by GRPO data preparation and audits."""

from __future__ import annotations

import re
from typing import Any, Mapping

from rft.prompt import format_chat_prompt
from scripts.add_symbolic_v3_few_shots import FEW_SHOT_MARKER, add_few_shots

ORDER_TRACE_INSTRUCTION = (
    "tom_order is exactly the number of names in belief_chain, not the number "
    "of story events. belief_trace contains exactly tom_order entries."
)
SCHEMA_MARKER = "\n\nSchema:\n"
NATURAL_COT_OUTPUT_INSTRUCTION = (
    "Work from the innermost person's belief outward through each nested-belief "
    "level. Output one block per level using Think N:, reasoning, and then "
    "State: <location>. Begin at Think 1 and increase N by one. Finish with "
    "Answer: <location>. Put every Think N:, State:, and Answer: marker at the "
    "beginning of a new line. Do not output JSON or belief_chain."
)


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


def build_natural_cot_prompt(source: Mapping[str, Any]) -> str:
    """Build the actor prompt for lightweight Think/State/Answer responses."""
    prompt = f"{build_natural_problem_prompt(source)} {NATURAL_COT_OUTPUT_INSTRUCTION}"
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
