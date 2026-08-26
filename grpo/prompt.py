"""Prompt construction shared by GRPO data preparation and audits."""

from __future__ import annotations

from typing import Any

from rft.prompt import format_chat_prompt
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
        raise ValueError("Expected the raw v3 prompt without the GRPO order instruction")
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

