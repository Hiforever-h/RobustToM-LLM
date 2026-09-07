"""Single source of truth for model-facing chat prompt construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


NATURAL_COT_PROMPT_VERSION = "natural-cot-think-state-v2-exact-order"
COMPACT_NATURAL_COT_PROMPT_VERSION = "natural-cot-think-state-v3-compact"
PRIVILEGED_TEACHER_PROMPT_VERSION = (
    "natural-cot-think-state-v3-compact-privileged-answer-support-events-v1"
)


def build_compact_process_prompt(record: Mapping[str, Any]) -> str:
    """Build a question plus concise response-format instructions.

    The compact prompt is rebuilt from ``judge_prompt`` instead of trimming the
    actor prompt. This guarantees that the v2 reasoning rules and empty output
    scaffold are not retained accidentally.
    """
    sample = record.get("global_sample_id", "<unknown>")
    judge_prompt = record.get("judge_prompt")
    if not isinstance(judge_prompt, str) or not judge_prompt.strip():
        raise ValueError(f"Missing judge_prompt for compact prompt: {sample}")

    format_instruction = (
        "Output requirements: In these instructions, N means the theory-of-mind (ToM) "
        "order implied by the question—that is, the number of nested belief levels in "
        "the question. Infer N from the question itself. Give a natural-language "
        "chain-of-thought answer with exactly N numbered reasoning blocks. Number them "
        "consecutively starting at 1, using `Think 1:`, `Think 2:`, and so on; use actual "
        "integers in the output, never the letter N. Each block must begin with its Think "
        "marker at the beginning of a new line, contain non-empty reasoning, and end on "
        "a new line with exactly one `State:` followed by one location from Choices. "
        "After the final block, output exactly one `Answer:` on a new line, repeating the "
        "final State. Put every marker at the beginning of a new line. Do not output "
        "JSON, Markdown, choice letters, extra blocks, or text after Answer."
    )
    return f"{judge_prompt.strip()}\n\n{format_instruction}"


def compact_process_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy whose model-facing prompt uses the compact protocol."""
    compact = dict(record)
    compact["process_prompt"] = build_compact_process_prompt(record)
    compact["process_prompt_version"] = COMPACT_NATURAL_COT_PROMPT_VERSION
    return compact


def format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Apply the tokenizer's chat template exactly once."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
