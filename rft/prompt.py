"""Single source of truth for model-facing chat prompt construction."""

from __future__ import annotations

from typing import Any


NATURAL_COT_PROMPT_VERSION = "natural-cot-think-state-v2-exact-order"


def format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Apply the tokenizer's chat template exactly once."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
