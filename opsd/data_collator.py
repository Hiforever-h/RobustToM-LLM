"""Batch construction for RobustToM on-policy self-distillation."""

from __future__ import annotations

from typing import Any

import torch


class SelfDistillationDataCollator:
    """Create aligned student and privileged-teacher prompt tensors.

    The derived OPSD JSONL stores the original unprivileged process prompt in
    ``problem`` and the augmented privileged prompt in ``solution``.  Both are
    complete user messages, so no task wording is added here.  The same model
    chat template is applied exactly once to each role.
    """

    def __init__(self, tokenizer: Any, max_prompt_length: int) -> None:
        if max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be positive")
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.tokenizer.padding_side = "right"

    def _render(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("OPSD prompts must be non-empty strings")
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _encode(self, prompts: list[str], role: str) -> tuple[dict[str, torch.Tensor], list[int]]:
        unpadded = self.tokenizer(
            prompts,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        lengths = [len(ids) for ids in unpadded["input_ids"]]
        longest = max(lengths)
        if longest > self.max_prompt_length:
            raise ValueError(
                f"{role} prompt has {longest} tokens, exceeding the audited "
                f"limit {self.max_prompt_length}; refusing to truncate"
            )
        encoded = self.tokenizer(
            prompts,
            padding="longest",
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return encoded, lengths

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty OPSD batch")

        student_texts = [self._render(feature["problem"]) for feature in features]
        teacher_texts = [self._render(feature["solution"]) for feature in features]
        student, student_lengths = self._encode(student_texts, "student")
        teacher, teacher_lengths = self._encode(teacher_texts, "teacher")

        return {
            "student_prompts": student["input_ids"],
            "student_prompt_attention_mask": student["attention_mask"],
            "student_prompt_length": student["input_ids"].shape[1],
            "student_prompt_lengths_per_example": torch.tensor(
                student_lengths, dtype=torch.long
            ),
            "teacher_prompts": teacher["input_ids"],
            "teacher_prompt_attention_mask": teacher["attention_mask"],
            "teacher_prompt_length": teacher["input_ids"].shape[1],
            "teacher_prompt_lengths_per_example": torch.tensor(
                teacher_lengths, dtype=torch.long
            ),
        }
