"""Response-only causal-LM dataset and dynamic padding collator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rft.common import prompt_from_record, read_jsonl
from rft.prompt import format_chat_prompt


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def build_response_only_example(
    row: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, Any]:
    """Build unpadded IDs/labels and fail instead of truncating story or response."""
    prompt_text = format_chat_prompt(tokenizer, prompt_from_record(row))
    prompt_ids = _encode(tokenizer, prompt_text)
    response = row.get("accepted_response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"Missing accepted_response: {row.get('global_sample_id', '<unknown>')}")
    response_text = response
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token and response_text.endswith(eos_token):
        response_text = response_text[: -len(eos_token)]
    response_ids = _encode(tokenizer, response_text)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    response_ids.append(int(eos_id))
    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        raise ValueError(
            f"Sample {row.get('global_sample_id', '<unknown>')} has {len(input_ids)} tokens, "
            f"exceeding max_length={max_length}; increase the limit instead of truncating."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
        "prompt_length": len(prompt_ids),
        "response_length": len(response_ids),
        "global_sample_id": row.get("global_sample_id"),
        "global_pair_id": row.get("global_pair_id"),
    }


class ResponseOnlyDataset:
    """Tokenize prompt and accepted response without truncating either side."""

    def __init__(self, data: str | Path | list[dict[str, Any]], tokenizer: Any, max_length: int):
        if isinstance(data, (str, Path)):
            data = read_jsonl(data)
        self.rows = list(data)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("Tokenizer must define pad_token_id before constructing the dataset")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.rows[index]
        example = build_response_only_example(row, self.tokenizer, self.max_length)
        return {
            **example,
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.long),
        }


class ResponseOnlyCollator:
    """Right-pad input, attention and labels; labels never train on padding."""

    def __init__(self, tokenizer: Any):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(item["input_ids"].numel() for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            length = item["input_ids"].numel()
            padding = max_length - length
            input_ids.append(torch.cat([item["input_ids"], torch.full((padding,), self.pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(padding, dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((padding,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
