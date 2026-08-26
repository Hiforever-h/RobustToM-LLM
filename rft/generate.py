#!/usr/bin/env python3
"""Generate deterministic process responses for RFT evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rft.common import prompt_from_record, read_jsonl, sha256_text, write_jsonl
from rft.prompt import format_chat_prompt


def generate_vllm(
    rows: list[dict[str, Any]],
    model: str,
    revision: str | None,
    max_new_tokens: int,
    seed: int,
    gpu_memory_utilization: float,
) -> list[dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("vLLM and transformers are required for GPU generation") from exc
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "tensor_parallel_size": 1,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if revision:
        llm_kwargs["revision"] = revision
    llm = LLM(**llm_kwargs)
    params = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, seed=seed)
    prompts = [format_chat_prompt(tokenizer, prompt_from_record(row)) for row in rows]
    generated = llm.generate(prompts, params)
    result = []
    for row, request in zip(rows, generated):
        output = request.outputs[0]
        result.append(
            {
                **row,
                "response": output.text,
                "raw_response": output.text,
                "response_token_ids": list(getattr(output, "token_ids", ()) or ()),
                "token_count": len(getattr(output, "token_ids", ()) or ()),
                "generation_reached_eos": getattr(output, "finish_reason", None) == "stop",
                "finish_reason": getattr(output, "finish_reason", None),
                "prompt_sha256": sha256_text(row["process_prompt"]),
                "formatted_prompt_sha256": sha256_text(
                    format_chat_prompt(tokenizer, row["process_prompt"])
                ),
            }
        )
    return result


def generate_transformers(
    rows: list[dict[str, Any]],
    model: str,
    revision: str | None,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch and transformers are required for HF generation") from exc
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_obj = AutoModelForCausalLM.from_pretrained(model, revision=revision, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_obj.to(device).eval()
    result = []
    for row in rows:
        formatted = format_chat_prompt(tokenizer, prompt_from_record(row))
        encoded = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            generated = model_obj.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        response_ids = generated[0, encoded["input_ids"].shape[1] :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        result.append(
            {
                **row,
                "response": response,
                "raw_response": response,
                "response_token_ids": response_ids.tolist(),
                "token_count": len(response_ids),
                "generation_reached_eos": bool(
                    response_ids.numel() > 0 and int(response_ids[-1]) == tokenizer.eos_token_id
                ),
                "finish_reason": "stop"
                if response_ids.numel() > 0 and int(response_ids[-1]) == tokenizer.eos_token_id
                else "length",
                "prompt_sha256": sha256_text(row["process_prompt"]),
                "formatted_prompt_sha256": sha256_text(formatted),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data)
    if args.backend == "vllm":
        predictions = generate_vllm(rows, args.model, args.revision, args.max_new_tokens, args.seed, args.gpu_memory_utilization)
    else:
        predictions = generate_transformers(rows, args.model, args.revision, args.max_new_tokens)
    write_jsonl(args.output, predictions)
    print(json.dumps({"prediction_count": len(predictions), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
