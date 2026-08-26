#!/usr/bin/env python3
"""Sample raw candidates with vLLM; this module never imports ``verl``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rft.common import prompt_from_record, read_jsonl, sha256_file, sha256_text, write_jsonl
from rft.prompt import format_chat_prompt


def sample_with_vllm(
    rows: list[dict[str, Any]],
    model: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    tensor_parallel_size: int = 1,
    revision: str | None = None,
    gpu_memory_utilization: float = 0.85,
) -> list[dict[str, Any]]:
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only on GPU host
        raise RuntimeError("vLLM is required for rft.sample; install rft/requirements.txt") from exc

    llm_kwargs: dict[str, Any] = {
        "model": model,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if revision:
        llm_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
        skip_special_tokens=True,
    )
    prompts = [format_chat_prompt(tokenizer, prompt_from_record(row)) for row in rows]
    outputs = llm.generate(prompts, sampling_params)
    candidates: list[dict[str, Any]] = []
    for row, request_output in zip(rows, outputs):
        for index, output in enumerate(request_output.outputs):
            text = output.text
            finish_reason = getattr(output, "finish_reason", None)
            token_ids = list(getattr(output, "token_ids", ()) or ())
            candidates.append(
                {
                    "candidate_id": f"{row['global_sample_id']}::seed{seed}::candidate{index}",
                    "global_sample_id": row["global_sample_id"],
                    "global_pair_id": row.get("global_pair_id"),
                    "process_target": row["process_target"],
                    "process_prompt": row["process_prompt"],
                    "prompt_sha256": sha256_text(row["process_prompt"]),
                    "formatted_prompt_sha256": sha256_text(
                        format_chat_prompt(tokenizer, row["process_prompt"])
                    ),
                    "raw_response": text,
                    "response_token_ids": token_ids,
                    "token_count": len(token_ids),
                    "candidate_index": index,
                    "seed": seed,
                    "generation_reached_eos": finish_reason == "stop",
                    "finish_reason": finish_reason,
                    "model": model,
                    "model_revision": revision,
                    "generation_config": {
                        "num_samples": num_samples,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "tensor_parallel_size": tensor_parallel_size,
                    },
                    "source_dataset": row.get("source_dataset"),
                    "question_order": row.get("question_order"),
                    "intervention_type": row.get("intervention_type"),
                    "shortcut_conflict": row.get("shortcut_conflict", False),
                    "last_mention_conflict": row.get("last_mention_conflict", False),
                    "shortcut_prediction": row.get("shortcut_prediction"),
                    "last_mentioned_container": row.get("last_mentioned_container"),
                }
            )
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data)
    candidates = sample_with_vllm(
        rows,
        model=args.model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        revision=args.revision,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    write_jsonl(args.output, candidates)
    manifest = {
        "input_data": str(args.data),
        "input_sha256": sha256_file(args.data),
        "candidate_count": len(candidates),
        "prompt_count": len(rows),
        "model": args.model,
        "revision": args.revision,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_name("candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
