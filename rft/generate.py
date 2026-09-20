#!/usr/bin/env python3
"""Generate deterministic Think/State/Answer responses for RFT evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rft.common import prompt_from_record, read_jsonl, sha256_text, write_jsonl
from rft.prompt import compact_process_record, format_chat_prompt


def load_adapter_spec(adapter: Path | None) -> tuple[str | None, int | None]:
    """Validate a PEFT LoRA directory and return its absolute path and rank."""
    if adapter is None:
        return None, None
    if not adapter.is_dir():
        raise FileNotFoundError(f"Missing adapter directory: {adapter}")
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = config.get("r")
    if type(rank) is not int or rank <= 0:
        raise ValueError(f"Invalid LoRA rank in {config_path}: {rank!r}")
    peft_type = config.get("peft_type")
    if peft_type is not None and str(peft_type).upper() != "LORA":
        raise ValueError(f"Only LoRA adapters are supported, found {peft_type!r}")
    return str(adapter.resolve()), rank


def prepare_generation_rows(
    rows: list[dict[str, Any]], compact_prompt: bool = False
) -> list[dict[str, Any]]:
    """Prepare model inputs without mutating the source dataset records."""
    if not compact_prompt:
        return rows
    return [compact_process_record(row) for row in rows]


def generate_vllm(
    rows: list[dict[str, Any]],
    model: str,
    revision: str | None,
    max_new_tokens: int,
    seed: int,
    gpu_memory_utilization: float,
    adapter: Path | None = None,
) -> list[dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("vLLM and transformers are required for GPU generation") from exc
    adapter_path, adapter_rank = load_adapter_spec(adapter)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "tensor_parallel_size": 1,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if adapter_path is not None:
        llm_kwargs.update(enable_lora=True, max_lora_rank=adapter_rank)
    if revision:
        llm_kwargs["revision"] = revision
    llm = LLM(**llm_kwargs)
    params = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, seed=seed)
    prompts = [format_chat_prompt(tokenizer, prompt_from_record(row)) for row in rows]
    if adapter_path is not None:
        from vllm.lora.request import LoRARequest

        generated = llm.generate(
            prompts,
            params,
            lora_request=LoRARequest("evaluation_adapter", 1, adapter_path),
        )
    else:
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
                **(
                    {"generation_adapter": adapter_path}
                    if adapter_path is not None
                    else {}
                ),
            }
        )
    return result


def generate_transformers(
    rows: list[dict[str, Any]],
    model: str,
    revision: str | None,
    max_new_tokens: int,
    adapter: Path | None = None,
) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch and transformers are required for HF generation") from exc
    adapter_path, _ = load_adapter_spec(adapter)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_obj = AutoModelForCausalLM.from_pretrained(model, revision=revision, torch_dtype=dtype)
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("peft is required for adapter evaluation") from exc
        model_obj = PeftModel.from_pretrained(
            model_obj,
            adapter_path,
            is_trainable=False,
        )
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
                **(
                    {"generation_adapter": adapter_path}
                    if adapter_path is not None
                    else {}
                ),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Optional PEFT LoRA adapter loaded on top of --model without merging",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument(
        "--compact-prompt",
        action="store_true",
        help=(
            "Generate from judge_prompt plus the same concise Think/State/Answer "
            "instructions used by build_dataset --compact-prompt"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = prepare_generation_rows(read_jsonl(args.data), args.compact_prompt)
    if args.backend == "vllm":
        predictions = generate_vllm(
            rows,
            args.model,
            args.revision,
            args.max_new_tokens,
            args.seed,
            args.gpu_memory_utilization,
            args.adapter,
        )
    else:
        predictions = generate_transformers(
            rows,
            args.model,
            args.revision,
            args.max_new_tokens,
            args.adapter,
        )
    write_jsonl(args.output, predictions)
    print(
        json.dumps(
            {
                "prediction_count": len(predictions),
                "output": str(args.output),
                "base_model": args.model,
                "adapter": str(args.adapter.resolve()) if args.adapter else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
