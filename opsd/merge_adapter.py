#!/usr/bin/env python3
"""Merge a trained RobustToM OPSD LoRA adapter into its GRPO base model."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.adapter.is_dir():
        raise FileNotFoundError(f"Missing adapter directory: {args.adapter}")
    if args.output.exists():
        raise FileExistsError(f"Merge output already exists: {args.output}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter,
        trust_remote_code=False,
    )
    tokenizer.save_pretrained(args.output)
    print(f"Merged OPSD model saved to {args.output}")


if __name__ == "__main__":
    main()
