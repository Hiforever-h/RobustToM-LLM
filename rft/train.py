#!/usr/bin/env python3
"""Train an independent response-only Hugging Face causal-LM RFT model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rft.dataset import ResponseOnlyCollator, ResponseOnlyDataset, build_response_only_example


def _percentile(values: list[int], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def audit_lengths(dataset: ResponseOnlyDataset) -> dict[str, object]:
    lengths: list[int] = []
    over_limit: list[dict[str, object]] = []
    for row in dataset.rows:
        item = build_response_only_example(row, dataset.tokenizer, max_length=10**9)
        length = len(item["input_ids"])
        lengths.append(length)
        if length > dataset.max_length:
            over_limit.append({"global_sample_id": row.get("global_sample_id"), "length": length})
    return {
        "count": len(lengths),
        "min": min(lengths) if lengths else 0,
        "p50": _percentile(lengths, 0.50),
        "p90": _percentile(lengths, 0.90),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "max": max(lengths) if lengths else 0,
        "max_seq_length": dataset.max_length,
        "over_limit": over_limit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--logging-dir",
        type=Path,
        help="TensorBoard event directory (default: <output-dir>/tensorboard)",
    )
    parser.add_argument("--revision")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - requires the GPU training stack
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError("rft.train requires torch and transformers; install rft/requirements.txt") from exc

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logging_dir = args.logging_dir or args.output_dir / "tensorboard"
    args.logging_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = ResponseOnlyDataset(args.train_file, tokenizer, args.max_seq_length)
    length_report = audit_lengths(train_dataset)
    (args.output_dir / "token_length_report.json").write_text(
        json.dumps(length_report, indent=2) + "\n", encoding="utf-8"
    )
    if length_report["over_limit"]:
        first = length_report["over_limit"][:10]
        raise ValueError(
            f"{len(length_report['over_limit'])} samples exceed max_seq_length={args.max_seq_length}; "
            f"increase the limit instead of truncating. First samples: {first}"
        )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"revision": args.revision, "torch_dtype": dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation="flash_attention_2", **model_kwargs
        )
    except (ImportError, TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        logging_dir=str(args.logging_dir),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        evaluation_strategy="no",
        remove_unused_columns=False,
        report_to=["tensorboard"],
        run_name=args.output_dir.name,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": ResponseOnlyCollator(tokenizer),
    }
    try:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    (args.output_dir / "train_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir / "final"), "train_samples": len(train_dataset)}, indent=2))


if __name__ == "__main__":
    main()
