#!/usr/bin/env python3
"""Run 100-step privileged-context OPSD on one A800 80GB GPU."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATA = Path(
    "data/counterfactual_process_reward_v4_natural_compact_opsd/train.jsonl"
)
DEFAULT_OUTPUT = Path("runs/opsd/qwen25-3b-grpo-privileged-opsd-100step")
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _sanitize_allocator_environment(vllm_sleep_mode: bool) -> None:
    """Remove allocator settings that are incompatible with vLLM sleep mode."""
    if not vllm_sleep_mode:
        return
    for variable in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        value = os.environ.get(variable)
        if not value:
            continue
        settings = [item.strip() for item in value.split(",") if item.strip()]
        compatible = [
            item
            for item in settings
            if item.lower() != "expandable_segments:true"
        ]
        if len(compatible) == len(settings):
            continue
        if compatible:
            os.environ[variable] = ",".join(compatible)
        else:
            os.environ.pop(variable, None)
        print(
            f"Removed expandable_segments:True from {variable}: it is "
            "incompatible with the vLLM sleep-mode memory pool.",
            file=sys.stderr,
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Full Hugging Face GRPO checkpoint")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="qwen25-3b-grpo-privileged-opsd-100step")
    parser.add_argument("--wandb-project", default="RobustToM-OPSD")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-sequence-length", type=int, default=3072)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument(
        "--vllm-sleep-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable vLLM CUDA memory-pool sleep/wake (disabled for kernel compatibility)",
    )
    parser.add_argument("--jsd-token-clip", type=float, default=1e-6)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def _validate_cli(args: argparse.Namespace) -> None:
    if args.max_steps != 100:
        raise ValueError("This experiment is fixed to exactly --max-steps 100")
    if args.max_completion_length <= 0:
        raise ValueError("--max-completion-length must be positive")
    if args.max_sequence_length <= args.max_completion_length:
        raise ValueError("max sequence length must exceed max completion length")
    if args.per_device_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    if not 0.0 < args.vllm_gpu_memory_utilization < 1.0:
        raise ValueError("vLLM GPU utilization must be between zero and one")
    if args.jsd_token_clip <= 0:
        raise ValueError("--jsd-token-clip must be positive")
    if not args.data.is_file():
        raise FileNotFoundError(f"Missing OPSD dataset: {args.data}")
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(
            "--model must be a local full Hugging Face GRPO checkpoint directory: "
            f"{model_path}"
        )
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {model_path / 'config.json'}")
    if args.output_dir.exists() and args.resume_from_checkpoint is None:
        # ``run_manifest.json`` is written before the heavyweight trainer is
        # initialized.  Allow a clean retry when initialization failed after
        # that write but before any checkpoint or model artifact was created.
        blocking_entries = sorted(
            path.name
            for path in args.output_dir.iterdir()
            if path.name != "run_manifest.json"
        )
        if blocking_entries:
            preview = ", ".join(blocking_entries[:5])
            raise FileExistsError(
                f"Output directory already contains training artifacts: "
                f"{args.output_dir} ({preview}). Use a new directory or pass "
                "--resume-from-checkpoint."
            )


def _render_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _audit_dataset(
    dataset: Any,
    tokenizer: Any,
    max_prompt_length: int,
) -> dict[str, Any]:
    if len(dataset) != 3200:
        raise ValueError(
            f"Expected the original 3,200-sample train split, found {len(dataset)}"
        )
    required = {"problem", "solution", "global_sample_id", "question_order"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"OPSD dataset is missing columns: {missing}")
    sample_ids = dataset["global_sample_id"]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("OPSD dataset contains duplicate global_sample_id values")

    student_lengths: list[int] = []
    teacher_lengths: list[int] = []
    for row in dataset:
        for key in ("problem", "solution"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Empty {key} for {row['global_sample_id']}")
        student = _render_prompt(tokenizer, row["problem"])
        teacher = _render_prompt(tokenizer, row["solution"])
        student_lengths.append(
            len(tokenizer(student, add_special_tokens=False)["input_ids"])
        )
        teacher_lengths.append(
            len(tokenizer(teacher, add_special_tokens=False)["input_ids"])
        )

    longest = max(max(student_lengths), max(teacher_lengths))
    if longest > max_prompt_length:
        raise ValueError(
            f"Longest prompt has {longest} tokens but only {max_prompt_length} "
            "tokens remain after reserving completion capacity"
        )

    def percentile(values: list[int], fraction: float) -> int:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "count": len(dataset),
        "question_order_counts": {
            str(order): dataset["question_order"].count(order)
            for order in sorted(set(dataset["question_order"]))
        },
        "student_prompt_tokens": {
            "max": max(student_lengths),
            "p95": percentile(student_lengths, 0.95),
        },
        "teacher_prompt_tokens": {
            "max": max(teacher_lengths),
            "p95": percentile(teacher_lengths, 0.95),
        },
        "max_prompt_length": max_prompt_length,
    }


def main() -> None:
    args = parse_args()
    _validate_cli(args)
    _sanitize_allocator_environment(args.vllm_sleep_mode)

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoTokenizer
        from trl.experimental.gold import GOLDConfig
    except ImportError as exc:
        raise RuntimeError(
            "Install the isolated OPSD environment from opsd/requirements.txt"
        ) from exc

    from opsd.data_collator import SelfDistillationDataCollator
    from opsd.trainer import OPSDTrainer

    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("OPSD training requires Linux and a CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"This launcher is configured for exactly one GPU, found {torch.cuda.device_count()}"
        )
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 70 * 1024**3:
        raise RuntimeError(
            f"Expected an 80GB-class GPU, found {props.total_memory / 1024**3:.1f} GiB"
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_WATCH", "false")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=False,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(args.data), split="train")
    max_prompt_length = args.max_sequence_length - args.max_completion_length
    audit = _audit_dataset(dataset, tokenizer, max_prompt_length)

    effective_batch_size = (
        args.per_device_batch_size * args.gradient_accumulation_steps
    )
    expected_examples = args.max_steps * effective_batch_size
    if expected_examples != len(dataset):
        raise ValueError(
            "The fixed 100-step recipe must consume the 3,200 examples exactly once: "
            f"max_steps({args.max_steps}) * effective_batch({effective_batch_size}) "
            f"= {expected_examples}, dataset={len(dataset)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "method": "on-policy self-distillation",
        "upstream_opsd_commit": "ae7d2519e94920c4eb6206c0c26de46d9c50abae",
        "model": str(Path(args.model).resolve()),
        "data": str(args.data.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "run_name": args.run_name,
        "wandb_project": args.wandb_project,
        "wandb_entity": os.environ.get("WANDB_ENTITY"),
        "wandb_mode": os.environ.get("WANDB_MODE", "online"),
        "max_steps": args.max_steps,
        "num_train_epochs_equivalent": 1.0,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "student_rollouts_per_prompt": 1,
        "learning_rate": args.learning_rate,
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "constant",
        "max_sequence_length": args.max_sequence_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "teacher": "fixed initial GRPO policy via disabled LoRA adapter",
        "loss": "full-vocabulary generalized JSD (beta=0 forward KL)",
        "jsd_token_clip": args.jsd_token_clip,
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_targets": LORA_TARGETS,
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_sleep_mode": args.vllm_sleep_mode,
        "validation_during_training": False,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "gpu": props.name,
        "gpu_memory_gib": round(props.total_memory / 1024**3, 2),
        "dataset_audit": audit,
        "argv": sys.argv,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2), flush=True)

    training_args = GOLDConfig(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        report_to=["wandb"],
        wandb_project=args.wandb_project,
        wandb_entity=os.environ.get("WANDB_ENTITY"),
        max_steps=args.max_steps,
        num_train_epochs=1.0,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim="adamw_torch_fused",
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        max_grad_norm=0.1,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_sequence_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        beta=0.0,
        lmbda=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_sleep_mode,
        vllm_sync_frequency=1,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        log_completions=True,
        log_completions_steps=5,
        num_completions_to_print=4,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="no",
        disable_tqdm=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        # OPSD's collator consumes the raw ``problem``/``solution`` columns and
        # constructs two differently conditioned sequences on the fly.  The
        # inherited SFTTrainer preprocessing otherwise tries to tokenize the
        # default ``text`` column before our collator runs.
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=args.seed,
        data_seed=args.seed,
        model_init_kwargs={
            "attn_implementation": "flash_attention_2",
            "dtype": torch.bfloat16,
            "use_cache": False,
            "trust_remote_code": False,
        },
    )
    training_args.presence_penalty = 0.0

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        target_modules=LORA_TARGETS,
    )
    collator = SelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
    )
    trainer = OPSDTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=collator,
        peft_config=peft_config,
        use_thinking_machines_loss=False,
        fixed_teacher=True,
        reason_first=False,
        top_k_loss=None,
        jsd_token_clip=args.jsd_token_clip,
        use_ema_teacher=False,
        student_thinking=False,
        teacher_thinking=False,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final_adapter"))
    tokenizer.save_pretrained(args.output_dir / "final_adapter")
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
