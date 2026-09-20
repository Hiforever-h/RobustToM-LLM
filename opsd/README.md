# RobustToM OPSD

This directory adds a single-GPU, 100-step On-Policy Self-Distillation
(OPSD) run for the 3,200-example RobustToM training split.

The student sees the original compact process prompt. The fixed teacher is
the same GRPO checkpoint with its LoRA adapter disabled, but receives the
verified answer and one support event per belief level. The student samples
one on-policy completion; the teacher only scores that same completion. No
teacher completion is used as an SFT target.

The trainer is adapted from
[`siyan-zhao/OPSD`](https://github.com/siyan-zhao/OPSD) commit
`ae7d2519e94920c4eb6206c0c26de46d9c50abae` under Apache-2.0.

## Environment

Use a separate environment. The repository's existing `verl` environment
pins Transformers 4.46 and vLLM 0.6.3, while the upstream GOLD trainer needs
the newer stack in `opsd/requirements.txt`.

```bash
conda create -n robusttom-opsd python=3.10 pip -y
conda activate robusttom-opsd
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r opsd/requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e . --no-deps
```

Log in to Weights & Biases once on the training host:

```bash
wandb login
```

## Data

The checked-in derived data can be reproduced into a separate audit directory
with:

```bash
OPSD_DATA_DIR=data/counterfactual_process_reward_v4_natural_compact_opsd_rebuilt \
  bash opsd/run_opsd.sh build
```

Each `train.jsonl` row uses the upstream-compatible names:

- `problem`: original student process prompt;
- `solution`: complete privileged-teacher process prompt.

The builder validates that each sample has exactly one support event per
belief level and records hashes and counts in `manifest.json`.

## Train for 100 steps

Point `OPSD_MODEL_PATH` at the full Hugging Face-format GRPO actor checkpoint,
not at a LoRA adapter:

```bash
export OPSD_MODEL_PATH=/path/to/grpo/actor/global_step_800
bash opsd/run_opsd.sh train
```

The launcher uses one GPU, enables the terminal tqdm progress bar, and logs
every optimizer step plus sampled completions to W&B project
`RobustToM-OPSD`. Set `WANDB_PROJECT`, `WANDB_ENTITY`, or `WANDB_MODE` to
override W&B behavior. There is no validation during training. Checkpoints
are saved at steps 50 and 100, followed by `final_adapter`.

vLLM sleep mode is disabled by default. Its CUDA memory-pool sleep/wake path
is not reliable on the Linux 4.19 kernels still used by some GPU hosts. The
A800 recipe instead leaves vLLM resident at 30% GPU utilization. On a newer
kernel, sleep mode can be explicitly tested with `--vllm-sleep-mode`; the
trainer then removes the incompatible `expandable_segments:True` allocator
setting. The launcher also creates a writable Triton cache under `/tmp`.

## Fixed training recipe

| Setting | Value |
| --- | ---: |
| optimizer steps | 100 |
| dataset passes | 1 |
| rollout count per prompt | 1 |
| micro-batch | 2 |
| gradient accumulation | 16 |
| effective batch | 32 |
| max sequence / completion | 3072 / 384 |
| learning rate | 5e-6, constant |
| optimizer | fused AdamW |
| precision | bfloat16 + TF32 |
| student update | LoRA rank 64, alpha 128 |
| teacher | fixed initial GRPO policy |
| loss | full-vocabulary forward KL (`beta=0`) |
| per-token KL clip | disabled by default |
| sampling | temperature 0.8, top-p 0.95 |
| vLLM | colocate, resident (sleep off), utilization 0.30 |
| seed | 2026 |

At startup the trainer tokenizes all student and teacher prompts and refuses
to truncate any prompt. It also writes the resolved configuration and prompt
length audit to `run_manifest.json` before loading the training model.

Prompt batches use left padding so each real prompt remains directly adjacent
to its sampled completion. Position IDs are derived from the attention mask,
making them invariant to the amount of left padding. If
`--jsd-token-clip VALUE` is explicitly supplied, the forward KL is first
summed across the vocabulary and only then clipped per completion token.

## Evaluate without merging

The evaluator can attach a checkpoint or final PEFT adapter to the unchanged
GRPO base model at runtime. For example, evaluate the final adapter on the
unprivileged validation split with deterministic decoding:

```bash
export BASE_MODEL=/path/to/grpo/actor/global_step_800
export ADAPTER=runs/opsd/qwen25-3b-grpo-privileged-opsd-100step/final_adapter
export EVAL_DIR=runs/opsd_eval/final

python -m rft.generate \
  --data data/counterfactual_process_reward_v4_natural/val.jsonl \
  --model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --output "$EVAL_DIR/val_predictions.jsonl" \
  --backend vllm \
  --max-new-tokens 384 \
  --seed 2026 \
  --compact-prompt

python -m rft.evaluate \
  --predictions "$EVAL_DIR/val_predictions.jsonl" \
  --data data/counterfactual_process_reward_v4_natural/val.jsonl \
  --output "$EVAL_DIR/val_rule_metrics.json" \
  --compact-prompt
```

Set `ADAPTER` to `checkpoint-50` to evaluate the midpoint. vLLM loads the
adapter with its native `LoRARequest`; `--backend transformers` instead uses
PEFT. In both cases the base checkpoint is left unchanged, and
`rft.evaluate` needs no adapter argument.

## Optional merge

Merging is only needed when another inference system cannot load PEFT adapters:

```bash
export OPSD_MODEL_PATH=/path/to/grpo/actor/global_step_800
bash opsd/run_opsd.sh merge
```

The merged Hugging Face model is written to
`runs/opsd/qwen25-3b-grpo-privileged-opsd-100step/merged_model`.
