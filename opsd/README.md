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

The launcher removes `expandable_segments:True` from PyTorch's allocator
configuration because it is incompatible with the vLLM sleep-mode memory
pool. It also creates a writable Triton cache under `/tmp`. A Linux kernel
older than 5.5 may still produce an Accelerate warning and can hang under
heavy distributed/CUDA workloads; changing the kernel requires a newer host
or container host rather than a Python package change.

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
| point-wise token clip | 1e-6 |
| sampling | temperature 0.8, top-p 0.95 |
| vLLM | colocate, sleep mode, utilization 0.30 |
| seed | 2026 |

At startup the trainer tokenizes all student and teacher prompts and refuses
to truncate any prompt. It also writes the resolved configuration and prompt
length audit to `run_manifest.json` before loading the training model.

## Merge for evaluation

Training saves a LoRA adapter. Merge it into the same GRPO checkpoint before
using the existing `rft.generate` evaluator:

```bash
export OPSD_MODEL_PATH=/path/to/grpo/actor/global_step_800
bash opsd/run_opsd.sh merge
```

The merged Hugging Face model is written to
`runs/opsd/qwen25-3b-grpo-privileged-opsd-100step/merged_model`.
