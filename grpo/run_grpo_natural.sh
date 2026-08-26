#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

CONFIG_NAME="${GRPO_CONFIG_NAME:-robust_tom_natural_grpo}"
RAW_DATA_DIR="${RAW_DATA_DIR:-data/counterfactual_process_reward_v3}"
NATURAL_SOURCE_DIR="${NATURAL_SOURCE_DIR:-data/counterfactual_process_reward_v4_natural}"
DATA_DIR="${GRPO_DATA_DIR:-data/grpo/counterfactual_process_reward_v4_natural}"
MODEL_PATH="${RFT_MODEL_PATH:-runs/final}"
OUTPUT_ROOT="${GRPO_OUTPUT_ROOT:-runs/grpo}"
LOG_DIR="${GRPO_LOG_DIR:-${OUTPUT_ROOT}/logs}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

usage() {
    cat <<'EOF'
Usage: bash grpo/run_grpo_natural.sh MODE [Hydra overrides...]

Modes:
  build      Build natural-CoT JSONL and audited parquet files. No GPU/Judge.
  validate   Generate the validation split and run deterministic rule metrics.
             No Judge request and no optimizer update.
  smoke      Run one optimizer step with 1 prompt x 2 rollouts and a real Judge.
  pilot      Run 50 optimizer steps with the production 8 x 16 rollout shape.
  train      Run the full robust_tom_natural_grpo configuration (800 steps).

Important environment variables:
  RFT_MODEL_PATH       Actor/reference checkpoint (default: runs/final)
  RAW_DATA_DIR         Symbolic v3 JSONL input directory
  NATURAL_SOURCE_DIR   Generated natural-CoT JSONL directory
  GRPO_DATA_DIR        Generated parquet directory
  GRPO_OUTPUT_ROOT     Checkpoint/output root
  GRPO_LOG_DIR         Terminal log directory
  DEEPSEEK_API_KEY     Required by smoke, pilot and train (or set it in .env)
  PYTHON_BIN           Python executable (default: python)

Any remaining arguments are forwarded as Hydra overrides.
EOF
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        echo "Run: bash grpo/run_grpo_natural.sh build" >&2
        exit 1
    fi
}

require_training_runtime() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        echo "GRPO trainer modes require Linux + NVIDIA CUDA; current OS is $(uname -s)." >&2
        exit 1
    fi
    if ! "${PYTHON_BIN}" -c 'import torch, ray, vllm; assert torch.cuda.is_available()'; then
        echo "Missing a working CUDA PyTorch/Ray/vLLM runtime." >&2
        echo "Install requirements.txt and flash-attn on the training host." >&2
        exit 1
    fi
}

require_judge_credentials() {
    if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
        return
    fi
    if [[ -f .env ]] && grep -Eq '^[[:space:]]*DEEPSEEK_API_KEY=' .env; then
        return
    fi
    echo "DEEPSEEK_API_KEY is required for ${MODE}. Set it in the environment or .env." >&2
    exit 1
}

build_data() {
    "${PYTHON_BIN}" -m grpo.build_natural_dataset \
        --input-dir "${RAW_DATA_DIR}" \
        --source-output-dir "${NATURAL_SOURCE_DIR}" \
        --parquet-output-dir "${DATA_DIR}" \
        --tokenizer "${MODEL_PATH}" \
        --max-prompt-length 2048
}

run_trainer() {
    local run_name="$1"
    shift
    require_training_runtime
    require_file "${DATA_DIR}/train.parquet"
    require_file "${DATA_DIR}/val.parquet"

    "${PYTHON_BIN}" -m verl.trainer.main_robust_tom_grpo \
        --config-name="${CONFIG_NAME}" \
        data.train_files="${DATA_DIR}/train.parquet" \
        data.val_files="${DATA_DIR}/val.parquet" \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.ref.model_path="${MODEL_PATH}" \
        trainer.experiment_name="${run_name}" \
        trainer.default_local_dir="${OUTPUT_ROOT}/${run_name}" \
        "$@" 2>&1 | tee "${LOG_DIR}/${run_name}.log"
}

case "${MODE}" in
    build)
        build_data
        ;;
    validate)
        run_trainer natural_cot_validate \
            trainer.val_only=true \
            trainer.resume_from_path=null \
            trainer.logger='[console]' \
            reward.judge.preflight_remote=false \
            "$@"
        ;;
    smoke)
        require_judge_credentials
        run_trainer natural_cot_smoke \
            data.train_batch_size=1 \
            actor_rollout_ref.rollout.n=2 \
            actor_rollout_ref.actor.ppo_mini_batch_size=2 \
            actor_rollout_ref.actor.ppo_micro_batch_size=1 \
            actor_rollout_ref.rollout.max_num_seqs=2 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
            actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
            reward.judge.max_workers=1 \
            trainer.total_epochs=1 \
            trainer.total_training_steps=1 \
            trainer.val_before_train=false \
            trainer.test_freq=-1 \
            trainer.save_freq=-1 \
            trainer.save_at_end=false \
            trainer.validate_at_end=false \
            trainer.logger='[console]' \
            "$@"
        ;;
    pilot)
        require_judge_credentials
        run_trainer natural_cot_pilot_n16_seed2026 \
            trainer.total_epochs=1 \
            trainer.total_training_steps=50 \
            trainer.val_before_train=true \
            trainer.test_freq=10 \
            trainer.save_freq=25 \
            trainer.save_at_end=true \
            trainer.validate_at_end=true \
            "$@"
        ;;
    train)
        require_judge_credentials
        run_trainer qwen25_3b_natural_cot_judge_n16_seed2026 "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        usage >&2
        exit 2
        ;;
esac
