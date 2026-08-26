#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-train}"
if [[ $# -gt 0 ]]; then
    shift
fi

DATA_DIR="${GRPO_DATA_DIR:-data/grpo/counterfactual_process_reward_v3_fewshot}"
LOG_DIR="${GRPO_LOG_DIR:-runs/grpo/logs}"
mkdir -p "${LOG_DIR}"

export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

build_data() {
    "${PYTHON_BIN}" -m grpo.build_dataset \
        --input-dir data/counterfactual_process_reward_v3 \
        --output-dir "${DATA_DIR}" \
        --tokenizer runs/final \
        --max-prompt-length 2048 \
        --max-response-length 256
}

run_trainer() {
    local run_name="$1"
    shift
    "${PYTHON_BIN}" -m verl.trainer.main_robust_tom_grpo \
        data.train_files="${DATA_DIR}/train.parquet" \
        data.val_files="${DATA_DIR}/val.parquet" \
        "$@" 2>&1 | tee "${LOG_DIR}/${run_name}.log"
}

case "${MODE}" in
    build)
        build_data
        ;;
    validate)
        run_trainer validate \
            trainer.val_only=true \
            trainer.resume_from_path=null \
            trainer.experiment_name=qwen25_3b_rft_grpo_validate \
            "$@"
        ;;
    smoke)
        run_trainer smoke \
            data.train_batch_size=2 \
            actor_rollout_ref.rollout.n=4 \
            actor_rollout_ref.actor.ppo_mini_batch_size=8 \
            actor_rollout_ref.actor.ppo_micro_batch_size=1 \
            actor_rollout_ref.rollout.max_num_seqs=8 \
            trainer.total_epochs=1 \
            trainer.total_training_steps=2 \
            trainer.val_before_train=false \
            trainer.test_freq=-1 \
            trainer.save_freq=-1 \
            trainer.save_at_end=false \
            trainer.validate_at_end=false \
            trainer.experiment_name=qwen25_3b_rft_grpo_smoke \
            "$@"
        ;;
    pilot)
        run_trainer pilot \
            trainer.total_epochs=1 \
            trainer.total_training_steps=50 \
            trainer.test_freq=10 \
            trainer.save_freq=-1 \
            trainer.save_at_end=false \
            trainer.experiment_name=qwen25_3b_rft_grpo_pilot_n16_seed2026 \
            "$@"
        ;;
    train)
        run_trainer train "$@"
        ;;
    *)
        echo "Usage: $0 {build|validate|smoke|pilot|train} [hydra overrides...]" >&2
        exit 2
        ;;
esac
