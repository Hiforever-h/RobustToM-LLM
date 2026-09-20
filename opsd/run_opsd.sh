#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

DATA_DIR="${OPSD_DATA_DIR:-data/counterfactual_process_reward_v4_natural_compact_opsd}"
OUTPUT_DIR="${OPSD_OUTPUT_DIR:-runs/opsd/qwen25-3b-grpo-privileged-opsd-100step}"
RUN_NAME="${OPSD_RUN_NAME:-qwen25-3b-grpo-privileged-opsd-100step}"
WANDB_PROJECT_NAME="${WANDB_PROJECT:-RobustToM-OPSD}"
LOG_DIR="${OPSD_LOG_DIR:-runs/opsd/logs}"

mkdir -p "${LOG_DIR}"

usage() {
    printf '%s\n' \
        "Usage: bash opsd/run_opsd.sh MODE [extra arguments]" \
        "" \
        "Modes:" \
        "  build   Build the 3,200-row paired student/teacher JSONL." \
        "  train   Run exactly 100 OPSD optimizer steps on one 80GB GPU." \
        "  merge   Merge the final LoRA adapter into the GRPO base model." \
        "" \
        "Required for train/merge:" \
        "  export OPSD_MODEL_PATH=/path/to/full/grpo/huggingface/checkpoint" \
        "" \
        "Optional: OPSD_OUTPUT_DIR, OPSD_RUN_NAME, WANDB_PROJECT, WANDB_ENTITY," \
        "          WANDB_MODE, PYTHON_BIN. Extra train arguments are forwarded." 
}

require_model() {
    if [[ -z "${OPSD_MODEL_PATH:-}" ]]; then
        printf '%s\n' "OPSD_MODEL_PATH must point to the full GRPO checkpoint." >&2
        exit 2
    fi
    if [[ ! -f "${OPSD_MODEL_PATH}/config.json" ]]; then
        printf '%s\n' "Missing model config: ${OPSD_MODEL_PATH}/config.json" >&2
        exit 2
    fi
}

case "${MODE}" in
    build)
        "${PYTHON_BIN}" -m opsd.build_dataset \
            --output-dir "${DATA_DIR}" \
            "$@"
        ;;
    train)
        require_model
        if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
            printf '%s\n' "Missing ${DATA_DIR}/train.jsonl; run build first." >&2
            exit 2
        fi
        if [[ "$(uname -s)" != "Linux" ]]; then
            printf '%s\n' "OPSD training requires Linux + NVIDIA CUDA." >&2
            exit 2
        fi
        export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
        export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/robusttom-opsd-triton-cache}"
        mkdir -p "${TRITON_CACHE_DIR}"
        if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
            printf '%s\n' \
                "Removing PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; vLLM sleep mode is incompatible with it." \
                >&2
            unset PYTORCH_CUDA_ALLOC_CONF
        fi
        export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
        "${PYTHON_BIN}" -m accelerate.commands.launch \
            --config_file opsd/accelerate_single_gpu.yaml \
            -m opsd.train \
            --model "${OPSD_MODEL_PATH}" \
            --data "${DATA_DIR}/train.jsonl" \
            --output-dir "${OUTPUT_DIR}" \
            --run-name "${RUN_NAME}" \
            --wandb-project "${WANDB_PROJECT_NAME}" \
            --max-steps 100 \
            "$@" 2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}.log"
        ;;
    merge)
        require_model
        "${PYTHON_BIN}" -m opsd.merge_adapter \
            --base-model "${OPSD_MODEL_PATH}" \
            --adapter "${OUTPUT_DIR}/final_adapter" \
            --output "${OUTPUT_DIR}/merged_model" \
            "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        printf '%s\n' "Unknown mode: ${MODE}" >&2
        usage >&2
        exit 2
        ;;
esac
