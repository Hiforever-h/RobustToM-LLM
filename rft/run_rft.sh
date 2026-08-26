#!/usr/bin/env bash
set -euo pipefail

# This entrypoint intentionally starts from an existing candidate JSONL. It does
# not silently resample or fabricate a canonical process_response completion.
NATURAL_DATA_DIR="${NATURAL_DATA_DIR:-data/counterfactual_process_reward_v4_natural}"
DATA_DIR="${DATA_DIR:-data/rft/derived_v3_fewshot}"
SCORING_DIR="${SCORING_DIR:-runs/rft_scoring}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TRAIN_DIR="${TRAIN_DIR:-data/rft/accepted/${RUN_ID}}"
CANDIDATES="${CANDIDATES:?Set CANDIDATES to an existing raw candidate JSONL}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

mkdir -p "${SCORING_DIR}/${RUN_ID}" "${TRAIN_DIR}"

if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
  python -m rft.prepare_data \
    --input-dir "${NATURAL_DATA_DIR}" \
    --output-dir "${DATA_DIR}"
fi

python -m rft.score_candidates \
  --candidates "${CANDIDATES}" \
  --data "${DATA_DIR}/train.jsonl" \
  --output "${SCORING_DIR}/${RUN_ID}/scored.jsonl"

python -m rft.build_dataset \
  --scored "${SCORING_DIR}/${RUN_ID}/scored.jsonl" \
  --output "${TRAIN_DIR}/train.jsonl" \
  --min-samples "${MIN_SAMPLES:-1000}" \
  --max-samples "${MAX_SAMPLES:-3000}"

python -m rft.train \
  --model "${MODEL}" \
  --train-file "${TRAIN_DIR}/train.jsonl" \
  --output-dir "${RUNS_DIR:-runs/rft_train}/${RUN_ID}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-2048}"
