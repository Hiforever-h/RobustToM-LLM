#!/usr/bin/env bash
set -euo pipefail

# This entrypoint intentionally starts from an existing candidate JSONL. It does
# not silently resample or use canonical process_response as a completion.
DATA_DIR="${DATA_DIR:-data/rft/derived}"
SCORING_DIR="${SCORING_DIR:-runs/rft_scoring}"
TRAIN_DIR="${TRAIN_DIR:-data/rft/accepted}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
CANDIDATES="${CANDIDATES:?Set CANDIDATES to an existing raw candidate JSONL}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

mkdir -p "${SCORING_DIR}/${RUN_ID}" "${TRAIN_DIR}"

python -m rft.prepare_data \
  --input-dir data/counterfactual_process_reward \
  --output-dir "${DATA_DIR}" \
  --seed 2026

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
