# Standalone RobustToM-RL RFT

This directory implements rejection-sampling fine-tuning independently of
`verl`. It consumes the process-target JSONL files produced by the project and
exports a standard Hugging Face checkpoint. No canonical `process_response` is
used as a training completion.

## Pipeline

1. `prepare_data` creates the fixed train/dev/test split and validates pairs.
2. An external vLLM run (or `rft.sample`) writes raw candidates.
3. `score_candidates` accepts only strict JSON candidates with reward `1.0` and EOS.
4. `build_dataset` deduplicates by parsed JSON and keeps only complete pairs.
5. `train` performs response-only causal-LM training: prompt labels are `-100`,
   and the accepted response plus one EOS are trainable.
6. `generate` and `evaluate` report process, pair and shortcut-conflict metrics.

The dataset builder keeps every full-reward response by default and does not
require observed/hidden pairs. It never falls back to gold/canonical responses
when coverage is low. Use `--require-complete-pairs` only for the stricter
pair-balanced ablation.

## Environment

Run all commands from the repository root on the A800 machine:

```bash
conda create -n robusttom-rft python=3.10 -y
conda activate robusttom-rft

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r rft/requirements.txt
python -m pip install flash-attn==2.6.3 --no-build-isolation
```

`flash-attn` is optional. If it is unavailable, `rft.train` automatically
falls back to the Transformers attention implementation.

## Prepare the fixed split

The derived files are already present in the repository. Regenerate and
validate them when the source JSONL changes:

```bash
python -m rft.prepare_data \
  --input-dir data/counterfactual_process_reward \
  --output-dir data/rft/derived \
  --seed 2026
```

## Existing candidates

If candidates have already been sampled, point `rft/run_rft.sh` at that JSONL:

```bash
CANDIDATES=/path/to/candidates.jsonl bash rft/run_rft.sh
```

Candidate rows must contain `global_sample_id`, `generation_reached_eos`, and
one of `raw_response`, `response`, or `accepted_response`. They may either
contain `process_target` and metadata or use `--data` to resolve those fields
by sample ID. Missing EOS status is rejected rather than assumed successful.

## One-time sampling

For a new one-time vLLM run:

```bash
RUN_ID=20260819-qwen25-3b-k16
MODEL=Qwen/Qwen2.5-3B-Instruct

CUDA_VISIBLE_DEVICES=0 python -m rft.sample \
  --data data/rft/derived/train.jsonl \
  --model "${MODEL}" \
  --output "runs/rft_sampling/${RUN_ID}/candidates.jsonl" \
  --num-samples 16 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 256 \
  --gpu-memory-utilization 0.85 \
  --seed 2026
```

The sampler applies the Qwen chat template exactly once and persists raw
responses, token IDs, stop reason, prompt hashes and generation parameters.
This command is one pass over 3,080 prompts and produces 49,280 raw candidates.

## Score and build the accepted dataset

```bash
python -m rft.score_candidates \
  --candidates "runs/rft_sampling/${RUN_ID}/candidates.jsonl" \
  --data data/rft/derived/train.jsonl \
  --output "runs/rft_sampling/${RUN_ID}/scored.jsonl"

python -m rft.build_dataset \
  --scored "runs/rft_sampling/${RUN_ID}/scored.jsonl" \
  --output "data/rft/accepted/${RUN_ID}/train.jsonl" \
  --min-samples 0 \
  --max-samples 3000 \
  --seed 2026
```

Only responses with process reward `1.0`, strict JSON format and a normal EOS
are accepted. By default all accepted trajectories are retained, up to
`--max-samples`; observed and hidden sides may be incomplete. Add
`--deduplicate-semantic` to collapse equivalent JSON responses, or add
`--require-complete-pairs` for the previous pair-balanced behavior.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python -m rft.train \
  --model "${MODEL}" \
  --train-file "data/rft/accepted/${RUN_ID}/train.jsonl" \
  --output-dir "runs/rft_train/${RUN_ID}" \
  --logging-dir "runs/rft_train/${RUN_ID}/tensorboard" \
  --max-seq-length 2048 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 1 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --logging-steps 10 \
  --seed 2026
```

Before loading the model, training writes `token_length_report.json`. If any
sample exceeds 2,048 tokens, the command fails instead of truncating the story;
rerun with `--max-seq-length 4096`. The standard Hugging Face checkpoint is
written to `runs/rft_train/${RUN_ID}/final`.

TensorBoard records `loss`, `grad_norm`, `learning_rate` and `epoch` every 10
optimizer steps. Start it in another terminal on the training machine:

```bash
conda activate robusttom-rft
tensorboard \
  --logdir "runs/rft_train/${RUN_ID}/tensorboard" \
  --host 127.0.0.1 \
  --port 6006
```

For a remote A800 server, forward the port from your local machine:

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<a800-host>
```

Then open `http://127.0.0.1:6006`. Use `--logging-steps 1` for a short smoke
run where per-step visibility is more useful than lower logging overhead.

## Evaluation

```bash
python -m rft.generate \
  --data data/rft/derived/dev.jsonl \
  --model "runs/rft_train/${RUN_ID}/final" \
  --output "runs/rft_eval/${RUN_ID}/dev_predictions.jsonl"

python -m rft.evaluate \
  --predictions "runs/rft_eval/${RUN_ID}/dev_predictions.jsonl" \
  --data data/rft/derived/dev.jsonl \
  --output "runs/rft_eval/${RUN_ID}/dev_metrics.json"
```

Sealed order-4 test data must only be evaluated after model/config selection is
complete. The metrics include strict format, full process reward, answer and
core-state accuracy, pair accuracy, intervention sensitivity, shortcut-copy
rate, last-mention-copy rate, answer-state consistency, EOS rate and p95 length.

## Hi-ToM order-4 answer-only benchmark

Build the 600-example Hi-ToM order-4 benchmark from the checked-in source CSV:

```bash
python -m rft.prepare_hitom_eval \
  --input data/cleaned_tom/raw/hi_tom_3000.csv \
  --output-dir data/rft/hitom_order4 \
  --order 4 \
  --expected-count 600
```

As in the symbolic-v3 data, every event in the Hi-ToM `story` field and in the
corresponding `Story:` prompt section is written on its own line with a
one-based `1`, `2`, `3`, ... prefix.

The builder calls `build_grpo_prompt`, so every benchmark prompt contains the
same fixed order-1/2/3 few-shot block used by symbolic v3 and exactly one copy
of the following clarification:

```text
tom_order is exactly the number of names in belief_chain, not the number of story events. belief_trace contains exactly tom_order entries.
```

Generate the model responses without changing the existing generation CLI:

```bash
python -m rft.generate \
  --data data/rft/hitom_order4/test.jsonl \
  --model runs/grpo/qwen25_3b_rft_grpo_n16_seed2026/final \
  --output runs/hitom_order4/grpo_predictions.jsonl \
  --max-new-tokens 384 \
  --seed 2026
```

Score only the final JSON `answer`. Malformed JSON, a missing `answer`, and a
prediction/data ID mismatch cannot receive credit. The intermediate Hi-ToM
belief trace is generated by the model but is intentionally not scored because
the source dataset provides only final-answer labels.

```bash
python -m rft.evaluate \
  --predictions runs/hitom_order4/grpo_predictions.jsonl \
  --data data/rft/hitom_order4/test.jsonl \
  --output runs/hitom_order4/grpo_metrics.json \
  --answer-only
```

## Official Hugging Face Hi-ToM release

After downloading `Hi-ToM/Hi-ToM_Dataset` into `data/Hi-ToM`, convert the
official JSON with:

```bash
python -m rft.prepare_hitom_hf \
  --input data/Hi-ToM/Hi-ToM_data.json \
  --output-dir data/Hi-ToM \
  --expected-source-count 1200
```

The source contains 600 underlying tasks represented once as CoTP and once as
VP. `data/Hi-ToM/all_prompt_variants.jsonl` preserves all 1,200 source rows,
`data/Hi-ToM/test.jsonl` selects the 600 CoTP rows to avoid double weighting,
and `data/Hi-ToM/order4_test.jsonl` contains the 120 fourth-order examples.
The conflict-free counterparts are `data/Hi-ToM/consistent_test.jsonl` with
462 rows and `data/Hi-ToM/order4_consistent_test.jsonl` with 65 rows.

The official source has 138 duplicated CoTP/VP task pairs whose answer labels
disagree. They are recorded in `data/Hi-ToM/label_conflicts.jsonl`, and each
converted example exposes `source_label_conflict`. The recommended files retain
the upstream CoTP labels rather than silently changing labels.

For the model described here, use `data/Hi-ToM/order4_consistent_test.jsonl`
when label consistency matters more than benchmark size. Use `order4_test.jsonl`
for the complete upstream CoTP fourth-order split. Both use the same generation
CLI and final-answer-only score:

```bash
python -m rft.generate \
  --data data/Hi-ToM/order4_consistent_test.jsonl \
  --model runs/grpo/qwen25_3b_rft_grpo_n16_seed2026/final \
  --output runs/hitom_hf_order4/predictions.jsonl \
  --max-new-tokens 384 \
  --seed 2026

python -m rft.evaluate \
  --predictions runs/hitom_hf_order4/predictions.jsonl \
  --data data/Hi-ToM/order4_consistent_test.jsonl \
  --output runs/hitom_hf_order4/metrics.json \
  --answer-only
```

## Official Hugging Face ExploreToM sample

Convert the downloaded `facebook/ExploreToM` CSV with:

```bash
python -m rft.prepare_exploretom_hf \
  --input data/ExploreToM/ExploreToM-data-sample.csv \
  --output-dir data/ExploreToM \
  --expected-source-count 13309
```

The publisher describes this as a Llama-3.1-70B-targeted adversarial sample and
explicitly says it is not the canonical ExploreToM test set. It contains only
first- and second-order questions. The converter selects container-location
ToM questions, uses the structured story representation, numbers every event,
and derives deterministic choices from containers used for the queried object.

`data/ExploreToM/all_container_questions.jsonl` preserves all 1,485 matching
rows. The recommended `data/ExploreToM/test.jsonl` has 1,053 rows after removing
non-unique mental-state questions, one-candidate shortcuts, and exact duplicate
tasks. Its order-specific subsets are `order1_test.jsonl` with 422 rows and
`order2_test.jsonl` with 631 rows.

Generate and evaluate the recommended second-order subset with:

```bash
python -m rft.generate \
  --data data/ExploreToM/order2_test.jsonl \
  --model runs/grpo/qwen25_3b_rft_grpo_n16_seed2026/final \
  --output runs/exploretom_hf_order2/predictions.jsonl \
  --max-new-tokens 384 \
  --seed 2026

python -m rft.evaluate \
  --predictions runs/exploretom_hf_order2/predictions.jsonl \
  --data data/ExploreToM/order2_test.jsonl \
  --output runs/exploretom_hf_order2/metrics.json \
  --answer-only
```
