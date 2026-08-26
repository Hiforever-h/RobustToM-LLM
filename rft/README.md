# Standalone natural-CoT RFT

`rft/` implements rejection-sampling fine-tuning without `verl`. The current
pipeline consumes `data/counterfactual_process_reward_v4_natural` and uses
`scripts/reward.py` as the reward implementation. Model responses are natural
`Think N / State / Answer` text; they are not JSON, and no canonical
`process_response` is used as training data.

`rft/reward.py` remains only for the repository's legacy JSON-reward paths.
The current RFT preparation, scoring, dataset construction, and evaluation
commands import the natural-CoT parser/reward from `scripts/reward.py`.

## Data and response format

The checked-in RFT split is `data/rft/derived_v3_fewshot`:

- `train.jsonl`: the natural source `train.jsonl` (3,200 examples)
- `dev.jsonl`: the natural source `val.jsonl` (400 examples), with `split=dev`
- `test.jsonl`: the natural source `test.jsonl` (600 sealed order-4 examples)

These files contain the same sample IDs, pair IDs, questions, choices, answers,
and `process_target` values as the previous symbolic-v3 few-shot files. They are
not an independently sampled dataset. The natural v4 source removes story event
numbers, supplies the current natural `process_prompt`, adds `judge_prompt`, and
removes `process_response`, token-count fields, and the old `prompt` field.

The actor response protocol is:

```text
Think 1:
<natural-language reasoning for the innermost belief>
State: <location>
Think 2:
<natural-language reasoning for the next outer belief>
State: <location>
Answer: <final outermost location>
```

There must be exactly one numbered block per gold trace step, in order, one
`State:` per block, one final `Answer:`, no duplicate/extra steps, and no text
outside the blocks. Markers must begin on a new line.

## Reward and acceptance

`scripts/reward.py` combines deterministic checks with a packed pointwise LLM
Judge. For every prompt, all sampled candidates are sent in one Judge request.
The deterministic part checks structure, every `State`, and the final `Answer`;
the Judge grades only the natural-language reasoning at each step.

With the default weights:

- process reward is 0.8 of the total and answer bonus is 0.2;
- within each process step, state correctness is weighted 0.4 and Judge
  reasoning quality is weighted 0.6;
- reasoning scores may be any finite value in `[0, 1]`; 0, 0.5, and 1 are
  Judge calibration anchors;
- a wrong state or missing reasoning gates that step's reasoning score;
- the answer bonus requires every state and the final answer to be correct.

RFT accepts a candidate only when all conditions hold:

1. combined reward is at least `--min-reward` (default 0.88);
2. every effective per-step reasoning score is at least
   `--min-reasoning-score` (default 0.5);
3. the complete `Think/State/Answer` structure is valid;
4. every `State` and the final `Answer` are correct;
5. generation ended normally with EOS.

With the default reward weights and correct states/answer, reward 0.88
corresponds to an average reasoning score of 0.75. The per-step 0.5 floor stops
one clearly bad step from being hidden by high scores on the other steps.
Thresholds change acceptance only; accepted samples keep their actual reward
and are never rewritten as reward 1.0.

`--rule-only` is a diagnostic mode. It supplies zero Judge reasoning scores and
therefore intentionally accepts no candidates.

## Environment

Run from the repository root. GPU sampling/training needs the packages in
`rft/requirements.txt`; the reward implementation itself uses the standard
library.

```bash
conda create -n robusttom-rft python=3.10 -y
conda activate robusttom-rft
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r rft/requirements.txt
python -m pip install flash-attn==2.6.3 --no-build-isolation  # optional
```

Judge scoring reads `DEEPSEEK_API_KEY` from the environment or repository
`.env`. `DEEPSEEK_BASE_URL` may also be set there. The CLI defaults to model
`deepseek-v4-flash`, thinking disabled, eight concurrent prompt groups, two
retries, and a 180-second request timeout.

## Rebuild or validate the fixed split

`rft.prepare_data` refuses to overwrite an existing output directory. Use a
fresh path when auditing a rebuild:

```bash
python -m rft.prepare_data \
  --input-dir data/counterfactual_process_reward_v4_natural \
  --output-dir /tmp/derived_v4_natural_audit
```

The command verifies natural prompt/version fields, absence of
`process_response`, target/answer agreement, complete observed/hidden pairs,
and split isolation. It maps source `val` to RFT `dev` without resampling.

## Sample candidates

```bash
RUN_ID=20260826-qwen25-3b-natural-k16
MODEL=Qwen/Qwen2.5-3B-Instruct

CUDA_VISIBLE_DEVICES=0 python -m rft.sample \
  --data data/rft/derived_v3_fewshot/train.jsonl \
  --model "${MODEL}" \
  --output "runs/rft_sampling/${RUN_ID}/candidates.jsonl" \
  --num-samples 16 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 384 \
  --gpu-memory-utilization 0.85 \
  --seed 2026
```

The sampler applies the tokenizer's chat template once and records the raw
response, response token IDs, EOS status, prompt hashes, and generation config.

## Score and build the accepted dataset

```bash
python -m rft.score_candidates \
  --candidates "runs/rft_sampling/${RUN_ID}/candidates.jsonl" \
  --data data/rft/derived_v3_fewshot/train.jsonl \
  --output "runs/rft_sampling/${RUN_ID}/scored.jsonl" \
  --cache-dir "runs/rft_sampling/${RUN_ID}/judge_cache" \
  --max-workers 8 \
  --judge-model deepseek-v4-flash \
  --min-reward 0.88 \
  --min-reasoning-score 0.5

python -m rft.build_dataset \
  --scored "runs/rft_sampling/${RUN_ID}/scored.jsonl" \
  --output "data/rft/accepted/${RUN_ID}/train.jsonl" \
  --min-samples 0 \
  --max-samples 3000 \
  --seed 2026
```

The scored JSONL keeps the actual combined score, deterministic rule details,
Judge step scores, acceptance reason/policy, and Judge metadata for auditing.
The dataset builder consumes the resulting `accepted` flag; it does not apply a
second score policy. It trains only on accepted sampled responses and never
falls back to a gold response.

By default, every accepted trajectory is eligible and incomplete
observed/hidden pair coverage is allowed. Optional controls are:

- `--max-candidates-per-prompt N` to cap trajectories per prompt;
- `--deduplicate-semantic` to collapse normalized equivalent natural responses;
- `--require-complete-pairs` to keep only prompts whose pair has both sides.

For a local parser/state diagnostic without API calls:

```bash
python -m rft.score_candidates \
  --candidates "runs/rft_sampling/${RUN_ID}/candidates.jsonl" \
  --data data/rft/derived_v3_fewshot/train.jsonl \
  --output "runs/rft_sampling/${RUN_ID}/rule_only.jsonl" \
  --rule-only
```

This output is for inspection only and contains no accepted samples.

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

Training masks every prompt token and optimizes only the accepted response plus
one EOS. It writes `token_length_report.json` before model loading and fails on
overlength samples rather than truncating them. The final Hugging Face
checkpoint is written under `<output-dir>/final`.

## Evaluate

Generate on `dev` during model selection:

```bash
python -m rft.generate \
  --data data/rft/derived_v3_fewshot/dev.jsonl \
  --model "runs/rft_train/${RUN_ID}/final" \
  --output "runs/rft_eval/${RUN_ID}/dev_predictions.jsonl" \
  --max-new-tokens 384
```

Deterministic evaluation makes no Judge call and reports structure, reasoning
presence, per-step/all-state accuracy, answer accuracy, pair and shortcut
metrics. Its `mean_process_reward` is the rule-only combined score, so correct
reasoning receives no Judge credit and a completely correct structure/state/
answer response has a rule-only score of 0.52.

```bash
python -m rft.evaluate \
  --predictions "runs/rft_eval/${RUN_ID}/dev_predictions.jsonl" \
  --data data/rft/derived_v3_fewshot/dev.jsonl \
  --output "runs/rft_eval/${RUN_ID}/dev_rule_metrics.json"
```

Add `--judge` for the true combined reward and full-reward rate:

```bash
python -m rft.evaluate \
  --predictions "runs/rft_eval/${RUN_ID}/dev_predictions.jsonl" \
  --data data/rft/derived_v3_fewshot/dev.jsonl \
  --output "runs/rft_eval/${RUN_ID}/dev_judged_metrics.json" \
  --judge \
  --cache-dir "runs/rft_eval/${RUN_ID}/judge_cache" \
  --max-workers 8
```

Keep the order-4 `test.jsonl` sealed until model and configuration selection is
complete. `--answer-only` accepts either the current `Answer:` marker or a
legacy JSON `answer`, which keeps the external answer-only benchmark utilities
usable.

## One-command continuation from existing candidates

`run_rft.sh` starts from an existing candidate file, scores with the Judge,
builds accepted data, and trains. It does not resample or fabricate completions.

```bash
CANDIDATES=/path/to/candidates.jsonl \
RUN_ID=20260826-natural-rft \
bash rft/run_rft.sh
```
