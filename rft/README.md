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

The current actor prompt version is
`natural-cot-think-state-v2-exact-order`. For every row it states the exact
ToM order, renders the outermost-to-innermost belief chain, maps each `Think N`
to its belief level, and includes exactly that many empty output blocks. It has
no fixed three-step demonstration and does not expose any gold location.

There must be exactly one numbered block per gold trace step, in order, one
`State:` per block, one final `Answer:`, no duplicate/extra steps, and no text
outside the blocks. Markers must begin on a new line.

## Reward and acceptance

RFT candidate selection is local and deterministic by default; it makes zero
LLM Judge requests. The parser from `scripts/reward.py` checks every `State` and
the final `Answer`. If all are correct, the candidate receives binary reward
1.0; otherwise it receives 0.0.

A reward-1 candidate is accepted only when the complete
`Think/State/Answer` structure is valid and generation ended normally with EOS.
Reasoning text is retained for training but is not semantically graded during
default RFT rejection sampling.

The previous Judge-backed scorer remains available only as an explicit
`--use-judge` option. In that optional mode, `--min-reward` defaults to 0.88 and
every effective step score must meet `--min-reasoning-score` (default 0.5).

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

Default scoring needs no API key. Optional `--use-judge` scoring reads
`DEEPSEEK_API_KEY` from the environment or repository `.env`;
`DEEPSEEK_BASE_URL` may also be set there.

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

## Build the balanced pilot split

The checked-in pilot contains 50 complete observed/hidden pairs for each of
orders 1, 2, and 3: 300 prompts total. With 16 samples per prompt it produces
4,800 candidates.

```bash
python -m scripts.sample_rft_pilot \
  --input data/rft/derived_v3_fewshot/train.jsonl \
  --output data/rft/pilot_v2/train.jsonl \
  --pairs-per-order 50 \
  --orders 1 2 3 \
  --seed 2026 \
  --num-samples-per-prompt 16
```

Selection ranks complete pairs with a stable hash, so it is independent of
input row order and reproducible from the source hash recorded in the pilot
manifest.

## Sample candidates

Candidates are bound to the exact actor prompt hash. Runs sampled with the old
v1 fixed three-step example, including the `20260826-qwen25-3b-natural-k16`
run, must not be rescored against this v2 data; sample a new run instead.

```bash
RUN_ID=20260828-qwen25-3b-natural-v2-k16
MODEL=Qwen/Qwen2.5-3B-Instruct

CUDA_VISIBLE_DEVICES=0 python -m rft.sample \
  --data data/rft/pilot_v2/train.jsonl \
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
  --output "runs/rft_sampling/${RUN_ID}/scored.jsonl"

python -m rft.build_dataset \
  --scored "runs/rft_sampling/${RUN_ID}/scored.jsonl" \
  --output "data/rft/accepted/${RUN_ID}/train.jsonl" \
  --min-samples 0 \
  --max-samples 3000 \
  --seed 2026 \
  --compact-prompt
```

The scored JSONL keeps the binary score, deterministic rule details, acceptance
reason, and policy for auditing. `judge_score` and `judge_metadata` are null in
the default mode.
The dataset builder consumes the resulting `accepted` flag; it does not apply a
second score policy. It trains only on accepted sampled responses and never
falls back to a gold response.

With `--compact-prompt`, the builder reconstructs each training prompt from
`judge_prompt` and appends only a concise output contract: produce exactly the
number of `Think` blocks implied by the question's ToM order, one `State:` per
block, and one final `Answer:`. The prompt defines N as the number of nested
belief levels and requires the model to infer N from the question; it never
injects `question_order` or `process_target.tom_order`. It does not retain the
v2 reasoning-rules section, empty output scaffold, angle-bracket placeholders,
or any gold state. The accepted sampled response is unchanged. Output rows are marked
`natural-cot-think-state-v3-compact` and retain the source prompt hash for
auditability. Omitting the flag preserves the sampled v2 actor prompt.

By default, every accepted trajectory is eligible and incomplete
observed/hidden pair coverage is allowed. Optional controls are:

- `--max-candidates-per-prompt N` to cap trajectories per prompt;
- `--deduplicate-semantic` to collapse normalized equivalent natural responses;
- `--require-complete-pairs` to keep only prompts whose pair has both sides.

Optional Judge scoring must be requested explicitly and can make one request
per prompt group, so it is not recommended for the full 3,200 x 16 selection
run:

```bash
python -m rft.score_candidates \
  --candidates "runs/rft_sampling/${RUN_ID}/candidates.jsonl" \
  --data data/rft/derived_v3_fewshot/train.jsonl \
  --output "runs/rft_sampling/${RUN_ID}/judged.jsonl" \
  --use-judge \
  --cache-dir "runs/rft_sampling/${RUN_ID}/judge_cache" \
  --min-reward 0.88 \
  --min-reasoning-score 0.5
```

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
  --max-new-tokens 384 \
  --compact-prompt
```

Use `--compact-prompt` at generation time when the training dataset was built
with it, so SFT and inference see the same prompt contract.

Deterministic evaluation makes no Judge call and reports structure, reasoning
presence, per-step/all-state accuracy, answer accuracy, pair and shortcut
metrics. Its `mean_process_reward` is the rule-only combined score, so correct
reasoning receives no Judge credit and a completely correct structure/state/
answer response has a rule-only score of 0.52.

```bash
python -m rft.evaluate \
  --predictions "runs/rft_eval/${RUN_ID}/dev_predictions.jsonl" \
  --data data/rft/derived_v3_fewshot/dev.jsonl \
  --output "runs/rft_eval/${RUN_ID}/dev_rule_metrics.json" \
  --compact-prompt
```

Add `--judge` for the true combined reward and full-reward rate:

```bash
python -m rft.evaluate \
  --predictions "runs/rft_eval/${RUN_ID}/dev_predictions.jsonl" \
  --data data/rft/derived_v3_fewshot/dev.jsonl \
  --output "runs/rft_eval/${RUN_ID}/dev_judged_metrics.json" \
  --compact-prompt \
  --judge \
  --cache-dir "runs/rft_eval/${RUN_ID}/judge_cache" \
  --max-workers 8
```

Keep the order-4 `test.jsonl` sealed until model and configuration selection is
complete. `--answer-only` accepts either the current `Answer:` marker or a
legacy JSON `answer`, which keeps the external answer-only benchmark utilities
usable.

## One-command continuation from existing candidates

`run_rft.sh` starts from an existing candidate file, applies the default local
State+Answer scoring, builds accepted data, and trains. It does not call the
Judge, resample, or fabricate completions.

```bash
CANDIDATES=/path/to/candidates.jsonl \
RUN_ID=20260826-natural-rft \
COMPACT_PROMPT=1 \
bash rft/run_rft.sh
```

`COMPACT_PROMPT=1` passes `--compact-prompt` to the dataset builder. It changes
only the SFT conditioning prompts; scoring still validates candidates against
the exact v2 prompts used during sampling.
