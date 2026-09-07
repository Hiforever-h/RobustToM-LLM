# Validation privileged-teacher evaluation data

This directory contains the 400-sample, 200-pair validation split augmented
with the same teacher-only information as the order-4 OOD experiment:

- the verified final answer;
- one derived support event for each nested-belief level.

The split contains 100 order-1, 200 order-2, and 100 order-3 examples.
`val.jsonl` is directly compatible with `python -m rft.generate`. Its
`process_prompt` is the privileged teacher prompt, while the original compact
prompt is preserved in `student_process_prompt`.

The data was built with:

```bash
python -m scripts.build_privileged_teacher_eval_data \
  --input data/counterfactual_process_reward_v4_natural_compact/val.jsonl \
  --output-dir data/counterfactual_process_reward_v4_natural_compact_teacher_val
```

## Generate and evaluate

Set `GRPO_MODEL` to the same GRPO actor checkpoint used for the order-4 run:

```bash
export GRPO_MODEL=/path/to/grpo/actor/global_step_800
export RUN_ID=val-privileged-teacher

python -m rft.generate \
  --data data/counterfactual_process_reward_v4_natural_compact_teacher_val/val.jsonl \
  --model "$GRPO_MODEL" \
  --output "runs/grpo_eval/$RUN_ID/val_predictions.jsonl" \
  --max-new-tokens 384 \
  --seed 2026

python -m rft.evaluate \
  --predictions "runs/grpo_eval/$RUN_ID/val_predictions.jsonl" \
  --data data/counterfactual_process_reward_v4_natural_compact_teacher_val/val.jsonl \
  --output "runs/grpo_eval/$RUN_ID/val_rule_metrics.json"
```

Do not pass `--compact-prompt`, because it would rebuild the original student
prompt and remove the privileged context. Compare block counts and strict
format rates separately under the evaluator's `question_order` groups.
