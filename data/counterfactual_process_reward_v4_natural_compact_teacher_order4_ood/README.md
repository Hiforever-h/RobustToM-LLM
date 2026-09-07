# Order-4 OOD privileged-teacher evaluation data

This directory contains the 600-sample, 300-pair order-4 OOD split with one
teacher-only prompt augmentation:

- the verified final answer;
- one derived support event for each of the four nested-belief levels.

`test.jsonl` is directly compatible with `python -m rft.generate`. Its
`process_prompt` is the privileged teacher prompt. The original compact prompt
is preserved in `student_process_prompt`. Generator-only event IDs and role
labels are retained in `privileged_reference` for audit, but are not rendered
into the model-facing prompt.

The data was built from
`data/counterfactual_process_reward_v4_natural_compact/test.jsonl` with:

```bash
python -m scripts.build_privileged_teacher_eval_data \
  --output-dir data/counterfactual_process_reward_v4_natural_compact_teacher_order4_ood_rebuilt
```

## Deterministic generation and evaluation

Set `GRPO_MODEL` to the Hugging Face-format GRPO actor checkpoint:

```bash
export GRPO_MODEL=/path/to/grpo/actor/global_step_800
export RUN_ID=order4-privileged-teacher

python -m rft.generate \
  --data data/counterfactual_process_reward_v4_natural_compact_teacher_order4_ood/test.jsonl \
  --model "$GRPO_MODEL" \
  --output "runs/grpo_eval/$RUN_ID/test_predictions.jsonl" \
  --max-new-tokens 384 \
  --seed 2026

python -m rft.evaluate \
  --predictions "runs/grpo_eval/$RUN_ID/test_predictions.jsonl" \
  --data data/counterfactual_process_reward_v4_natural_compact_teacher_order4_ood/test.jsonl \
  --output "runs/grpo_eval/$RUN_ID/test_rule_metrics.json"
```

Do not pass `--compact-prompt` to either command: that option rebuilds the
original unprivileged compact prompt and would remove this augmentation.

The evaluator reports the existing answer, format, pair, and process metrics,
plus:

- `state_step_accuracy`: accuracy for State 1 through State 4;
- `intermediate_state_accuracy`: micro accuracy over State 1 through State 3;
- `all_intermediate_states_correct_rate`: fraction with all of State 1 through
  State 3 correct.

Because the prompt exposes the gold answer, `answer_accuracy` and State 4 are
copy-contaminated diagnostics. The two intermediate-state metrics are the main
indicators of whether the privileged context improves nested reasoning.
