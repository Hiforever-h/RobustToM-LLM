# Symbolic State-Separated Counterfactual ToM v3

This dataset is generated from symbolic movement and observation events. Labels
come from a bounded nested-belief oracle, never from an LLM. Each pair changes
one observer identity at a critical move while preserving event count and move
position. Every answer disagrees with world, initial, last, previous, and last
joint-move heuristics; order-2+ answers also disagree with outer and inner
first-order beliefs.

The source uses process-target schema `2.0` with a
`belief_trace` from the innermost suffix chain to the complete queried chain.
See `manifest.json` for shortcut, position, and token audits.

Rebuild from the repository root:

```bash
python3 scripts/generate_symbolic_counterfactual_v3.py \
  --output-dir data/counterfactual_process_reward_v3 \
  --tokenizer /path/to/Qwen2.5-tokenizer
```
