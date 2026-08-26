# Natural-CoT v4 RFT split

This directory is the RFT train/dev/test view of
`data/counterfactual_process_reward_v4_natural`.

The datasets have the same origin: every sample ID and pair ID matches the
previous symbolic-v3 few-shot split, and the question, choices, answer, and
`process_target` are unchanged. Natural v4 removes numbered story events,
replaces the actor `process_prompt` with the `Think N / State / Answer` prompt,
adds `judge_prompt`, and removes the canonical `process_response`, legacy
`prompt`, and old token-count fields.

Split mapping is `train -> train`, `val -> dev`, and `test -> test`; there is no
resampling. The `split` field of the 400 validation rows is changed to `dev`,
while `source_split` remains `val` for provenance.
