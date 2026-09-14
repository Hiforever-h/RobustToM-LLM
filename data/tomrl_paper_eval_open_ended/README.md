# ToM-RL paper evaluation: open-ended positive-order split

This directory is generated from `eval_tom/tom_eval_datasets.csv`. It contains only test rows whose inferred `question_order >= 1`; no training rows or synthetic validation rows are included.

Prompts use the project's natural `Think/State/Answer` protocol but omit `Choices`. The corresponding State rule asks for the exact location supported by the story. The upstream data supplies final answers but no intermediate belief-state targets, so evaluation must use `python -m rft.evaluate --answer-only`.

`test.jsonl` contains all retained rows. The other JSONL files split the same rows by the five source benchmarks reported by ToM-RL.
