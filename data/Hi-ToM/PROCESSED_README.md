# Processed Hi-ToM files

The upstream Hugging Face files in this directory are preserved. `all_prompt_variants.jsonl` converts all 1,200 source rows. `test.jsonl` selects the 600 CoTP rows so the duplicated VP prompt variant does not double-weight the same tasks. `order4_test.jsonl` contains the 120 fourth-order rows from that canonical selection. `consistent_test.jsonl` and `order4_consistent_test.jsonl` remove tasks whose two upstream answer labels disagree.

The source contains 138 CoTP/VP task pairs with conflicting answer labels; `label_conflicts.jsonl` records them without changing either upstream label. The recommended files retain the CoTP label and expose `source_label_conflict` on every record. Hi-ToM supplies only final answers, so no intermediate `process_target` is fabricated and evaluation must use `python -m rft.evaluate --answer-only`.

Every story event has a one-based line number. Every `process_prompt` contains the symbolic-v3 three-shot block and exactly one copy of: `tom_order is exactly the number of names in belief_chain, not the number of story events. belief_trace contains exactly tom_order entries.`
