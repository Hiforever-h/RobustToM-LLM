# Processed ExploreToM files

The upstream Hugging Face CSV and README are preserved in this directory. The publisher explicitly says this Llama-3.1-70B-targeted sample is not the canonical ExploreToM test set.

`all_container_questions.jsonl` contains all first- and second-order container-location ToM rows. `test.jsonl` additionally requires a unique mental state, at least two target-container candidates, and removes exact duplicate tasks. `order1_test.jsonl` and `order2_test.jsonl` split that recommended set by ToM order. The official sample has no third- or fourth-order questions.

The structured `story_structure` field is used as `story`, with one numbered event per line. Candidate choices are derived only from containers used for the queried object. Every `process_prompt` has the symbolic-v3 three-shot block and exactly one copy of: `tom_order is exactly the number of names in belief_chain, not the number of story events. belief_trace contains exactly tom_order entries.`

Only final answers are supplied upstream, so no intermediate `process_target` is fabricated. Evaluate using `python -m rft.evaluate --answer-only`.
