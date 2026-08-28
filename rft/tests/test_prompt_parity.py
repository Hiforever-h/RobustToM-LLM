import unittest

from rft.generate import prepare_generation_rows
from rft.prompt import (
    COMPACT_NATURAL_COT_PROMPT_VERSION,
    build_compact_process_prompt,
    format_chat_prompt,
)
from rft.sample import prepare_sampling_rows
from scripts.build_compact_rft_data import build_compact_data


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        return f"<user>{messages[0]['content']}</user><assistant>"


class PromptParityTest(unittest.TestCase):
    def test_template_is_single_and_deterministic(self):
        tokenizer = FakeTokenizer()
        first = format_chat_prompt(tokenizer, "Return JSON.")
        second = format_chat_prompt(tokenizer, "Return JSON.")
        self.assertEqual(first, second)
        self.assertEqual(first.count("<assistant>"), 1)

    def test_compact_prompt_contains_only_question_and_format_contract(self):
        record = {
            "global_sample_id": "sample-3",
            "question_order": 3,
            "judge_prompt": "Story and question. Choices: A. chest, B. drawer",
            "process_prompt": "Story and question.\n\nReasoning rules:\nold rules and scaffold",
            "process_target": {"tom_order": 3, "answer": "chest"},
        }
        compact = build_compact_process_prompt(record)
        self.assertTrue(compact.startswith(record["judge_prompt"] + "\n\n"))
        self.assertIn("N means the theory-of-mind (ToM) order implied", compact)
        self.assertIn("number of nested belief levels", compact)
        self.assertIn("Infer N from the question itself", compact)
        self.assertIn("exactly N numbered reasoning blocks", compact)
        self.assertIn("using `Think 1:`, `Think 2:`, and so on", compact)
        self.assertIn("never the letter N", compact)
        self.assertNotIn("N = 3", compact)
        self.assertIn("`State:`", compact)
        self.assertIn("`Answer:`", compact)
        self.assertNotIn("Reasoning rules:", compact)
        self.assertNotIn("old rules and scaffold", compact)
        self.assertNotIn("<", compact)
        self.assertNotIn(">", compact)

    def test_generation_compact_prompt_uses_shared_builder_without_mutation(self):
        original = {
            "global_sample_id": "sample-2",
            "question_order": 2,
            "judge_prompt": "Question only.",
            "process_prompt": "Old actor prompt.",
            "process_prompt_version": "old-version",
            "process_target": {"tom_order": 2, "answer": "locker"},
        }
        prepared = prepare_generation_rows([original], compact_prompt=True)
        self.assertEqual(original["process_prompt"], "Old actor prompt.")
        self.assertNotEqual(prepared[0]["process_prompt"], original["process_prompt"])
        self.assertEqual(
            prepared[0]["process_prompt_version"],
            COMPACT_NATURAL_COT_PROMPT_VERSION,
        )

        sampled = prepare_sampling_rows([original], compact_prompt=True)
        fixed_data = build_compact_data([original])
        self.assertEqual(sampled, prepared)
        self.assertEqual(fixed_data, prepared)

    def test_compact_prompt_does_not_require_or_reveal_target_order(self):
        compact = build_compact_process_prompt(
            {
                "global_sample_id": "inference-row",
                "judge_prompt": "Question without a supplied target.",
            }
        )
        self.assertIn("Infer N from the question itself", compact)
        self.assertNotIn("N =", compact)


if __name__ == "__main__":
    unittest.main()
