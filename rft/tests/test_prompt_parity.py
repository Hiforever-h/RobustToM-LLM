import unittest

from rft.prompt import format_chat_prompt


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


if __name__ == "__main__":
    unittest.main()
