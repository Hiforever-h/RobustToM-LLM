import unittest

from rft.dataset import build_response_only_example


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "P:" + messages[0]["content"] + ":A"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) % 31 + 1 for char in text]}


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.row = {
            "global_sample_id": "sample-1",
            "global_pair_id": "pair-1",
            "process_prompt": "Return JSON",
            "accepted_response": '{"answer":"box"}',
        }

    def test_masks_prompt_and_trains_response_plus_eos(self):
        item = build_response_only_example(self.row, self.tokenizer, max_length=100)
        prompt_length = item["prompt_length"]
        self.assertTrue(all(label == -100 for label in item["labels"][:prompt_length]))
        self.assertEqual(item["labels"][prompt_length:], item["input_ids"][prompt_length:])
        self.assertEqual(item["labels"][-1], self.tokenizer.eos_token_id)
        self.assertEqual(item["labels"].count(self.tokenizer.eos_token_id), 1)

    def test_preserves_response_whitespace(self):
        row = dict(self.row, accepted_response=' {"answer":"box"} ')
        item = build_response_only_example(row, self.tokenizer, max_length=100)
        expected = self.tokenizer(row["accepted_response"], add_special_tokens=False)["input_ids"]
        self.assertEqual(item["input_ids"][item["prompt_length"]:-1], expected)

    def test_overlength_raises_instead_of_truncating(self):
        with self.assertRaises(ValueError):
            build_response_only_example(self.row, self.tokenizer, max_length=2)


if __name__ == "__main__":
    unittest.main()
