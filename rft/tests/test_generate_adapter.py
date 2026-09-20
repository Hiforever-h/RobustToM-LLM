import json
import tempfile
import unittest
from pathlib import Path

from rft.generate import load_adapter_spec


class GenerateAdapterTest(unittest.TestCase):
    def test_no_adapter(self):
        self.assertEqual(load_adapter_spec(None), (None, None))

    def test_lora_adapter_path_and_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "r": 64}),
                encoding="utf-8",
            )
            path, rank = load_adapter_spec(adapter)
            self.assertEqual(path, str(adapter.resolve()))
            self.assertEqual(rank, 64)

    def test_adapter_requires_valid_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "r": 0}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_adapter_spec(adapter)

    def test_non_lora_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "IA3", "r": 64}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_adapter_spec(adapter)


if __name__ == "__main__":
    unittest.main()
