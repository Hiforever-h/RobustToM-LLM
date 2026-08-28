import tempfile
import unittest
from pathlib import Path

from rft.common import write_jsonl
from rft.prepare_data import derive_split
from rft.prompt import NATURAL_COT_PROMPT_VERSION


class PrepareDataTest(unittest.TestCase):
    @staticmethod
    def rows(split: str) -> list[dict]:
        rows = []
        for intervention, answer in (
            ("observed", "archive drawer"),
            ("hidden", "linen chest"),
        ):
            target = {
                "tom_order": 2,
                "belief_chain": ["Alice", "Bob"],
                "object": "passport",
                "reasoning_mode": "nested_belief",
                "belief_trace": [
                    {"belief_chain": ["Bob"], "location": "metal trunk"},
                    {"belief_chain": ["Alice", "Bob"], "location": answer},
                ],
                "answer": answer,
            }
            rows.append(
                {
                    "global_sample_id": f"{split}-{intervention}",
                    "global_pair_id": f"pair-{split}",
                    "source_group_id": f"group-{split}",
                    "source_dataset": "symbolic-tom-v3",
                    "source_split": split,
                    "split": split,
                    "question_order": 2,
                    "intervention_type": intervention,
                    "process_target_version": "2.0",
                    "process_target": target,
                    "answer": answer,
                    "process_prompt": "Think/State/Answer actor prompt",
                    "judge_prompt": "Story and question only",
                    "process_prompt_version": NATURAL_COT_PROMPT_VERSION,
                }
            )
        return rows

    def test_derives_natural_split_and_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for split in ("train", "val", "test"):
                write_jsonl(source / f"{split}.jsonl", self.rows(split))

            output = root / "derived"
            manifest = derive_split(source, output)
            self.assertEqual(manifest["process_target_version"], "2.0")
            self.assertFalse(manifest["contains_process_response"])
            self.assertEqual(
                manifest["split_counts"], {"train": 2, "dev": 2, "test": 2}
            )
            dev_rows = (output / "dev.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("process_response", dev_rows)
            self.assertIn('"split":"dev"', dev_rows)
            with self.assertRaises(FileExistsError):
                derive_split(source, output)


if __name__ == "__main__":
    unittest.main()
