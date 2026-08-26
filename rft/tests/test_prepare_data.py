import json
import tempfile
import unittest
from pathlib import Path

from rft.common import write_jsonl
from rft.prepare_data import derive_split


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
                    {
                        "belief_chain": ["Alice", "Bob"],
                        "location": answer,
                    },
                ],
                "answer": answer,
            }
            rows.append(
                {
                    "global_sample_id": f"{split}-{intervention}",
                    "global_pair_id": f"pair-{split}",
                    "source_group_id": f"group-{split}",
                    "source_dataset": "symbolic-tom-v3",
                    "split": split,
                    "intervention_type": intervention,
                    "process_target_version": "2.0",
                    "process_target": target,
                    "process_response": json.dumps(target),
                }
            )
        return rows

    def test_derives_version_and_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for split in ("train", "val", "test"):
                write_jsonl(source / f"{split}.jsonl", self.rows(split))

            output = root / "derived"
            manifest = derive_split(source, output, seed=2026, holdout_pairs=0)
            self.assertEqual(manifest["process_target_version"], "2.0")
            self.assertEqual(manifest["split_counts"], {"train": 2, "dev": 2, "test": 2})
            with self.assertRaises(FileExistsError):
                derive_split(source, output, seed=2026, holdout_pairs=0)


if __name__ == "__main__":
    unittest.main()
