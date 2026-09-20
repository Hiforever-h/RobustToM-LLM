import ast
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from opsd.train import _validate_cli


ROOT = Path(__file__).resolve().parents[1]


class OPSDTrainConfigTest(unittest.TestCase):
    def _args(self, root: Path) -> Namespace:
        data = root / "train.jsonl"
        data.write_text("{}\n", encoding="utf-8")
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        return Namespace(
            max_steps=100,
            max_completion_length=384,
            max_sequence_length=3072,
            per_device_batch_size=2,
            gradient_accumulation_steps=16,
            vllm_gpu_memory_utilization=0.30,
            jsd_token_clip=1e-6,
            data=data,
            model=str(model),
            output_dir=root / "output",
            resume_from_checkpoint=None,
        )

    def test_manifest_only_output_directory_can_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            args.output_dir.mkdir()
            (args.output_dir / "run_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            _validate_cli(args)

    def test_existing_training_artifact_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            args.output_dir.mkdir()
            (args.output_dir / "checkpoint-50").mkdir()
            with self.assertRaises(FileExistsError):
                _validate_cli(args)

    def test_raw_opsd_columns_bypass_sft_dataset_preparation(self):
        tree = ast.parse((ROOT / "opsd/train.py").read_text(encoding="utf-8"))
        gold_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "GOLDConfig"
        ]
        self.assertEqual(len(gold_calls), 1)
        keyword = next(
            item for item in gold_calls[0].keywords if item.arg == "dataset_kwargs"
        )
        self.assertEqual(
            ast.literal_eval(keyword.value), {"skip_prepare_dataset": True}
        )


if __name__ == "__main__":
    unittest.main()
