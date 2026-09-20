import ast
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from opsd.train import _sanitize_allocator_environment, _validate_cli, parse_args


ROOT = Path(__file__).resolve().parents[1]


class OPSDTrainConfigTest(unittest.TestCase):
    def test_vllm_sleep_mode_is_disabled_by_default(self):
        with patch("sys.argv", ["opsd.train", "--model", "/tmp/model"]):
            self.assertFalse(parse_args().vllm_sleep_mode)

    def test_jsd_token_clip_is_disabled_by_default(self):
        with patch("sys.argv", ["opsd.train", "--model", "/tmp/model"]):
            self.assertIsNone(parse_args().jsd_token_clip)

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
            jsd_token_clip=None,
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

    def test_non_positive_jsd_clip_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            args.jsd_token_clip = 0.0
            with self.assertRaisesRegex(ValueError, "jsd-token-clip"):
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

    def test_vllm_incompatible_allocator_setting_is_removed(self):
        with patch.dict(
            "os.environ",
            {
                "PYTORCH_CUDA_ALLOC_CONF": (
                    "max_split_size_mb:512,expandable_segments:True"
                )
            },
            clear=True,
        ):
            _sanitize_allocator_environment(vllm_sleep_mode=True)
            self.assertEqual(
                os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "max_split_size_mb:512"
            )

    def test_allocator_setting_is_preserved_when_sleep_mode_is_off(self):
        with patch.dict(
            "os.environ",
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            clear=True,
        ):
            _sanitize_allocator_environment(vllm_sleep_mode=False)
            self.assertEqual(
                os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "expandable_segments:True",
            )

    def test_training_tokenizer_uses_left_padding(self):
        tree = ast.parse((ROOT / "opsd/train.py").read_text(encoding="utf-8"))
        tokenizer_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and any(item.arg == "padding_side" for item in node.keywords)
        ]
        self.assertEqual(len(tokenizer_calls), 1)
        padding_side = next(
            item for item in tokenizer_calls[0].keywords if item.arg == "padding_side"
        )
        self.assertEqual(ast.literal_eval(padding_side.value), "left")

    def test_collator_uses_left_padding(self):
        tree = ast.parse(
            (ROOT / "opsd/data_collator.py").read_text(encoding="utf-8")
        )
        padding_assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "padding_side"
                for target in node.targets
            )
        ]
        self.assertEqual(len(padding_assignments), 1)
        self.assertEqual(ast.literal_eval(padding_assignments[0].value), "left")

    def test_jsd_sums_vocabulary_before_optional_clipping(self):
        tree = ast.parse((ROOT / "opsd/trainer.py").read_text(encoding="utf-8"))
        loss_function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "generalized_jsd_loss"
        )
        vocab_sums = [
            node
            for node in ast.walk(loss_function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "jsd"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "sum"
        ]
        clips = [
            node
            for node in ast.walk(loss_function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "jsd"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "clamp"
        ]
        self.assertEqual(len(vocab_sums), 1)
        self.assertEqual(len(clips), 1)
        self.assertLess(vocab_sums[0].lineno, clips[0].lineno)


if __name__ == "__main__":
    unittest.main()
