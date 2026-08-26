import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from rft.train import parse_args


class TrainConfigTest(unittest.TestCase):
    def test_accepts_tensorboard_logging_directory(self):
        argv = [
            "rft.train",
            "--model",
            "model",
            "--train-file",
            "train.jsonl",
            "--output-dir",
            "run",
            "--logging-dir",
            "run/events",
            "--logging-steps",
            "2",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.logging_dir, Path("run/events"))
        self.assertEqual(args.logging_steps, 2)


if __name__ == "__main__":
    unittest.main()
