import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_hi_tom_counterfactual_3000.py"
SPEC = importlib.util.spec_from_file_location("generator_3000", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generator)


class SplitTest(unittest.TestCase):
    def test_assigns_exact_number_of_validation_groups(self) -> None:
        groups = {}
        for deception in (False, True):
            for length in (1, 2, 3):
                for scenario in range(50):
                    group = f"{deception}-{length}-{scenario}"
                    groups[group] = [
                        {"deception": deception, "story_length": length}
                    ]
        splits = generator.assign_lower_splits(groups, 2026, validation_groups=50)
        self.assertEqual(sum(split == "validation" for split in splits.values()), 50)
        self.assertEqual(sum(split == "train" for split in splits.values()), 250)


if __name__ == "__main__":
    unittest.main()
