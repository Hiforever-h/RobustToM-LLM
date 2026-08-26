import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "generate_exploretom_counterfactual_1200.py"
SPEC = importlib.util.spec_from_file_location("exploretom_generator", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generator)


class ExploreToMGeneratorTest(unittest.TestCase):
    def test_small_generation_is_paired_and_balanced(self) -> None:
        repo = Path(__file__).parents[1] / "tmp" / "ExploreToM"
        records = generator.generate_records(repo, seed=7, num_pairs=6)
        generator.validate(records, num_pairs=6)
        self.assertEqual(len(records), 12)
        self.assertEqual(sum(row["last_container_conflict"] for row in records), 6)
        self.assertTrue(all(len(set(row["rooms"])) == 2 for row in records))


if __name__ == "__main__":
    unittest.main()
