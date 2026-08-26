import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_hi_tom_counterfactual.py"
SPEC = importlib.util.spec_from_file_location("builder", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


class BuilderTest(unittest.TestCase):
    def make_source(self, path: Path) -> None:
        story = (
            "1 Alice, Bob, Carol, David and Eve entered the room.\n"
            "2 The apple is in the red_box.\n"
            "3 Alice moved the apple to the blue_box.\n"
            "4 Alice exited the room.\n"
            "5 Bob exited the room.\n"
            "6 Carol exited the room.\n"
            "7 David exited the room.\n"
            "8 Eve moved the apple to the green_box.\n"
            "9 Eve exited the room.\n"
            "10 Alice, Bob, Carol, David and Eve entered the waiting_room.\n\n"
        )
        questions = [
            ("Where is the apple really?", "green_box"),
            ("Where does Alice really think the apple is?", "blue_box"),
            ("Where does Bob think Alice thinks the apple is?", "blue_box"),
            ("Where does Carol think Bob thinks Alice thinks the apple is?", "blue_box"),
            (
                "Where does David think Carol thinks Bob thinks Alice thinks the apple is?",
                "blue_box",
            ),
        ]
        choices = (
            "A. red_box, B. blue_box, C. green_box, D. yellow_box, "
            "E. white_box, F. black_box"
        )
        prefix = "Read the story.\nStory:\n"
        suffix = "\nNote: Track what each person observes.\n"
        rows = []
        for order, (question, answer) in enumerate(questions):
            rows.append(
                {
                    "prompting_type": "CoTP",
                    "deception": False,
                    "story_length": 1,
                    "question_order": order,
                    "sample_id": order * 20,
                    "story": story,
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "prompt": (
                        f"{prefix}{story}\nQuestion: {question}\nChoices: {choices}"
                        f"\n{suffix}"
                    ),
                }
            )
        path.write_text(json.dumps({"data": rows}), encoding="utf-8")

    def test_builds_observed_hidden_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.json"
            self.make_source(source)
            records, metadata = builder.build_dataset(source, 42, 0.2, "CoTP")

        self.assertEqual(len(records), 10)
        self.assertEqual(metadata["source_story_groups"], 1)
        builder.validate_records(records)

        order_four = [row for row in records if row["question_order"] == 4]
        observed = next(row for row in order_four if row["intervention_type"] == "observed")
        hidden = next(row for row in order_four if row["intervention_type"] == "hidden")
        self.assertEqual(observed["answer"], observed["counterfactual_container"])
        self.assertEqual(hidden["answer"], hidden["counterfactual_anchor_container"])
        self.assertNotEqual(hidden["answer"], hidden["original_answer"])
        self.assertTrue(observed["shortcut_conflict"])
        self.assertTrue(hidden["shortcut_conflict"])
        self.assertTrue(hidden["last_mention_conflict"])


if __name__ == "__main__":
    unittest.main()
