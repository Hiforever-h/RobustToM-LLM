import importlib.util
import json
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "build_counterfactual_process_reward_v2.py"
SPEC = importlib.util.spec_from_file_location("counterfactual_v2", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


class CounterfactualV2Test(unittest.TestCase):
    def process_fields(self, order: int, observed: bool) -> tuple[dict, str]:
        target = {
            "tom_order": order,
            "belief_chain": ["Alice"] if order else [],
            "object": "apple",
            "reasoning_mode": "belief" if order else "world_state",
        }
        if order:
            target.update(
                final_move_observed=observed,
                nested_belief="black_box" if observed else "white_box",
                answer="black_box" if observed else "white_box",
            )
        else:
            target.update(world_state="black_box", answer="black_box")
        return target, json.dumps(target, separators=(",", ":"))

    def hi_record(self, intervention: str) -> dict:
        observed = intervention == "observed"
        target, response = self.process_fields(1, observed)
        base = (
            "1 Alice, Bob, Carol, David and Eve entered the room.\n"
            "2 The apple is in the red_box.\n"
            "3 Bob moved the apple to the blue_box.\n"
            "4 Alice exited the room.\n"
        )
        suffix = (
            "5 Alice, Bob, Carol, David, and Eve entered the observation_room together.\n"
            "6 An automated system moved the apple to the white_box.\n"
            "7 Everyone received an explicit observation cue.\n"
            "8 Alice left before or after the next event.\n"
            "9 An automated system then moved the apple to the black_box.\n"
            "10 Every listed observer saw the final move.\n"
        )
        return {
            "source_dataset": "hi-tom",
            "global_sample_id": f"hi-tom:test-{intervention}",
            "global_pair_id": "hi-tom:test",
            "source_group_id": "group-1",
            "split": "train",
            "question_order": 1,
            "question": "Where does Alice think the apple is?",
            "choices": "A. red_box, B. white_box, C. black_box",
            "answer": "black_box" if observed else "white_box",
            "intervention_type": intervention,
            "counterfactual_anchor_container": "white_box",
            "counterfactual_container": "black_box",
            "story": base + suffix,
            "process_target": target,
            "process_response": response,
        }

    def explore_record(self, intervention: str) -> dict:
        observed = intervention == "observed"
        target, response = self.process_fields(1, observed)
        common = (
            "Eve entered the hall. Alice entered the room. Bob entered the room. "
            "Carol entered the room. Carol moved the apple to the red_box, which "
            "is also located in the room. Carol moved the apple to the white_box, "
            "which is also located in the room. "
        )
        if observed:
            suffix = (
                "Carol moved the apple to the black_box, which is also located in "
                "the room. Alice left the room. Alice entered the hall."
            )
        else:
            suffix = (
                "Alice left the room. Alice entered the hall. Carol moved the "
                "apple to the black_box, which is also located in the room."
            )
        return {
            "source_dataset": "exploretom",
            "global_sample_id": f"exploretom:test-{intervention}",
            "global_pair_id": "exploretom:test",
            "split": "train",
            "question_order": 1,
            "question": "Where does Alice think the apple is?",
            "choices": "A. red_box, B. white_box, C. black_box",
            "answer": "black_box" if observed else "white_box",
            "intervention_type": intervention,
            "counterfactual_anchor_container": "white_box",
            "counterfactual_final_container": "black_box",
            "qprop=params": [["Alice"], "apple", f"container_location-{observed}"],
            "rooms": ["room", "hall"],
            "story": common + suffix,
            "process_target": target,
            "process_response": response,
        }

    def test_hi_tom_pair_has_matched_position_and_neutral_tail(self) -> None:
        counter = builder.TokenCounter(None)
        rows = builder.transform_pair(
            [self.hi_record("observed"), self.hi_record("hidden")], 7, counter
        )
        self.assertEqual(
            rows[0]["intervention_event_index"], rows[1]["intervention_event_index"]
        )
        self.assertEqual(rows[0]["story_event_count"], rows[1]["story_event_count"])
        self.assertIn(rows[0]["events_after_intervention"], range(2, 7))
        self.assertEqual(
            rows[0]["story"].strip().splitlines()[-1],
            rows[1]["story"].strip().splitlines()[-1],
        )
        self.assertNotIn("Every listed observer", rows[0]["story"])
        observed = next(row for row in rows if row["intervention_type"] == "observed")
        hidden = next(row for row in rows if row["intervention_type"] == "hidden")
        observed_lines = observed["story"].splitlines()
        hidden_lines = hidden["story"].splitlines()
        final_index = observed["intervention_event_index"] - 1
        self.assertNotIn("Alice left", observed_lines[final_index - 1])
        self.assertIn("Alice left", hidden_lines[final_index - 1])

    def test_exploretom_pair_swaps_departure_around_same_final_move(self) -> None:
        counter = builder.TokenCounter(None)
        rows = builder.transform_pair(
            [self.explore_record("observed"), self.explore_record("hidden")],
            7,
            counter,
        )
        observed = next(row for row in rows if row["intervention_type"] == "observed")
        hidden = next(row for row in rows if row["intervention_type"] == "hidden")
        self.assertEqual(observed["intervention_event_index"], 8)
        self.assertEqual(observed["story_event_count"], hidden["story_event_count"])
        self.assertEqual(
            builder.split_sentences(observed["story"])[-1],
            builder.split_sentences(hidden["story"])[-1],
        )
        final_index = observed["intervention_event_index"] - 1
        self.assertNotIn(
            "Alice left", builder.split_sentences(observed["story"])[final_index - 1]
        )
        self.assertIn(
            "Alice left", builder.split_sentences(hidden["story"])[final_index - 1]
        )

    def test_compact_prompt_uses_conservative_token_bound(self) -> None:
        counter = builder.TokenCounter(None)
        rows = builder.transform_pair(
            [self.hi_record("observed"), self.hi_record("hidden")], 7, counter
        )
        self.assertTrue(all(row["process_prompt_token_count"] < 2048 for row in rows))
        self.assertTrue(all(row["process_sequence_token_count"] < 2048 for row in rows))


if __name__ == "__main__":
    unittest.main()
