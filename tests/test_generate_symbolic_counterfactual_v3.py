import unittest

from scripts.generate_symbolic_counterfactual_v3 import (
    TokenCounter,
    build_pair,
    question_for,
)


class SymbolicCounterfactualGeneratorTest(unittest.TestCase):
    def test_question_grammar_for_multiple_orders(self) -> None:
        self.assertEqual(
            question_for(["Alice"], "apple"),
            "Where does Alice think the apple is?",
        )
        self.assertEqual(
            question_for(["Alice", "Bob", "Carol"], "apple"),
            "Where does Alice think Bob thinks Carol thinks the apple is?",
        )

    def test_pairs_flip_answer_and_change_only_critical_observer_membership(self) -> None:
        counter = TokenCounter(None)
        for order in range(1, 5):
            observed, hidden = build_pair(
                pair_number=order,
                split="test-unit",
                order=order,
                seed=2026,
                counter=counter,
                max_tokens=2048,
            )
            self.assertNotEqual(observed["answer"], hidden["answer"])
            self.assertEqual(
                observed["critical_event_index"], hidden["critical_event_index"]
            )
            self.assertEqual(observed["story_event_count"], hidden["story_event_count"])
            self.assertGreaterEqual(observed["events_after_critical"], 2)
            self.assertLessEqual(observed["events_after_critical"], 6)

            changed_events = []
            for observed_event, hidden_event in zip(
                observed["latent_events"], hidden["latent_events"]
            ):
                if observed_event != hidden_event:
                    changed_events.append(observed_event["event_id"])
                    observed_without_audience = dict(observed_event, observers=())
                    hidden_without_audience = dict(hidden_event, observers=())
                    self.assertEqual(observed_without_audience, hidden_without_audience)
            self.assertEqual(changed_events, [observed["critical_event_id"]])

            chain = observed["belief_chain"]
            trace = observed["process_target"]["belief_trace"]
            self.assertEqual(
                [step["belief_chain"] for step in trace],
                [chain[-depth:] for depth in range(1, order + 1)],
            )
            self.assertEqual(trace[-1]["location"], observed["answer"])
            self.assertLessEqual(observed["process_sequence_token_count"], 2048)

            if order >= 2:
                for row in (observed, hidden):
                    self.assertTrue(
                        all(
                            prediction != row["answer"]
                            for prediction in row["shortcut_predictions"].values()
                        )
                    )
                    separated = {
                        row["latent_state"]["world_state"],
                        row["latent_state"]["outer_belief"],
                        *(step["location"] for step in row["latent_state"]["belief_trace"]),
                    }
                    self.assertEqual(len(separated), order + 2)


if __name__ == "__main__":
    unittest.main()
