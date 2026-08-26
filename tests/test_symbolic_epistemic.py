import unittest

from scripts.symbolic_epistemic import EpistemicState, MoveEvent


def move(
    event_id: str,
    source: str | None,
    target: str,
    visibility: str,
    observers: tuple[str, ...] = (),
) -> MoveEvent:
    return MoveEvent(
        event_id=event_id,
        object="key",
        from_location=source,
        to_location=target,
        visibility=visibility,  # type: ignore[arg-type]
        observers=observers,
    )


class EpistemicStateTest(unittest.TestCase):
    def test_joint_observation_is_common_knowledge_only_within_group(self) -> None:
        state = EpistemicState(("Alice", "Bob", "Carol"), max_order=3)
        state.apply(move("e1", None, "red", "joint", ("Alice", "Bob", "Carol")))
        state.apply(move("e2", "red", "blue", "joint", ("Alice", "Bob")))

        self.assertEqual(state.query("key"), "blue")
        for chain in (("Alice",), ("Bob",), ("Alice", "Bob"), ("Bob", "Alice")):
            self.assertEqual(state.query("key", chain), "blue")
        self.assertEqual(state.query("key", ("Carol",)), "red")
        self.assertEqual(state.query("key", ("Alice", "Carol")), "red")

    def test_private_and_hidden_moves_separate_world_and_nested_beliefs(self) -> None:
        state = EpistemicState(("Alice", "Bob"), max_order=2)
        state.apply(move("e1", None, "red", "joint", ("Alice", "Bob")))
        state.apply(move("e2", "red", "blue", "private", ("Bob",)))
        state.apply(move("e3", "blue", "yellow", "private", ("Alice",)))
        state.apply(move("e4", "yellow", "green", "hidden"))

        self.assertEqual(state.query("key"), "green")
        self.assertEqual(state.query("key", ("Alice",)), "yellow")
        self.assertEqual(state.query("key", ("Bob",)), "blue")
        self.assertEqual(state.query("key", ("Alice", "Bob")), "red")

    def test_belief_trace_lists_distinct_suffix_states(self) -> None:
        state = EpistemicState(("Alice", "Bob", "Carol"), max_order=3)
        state.apply(move("e1", None, "red", "joint", ("Alice", "Bob", "Carol")))
        state.apply(move("e2", "red", "blue", "joint", ("Bob", "Carol")))
        state.apply(move("e3", "blue", "green", "private", ("Carol",)))

        self.assertEqual(
            state.belief_trace("key", ("Alice", "Bob", "Carol")),
            [
                {"belief_chain": ["Carol"], "location": "green"},
                {"belief_chain": ["Bob", "Carol"], "location": "blue"},
                {
                    "belief_chain": ["Alice", "Bob", "Carol"],
                    "location": "red",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
