#!/usr/bin/env python3
"""Bounded symbolic epistemic state for deterministic ToM generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Iterable, Literal


Visibility = Literal["joint", "private", "hidden"]


@dataclass(frozen=True)
class MoveEvent:
    """A physical move plus the epistemic conditions under which it occurred."""

    event_id: str
    object: str
    from_location: str | None
    to_location: str
    visibility: Visibility
    observers: tuple[str, ...]
    template_id: int = 0
    role: str = "ordinary"

    def __post_init__(self) -> None:
        if self.visibility == "hidden" and self.observers:
            raise ValueError("Hidden events cannot have observers")
        if self.visibility in {"joint", "private"} and not self.observers:
            raise ValueError(f"{self.visibility} events need at least one observer")
        if len(set(self.observers)) != len(self.observers):
            raise ValueError("Observer names must be unique")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def belief_chains(agents: Iterable[str], max_order: int) -> list[tuple[str, ...]]:
    """Enumerate non-repeating belief chains through the configured depth."""
    agents = tuple(agents)
    if max_order < 1 or max_order > len(agents):
        raise ValueError("max_order must be between 1 and the number of agents")
    return [
        chain
        for order in range(1, max_order + 1)
        for chain in permutations(agents, order)
    ]


class EpistemicState:
    """Track world state and bounded nested beliefs for object locations.

    Joint observers see both the move and one another observing it, so the move
    becomes common knowledge within that group up to ``max_order``. Private
    observers receive isolated feeds and update only their own first-order
    beliefs. Hidden events update reality without updating any belief chain.
    """

    def __init__(self, agents: Iterable[str], max_order: int):
        self.agents = tuple(agents)
        if len(set(self.agents)) != len(self.agents):
            raise ValueError("Agent names must be unique")
        self.max_order = max_order
        self.chains = belief_chains(self.agents, max_order)
        self.world: dict[str, str] = {}
        self.beliefs: dict[tuple[str, ...], dict[str, str]] = {
            chain: {} for chain in self.chains
        }
        self.events: list[MoveEvent] = []
        self.updated_chains: dict[str, tuple[tuple[str, ...], ...]] = {}

    def apply(self, event: MoveEvent) -> tuple[tuple[str, ...], ...]:
        if event.object in self.world:
            if event.from_location != self.world[event.object]:
                raise ValueError(
                    f"Event {event.event_id} starts at {event.from_location!r}; "
                    f"world is {self.world[event.object]!r}"
                )
        elif event.from_location is not None:
            raise ValueError(f"Initial event {event.event_id} needs from_location=None")
        unknown = set(event.observers) - set(self.agents)
        if unknown:
            raise ValueError(f"Unknown observers in {event.event_id}: {sorted(unknown)}")

        self.world[event.object] = event.to_location
        updated: list[tuple[str, ...]] = []
        observer_set = set(event.observers)
        if event.visibility == "joint":
            for chain in self.chains:
                if set(chain) <= observer_set:
                    self.beliefs[chain][event.object] = event.to_location
                    updated.append(chain)
        elif event.visibility == "private":
            for observer in event.observers:
                chain = (observer,)
                self.beliefs[chain][event.object] = event.to_location
                updated.append(chain)

        self.events.append(event)
        result = tuple(updated)
        self.updated_chains[event.event_id] = result
        return result

    def apply_all(self, events: Iterable[MoveEvent]) -> "EpistemicState":
        for event in events:
            self.apply(event)
        return self

    def query(self, object_name: str, chain: Iterable[str] = ()) -> str:
        chain = tuple(chain)
        if not chain:
            try:
                return self.world[object_name]
            except KeyError as exc:
                raise ValueError(f"Unknown world state for {object_name!r}") from exc
        if chain not in self.beliefs:
            raise ValueError(f"Unsupported belief chain: {chain}")
        try:
            return self.beliefs[chain][object_name]
        except KeyError as exc:
            raise ValueError(
                f"Belief chain {chain} has no state for {object_name!r}"
            ) from exc

    def belief_trace(
        self, object_name: str, chain: Iterable[str]
    ) -> list[dict[str, object]]:
        """Return suffix-chain states from the innermost belief to the query."""
        chain = tuple(chain)
        return [
            {
                "belief_chain": list(chain[-depth:]),
                "location": self.query(object_name, chain[-depth:]),
            }
            for depth in range(1, len(chain) + 1)
        ]

    def last_update_event(self, chain: Iterable[str]) -> MoveEvent | None:
        chain = tuple(chain)
        for event in reversed(self.events):
            if chain in self.updated_chains[event.event_id]:
                return event
        return None
