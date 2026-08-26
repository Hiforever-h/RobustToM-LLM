#!/usr/bin/env python3
"""Generate state-separated counterfactual ToM data from symbolic events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

try:
    from symbolic_epistemic import EpistemicState, MoveEvent
except ModuleNotFoundError:  # Allow importing this file as scripts.<module> in tests.
    from scripts.symbolic_epistemic import EpistemicState, MoveEvent


GENERATOR_VERSION = "3.0"
PROCESS_TARGET_VERSION = "2.0"
NAMES = (
    "Alice", "Benjamin", "Charlotte", "Daniel", "Eleanor", "Felix",
    "Grace", "Henry", "Isabella", "Jacob", "Kayla", "Liam", "Maya",
    "Noah", "Olivia", "Peter", "Quinn", "Rachel", "Samuel", "Taylor",
    "Uma", "Victor", "Willow", "Xavier", "Yasmin", "Zachary", "Amelia",
    "Brody", "Claire", "Dylan", "Eva", "George", "Hannah", "Ian",
    "Julia", "Kevin", "Lucy", "Matthew", "Nora", "Owen", "Phoebe",
    "Reid", "Sophie", "Thomas", "Violet", "William", "Addison", "Blake",
)
OBJECTS = (
    "apple", "camera", "contract", "flashlight", "keyring", "laptop",
    "notebook", "passport", "paintbrush", "tablet", "violin", "wallet",
    "watch", "puzzle", "folder", "medal", "map", "mug", "ticket", "badge",
)
DISTRACTOR_OBJECTS = (
    "newspaper", "umbrella", "coffee mug", "calendar", "headphones",
    "scarf", "notepad", "backpack", "photograph", "toolbox",
)
CONTAINERS = (
    "amber_cabinet", "blue_canvas_bag", "brass_locker", "cedar_chest",
    "ceramic_jar", "cloth_hamper", "desk_drawer", "filing_cabinet",
    "glass_case", "green_toolbox", "grey_suitcase", "leather_satchel",
    "metal_trunk", "oak_cupboard", "orange_backpack",
)
LETTERS = "ABCDEFGHIJKLMNO"
JOINT_TEMPLATES = (
    (
        "{observers} watched together as an automated system moved the {object} "
        "from the {source} to the {target}; every named observer could see who "
        "else was watching."
    ),
    (
        "In a shared viewing room, {observers} jointly observed the {object} move "
        "from the {source} to the {target}, with the full observer list visible "
        "to everyone there."
    ),
    (
        "A live display showed the {object} moving from the {source} to the "
        "{target} to {observers} together; each named viewer knew that all the "
        "other named viewers saw it."
    ),
)
PRIVATE_TEMPLATES = (
    (
        "{observers} alone received a private live feed showing the {object} move "
        "from the {source} to the {target}; nobody else was told who received it."
    ),
    (
        "On an isolated monitor, only {observers} saw the {object} move from the "
        "{source} to the {target}, and the feed was not announced to anyone else."
    ),
    (
        "A private camera showed {observers}, and no announced audience, the "
        "{object} moving from the {source} to the {target}."
    ),
)
HIDDEN_TEMPLATES = (
    (
        "With every viewing device switched off, an automated system moved the "
        "{object} from the {source} to the {target}; nobody received information "
        "about this move."
    ),
    (
        "While the viewing room was empty, the {object} was moved from the "
        "{source} to the {target}, unseen and unreported."
    ),
    (
        "An automated system privately moved the {object} from the {source} to "
        "the {target}, without showing or reporting the move to any person."
    ),
)
INITIAL_TEMPLATES = (
    (
        "{observers} jointly watched the {object} being placed in the {target}; "
        "every named observer could see all the others watching."
    ),
    (
        "Together, {observers} observed the initial placement of the {object} in "
        "the {target}, and everyone present knew the full audience."
    ),
)
NEUTRAL_TEMPLATES = (
    "{first} checked the wall clock.",
    "{second} placed a {distractor} beside the doorway.",
    "{first} and {second} discussed the weather.",
    "A bell rang in the hallway.",
    "{second} wrote a note about the room temperature.",
)


def stable_int(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_rng(*parts: Any) -> random.Random:
    return random.Random(stable_int(*parts))


def serial_join(items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        raise ValueError("Cannot render an empty observer list")
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


class TokenCounter:
    def __init__(self, tokenizer_name: str | None):
        self.name = tokenizer_name or "conservative-character-estimate"
        self.tokenizer = None
        if tokenizer_name:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                local_files_only=Path(tokenizer_name).exists(),
                use_fast=True,
            )

    def raw_count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return math.ceil(len(text.encode("utf-8")) / 3)

    def sequence_count(self, prompt: str, response: str) -> int:
        if self.tokenizer is None:
            return self.raw_count(prompt) + self.raw_count(response) + 32
        try:
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_count = len(prompt_ids)
        except (ImportError, TypeError, ValueError):
            wrapper = (
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
                "You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
            prompt_count = self.raw_count(wrapper)
        return prompt_count + self.raw_count(response) + 1


def choose_subset_excluding_full_chain(
    rng: random.Random,
    agents: list[str],
    chain: list[str],
    require_joint: bool = False,
) -> tuple[str, ...]:
    missing = rng.choice(chain)
    candidates = [agent for agent in agents if agent != missing]
    minimum = 1
    maximum = len(candidates)
    size = rng.randint(minimum, maximum)
    selected = rng.sample(candidates, size)
    if require_joint and len(selected) == 1 and len(candidates) > 1:
        selected.append(next(agent for agent in candidates if agent not in selected))
    return tuple(agent for agent in agents if agent in selected)


def make_symbolic_events(
    agents: list[str],
    chain: list[str],
    object_name: str,
    locations: list[str],
    pair_id: str,
    seed: int,
) -> tuple[list[MoveEvent], list[MoveEvent], str]:
    rng = stable_rng(seed, pair_id, "events")
    location_iter = iter(locations)
    current: str | None = None
    events: list[MoveEvent] = []

    def add(
        role: str,
        visibility: str,
        observers: Iterable[str] = (),
    ) -> MoveEvent:
        nonlocal current
        target = next(location_iter)
        event = MoveEvent(
            event_id=f"e{len(events) + 1}",
            object=object_name,
            from_location=current,
            to_location=target,
            visibility=visibility,  # type: ignore[arg-type]
            observers=tuple(observers),
            template_id=rng.randrange(3),
            role=role,
        )
        events.append(event)
        current = target
        return event

    add("initial", "joint", agents)
    for _ in range(rng.randint(0, 2)):
        mode = rng.choice(("joint", "private", "hidden"))
        if mode == "joint":
            observers = choose_subset_excluding_full_chain(rng, agents, chain)
        elif mode == "private":
            observers = (rng.choice(agents),)
        else:
            observers = ()
        add("prelude", mode, observers)

    controls = [agent for agent in agents if agent not in chain]
    baseline_observers = list(chain)
    baseline_observers.extend(
        rng.sample(controls, rng.randint(0, len(controls))) if controls else []
    )
    add("baseline", "joint", [a for a in agents if a in baseline_observers])

    replacement = rng.choice(controls)
    optional_controls = [agent for agent in controls if agent != replacement]
    critical_observers = list(chain)
    critical_observers.extend(
        rng.sample(optional_controls, rng.randint(0, len(optional_controls)))
        if optional_controls
        else []
    )
    critical = add(
        "critical",
        "joint",
        [agent for agent in agents if agent in critical_observers],
    )

    post_events: list[tuple[str, str, tuple[str, ...]]] = []
    if len(chain) >= 2:
        post_events.append(("outer_private", "private", (chain[0],)))
        for suffix_start in range(1, len(chain)):
            suffix = chain[suffix_start:]
            if len(suffix) == 1:
                post_events.append(("inner_private", "private", (suffix[0],)))
            else:
                extra_controls = rng.sample(
                    controls, rng.randint(0, len(controls))
                ) if controls else []
                observers = [
                    agent for agent in agents if agent in set(suffix + extra_controls)
                ]
                post_events.append(("suffix_joint", "joint", tuple(observers)))

        outer_event = post_events.pop(0)
        insertion = rng.randint(0, len(post_events))
        post_events.insert(insertion, outer_event)

    # Every queried suffix contains the innermost thinker. Excluding that person
    # prevents later control events from collapsing distinct trace levels.
    control_joint = choose_subset_excluding_full_chain(
        rng, agents, [chain[-1]], require_joint=True
    )
    post_events.append(("post_control", "joint", control_joint))
    core_after = len(post_events) + 1
    desired_after = rng.randint(core_after, 6)
    for extra_index in range(desired_after - core_after):
        mode = rng.choice(("joint", "private", "hidden"))
        if mode == "joint":
            observers = choose_subset_excluding_full_chain(
                rng, agents, [chain[-1]], require_joint=True
            )
        elif mode == "private":
            candidates = [agent for agent in agents if agent != chain[0]]
            observers = (rng.choice(candidates),)
        else:
            observers = ()
        insertion = rng.randint(0, len(post_events))
        post_events.insert(insertion, (f"post_extra_{extra_index}", mode, observers))

    for role, visibility, observers in post_events:
        add(role, visibility, observers)
    add("final_world", "hidden")

    hidden_observers = [
        replacement if agent == chain[0] else agent
        for agent in critical.observers
    ]
    hidden_critical = replace(
        critical,
        observers=tuple(agent for agent in agents if agent in hidden_observers),
    )
    hidden_events = [
        hidden_critical if event.event_id == critical.event_id else event
        for event in events
    ]
    return events, hidden_events, critical.event_id


def render_move(event: MoveEvent) -> str:
    if event.from_location is None:
        observers = serial_join(event.observers)
        template = INITIAL_TEMPLATES[event.template_id % len(INITIAL_TEMPLATES)]
        return template.format(
            observers=observers,
            object=event.object,
            target=event.to_location,
        )
    values = {
        "object": event.object,
        "source": event.from_location,
        "target": event.to_location,
    }
    if event.visibility == "hidden":
        return HIDDEN_TEMPLATES[event.template_id % len(HIDDEN_TEMPLATES)].format(
            **values
        )
    values["observers"] = serial_join(event.observers)
    if event.visibility == "joint":
        return JOINT_TEMPLATES[event.template_id % len(JOINT_TEMPLATES)].format(
            **values
        )
    if event.visibility == "private":
        return PRIVATE_TEMPLATES[event.template_id % len(PRIVATE_TEMPLATES)].format(
            **values
        )
    raise ValueError(f"Unsupported event visibility: {event.visibility!r}")


def render_story(
    events: list[MoveEvent],
    agents: list[str],
    distractor: str,
    critical_event_id: str,
    pair_id: str,
    seed: int,
) -> tuple[str, dict[str, int]]:
    rng = stable_rng(seed, pair_id, "render")
    critical_target_index = next(
        index for index, event in enumerate(events) if event.event_id == critical_event_id
    )
    target_after = len(events) - critical_target_index - 1
    after_budget = max(0, 6 - target_after)
    neutral_count = rng.randint(2, 5)
    neutral_after = rng.randint(0, min(2, after_budget, neutral_count))
    neutral_before = neutral_count - neutral_after
    first, second = rng.sample(agents, 2)
    neutral_template_ids = rng.sample(range(len(NEUTRAL_TEMPLATES)), neutral_count)
    neutral_lines = [
        NEUTRAL_TEMPLATES[template_id].format(
            first=first,
            second=second,
            distractor=distractor,
        )
        for template_id in neutral_template_ids
    ]
    before_slots = [rng.randrange(0, critical_target_index + 1) for _ in range(neutral_before)]
    by_slot: defaultdict[int, list[str]] = defaultdict(list)
    for slot, line in zip(before_slots, neutral_lines[:neutral_before]):
        by_slot[slot].append(line)

    lines: list[str] = []
    critical_story_index = 0
    event_story_indices: dict[str, int] = {}
    for index, event in enumerate(events):
        lines.extend(by_slot[index])
        lines.append(render_move(event))
        event_story_indices[event.event_id] = len(lines)
        if event.event_id == critical_event_id:
            critical_story_index = len(lines)
    lines.extend(neutral_lines[neutral_before:])
    numbered = [f"{index} {line}" for index, line in enumerate(lines, start=1)]
    return "\n".join(numbered) + "\n", {
        "critical_event_index": critical_story_index,
        "story_event_count": len(lines),
        "events_after_critical": len(lines) - critical_story_index,
        "target_moves_after_critical": target_after,
        "neutral_event_count": neutral_count,
        **{f"story_index_{key}": value for key, value in event_story_indices.items()},
    }


def question_for(chain: list[str], object_name: str) -> str:
    if not chain:
        raise ValueError("A belief question needs at least one thinker")
    inner = " ".join(f"{agent} thinks" for agent in chain[1:])
    nested = f"{inner} " if inner else ""
    return f"Where does {chain[0]} think {nested}the {object_name} is?"


def choices_for(containers: list[str]) -> str:
    return ", ".join(
        f"{letter}. {container}" for letter, container in zip(LETTERS, containers)
    )


def process_target(
    state: EpistemicState,
    chain: list[str],
    object_name: str,
) -> dict[str, Any]:
    answer = state.query(object_name, chain)
    return {
        "tom_order": len(chain),
        "belief_chain": chain,
        "object": object_name,
        "reasoning_mode": "nested_belief",
        "belief_trace": state.belief_trace(object_name, chain),
        "answer": answer,
    }


def process_prompt(story: str, question: str, choices: str) -> str:
    return (
        "Return exactly one JSON object and no markdown or extra text. Copy names "
        "exactly. A group that watched together saw both the move and one another "
        "watching, making that move common knowledge within the named group. A "
        "private feed updates only its named recipient's own belief. An unseen move "
        "updates reality but no person's belief. belief_chain runs from the outermost "
        "thinker to the innermost thinker. belief_trace must list suffix chains from "
        "the innermost belief through the full queried chain.\n\n"
        "Schema:\n"
        '{"tom_order":2,"belief_chain":["outer","inner"],"object":"...",'
        '"reasoning_mode":"nested_belief","belief_trace":['
        '{"belief_chain":["inner"],"location":"..."},'
        '{"belief_chain":["outer","inner"],"location":"..."}],'
        '"answer":"..."}\n\n'
        f"Story:\n{story.rstrip()}\n\nQuestion: {question}\nChoices: {choices}\n"
    )


def shallow_predictions(
    state: EpistemicState,
    events: list[MoveEvent],
    chain: list[str],
    object_name: str,
) -> dict[str, str]:
    joint_events = [event for event in events if event.visibility == "joint"]
    return {
        "world_state": state.query(object_name),
        "initial_location": events[0].to_location,
        "last_move": events[-1].to_location,
        "previous_move": events[-2].to_location,
        "outer_last_seen": state.query(object_name, (chain[0],)),
        "inner_actual_belief": state.query(object_name, (chain[-1],)),
        "last_joint_move": joint_events[-1].to_location,
    }


def suffix_prediction(
    events: list[MoveEvent], chain: list[str], move_limit: int
) -> str | None:
    for event in reversed(events[-move_limit:]):
        if event.visibility == "joint" and set(chain) <= set(event.observers):
            return event.to_location
        if (
            len(chain) == 1
            and event.visibility == "private"
            and chain[0] in event.observers
        ):
            return event.to_location
    return None


def build_pair(
    pair_number: int,
    split: str,
    order: int,
    seed: int,
    counter: TokenCounter,
    max_tokens: int,
) -> list[dict[str, Any]]:
    pair_id = f"symbolic-v3:{split}:{pair_number:05d}"
    rng = stable_rng(seed, pair_id, "scenario")
    agents = rng.sample(list(NAMES), 5)
    chain = rng.sample(agents, order)
    object_name = rng.choice(OBJECTS)
    distractor = rng.choice([item for item in DISTRACTOR_OBJECTS if item != object_name])
    locations = list(CONTAINERS)
    rng.shuffle(locations)
    observed_events, hidden_events, critical_event_id = make_symbolic_events(
        agents, chain, object_name, locations, pair_id, seed
    )

    choices = list(CONTAINERS)
    rng.shuffle(choices)
    choices_text = choices_for(choices)
    question = question_for(chain, object_name)
    records: list[dict[str, Any]] = []
    for intervention, events in (
        ("observed", observed_events),
        ("hidden", hidden_events),
    ):
        state = EpistemicState(agents, max_order=4).apply_all(events)
        target = process_target(state, chain, object_name)
        answer = target["answer"]
        proxies = shallow_predictions(state, events, chain, object_name)
        forbidden = {
            proxies[key]
            for key in (
                "world_state",
                "initial_location",
                "last_move",
                "previous_move",
                "last_joint_move",
            )
        }
        if order >= 2:
            forbidden.update(
                {proxies["outer_last_seen"], proxies["inner_actual_belief"]}
            )
        if answer in forbidden:
            raise ValueError(f"Hard-sample separation failed: {pair_id}-{intervention}")

        story, story_audit = render_story(
            events, agents, distractor, critical_event_id, pair_id, seed
        )
        if not 2 <= story_audit["events_after_critical"] <= 6:
            raise ValueError(
                f"Continuation length is outside 2-6: {pair_id}-{intervention}"
            )

        trace_locations = [step["location"] for step in target["belief_trace"]]
        separated_locations = {
            state.query(object_name),
            state.query(object_name, (chain[0],)),
            *trace_locations,
        }
        if order >= 2 and len(separated_locations) != order + 2:
            raise ValueError(
                f"Epistemic states are not fully separated: {pair_id}-{intervention}"
            )

        prompt = process_prompt(story, question, choices_text)
        response = json.dumps(target, ensure_ascii=True, separators=(",", ":"))
        prompt_tokens = counter.raw_count(prompt)
        sequence_tokens = counter.sequence_count(prompt, response)
        if prompt_tokens > max_tokens or sequence_tokens > max_tokens:
            raise ValueError(f"Token limit exceeded: {pair_id}-{intervention}")

        critical = next(event for event in events if event.event_id == critical_event_id)
        answer_event = state.last_update_event(chain)
        record = {
            "dataset": "Symbolic-Counterfactual-ToM-v3",
            "counterfactual_generation_version": GENERATOR_VERSION,
            "process_target_version": PROCESS_TARGET_VERSION,
            "source_dataset": "symbolic-tom-v3",
            "source_split": split,
            "split": split,
            "sample_id": f"{pair_id}-{intervention}",
            "global_sample_id": f"symbolic-tom-v3:{pair_id}-{intervention}",
            "pair_id": pair_id,
            "global_pair_id": f"symbolic-tom-v3:{pair_id}",
            "source_group_id": pair_id,
            "question_order": order,
            "belief_chain": chain,
            "object": object_name,
            "intervention_type": intervention,
            "critical_event_id": critical_event_id,
            "critical_event_observers": list(critical.observers),
            "answer_event_id": answer_event.event_id if answer_event else None,
            "story": story,
            "question": question,
            "choices": choices_text,
            "answer": answer,
            "answer_letter": LETTERS[choices.index(answer)],
            "prompt": (
                "Read the story and answer the question.\n"
                f"Story:\n{story}\nQuestion: {question}\nChoices: {choices_text}\n"
            ),
            "process_target": target,
            "process_response": response,
            "process_prompt": prompt,
            "process_prompt_token_count": prompt_tokens,
            "process_sequence_token_count": sequence_tokens,
            "latent_events": [event.to_dict() for event in events],
            "latent_state": {
                "world_state": state.query(object_name),
                "belief_trace": state.belief_trace(object_name, chain),
                "outer_belief": state.query(object_name, (chain[0],)),
                "inner_belief": state.query(object_name, (chain[-1],)),
            },
            "shortcut_predictions": proxies,
            "shortcut_name": "inner_actual_belief",
            "shortcut_prediction": proxies["inner_actual_belief"],
            "shortcut_conflict": proxies["inner_actual_belief"] != answer,
            "last_mentioned_container": events[-1].to_location,
            "last_mention_conflict": events[-1].to_location != answer,
            "suffix_3_prediction": suffix_prediction(events, chain, 3),
            "suffix_5_prediction": suffix_prediction(events, chain, 5),
            "target_move_count": len(events),
            "critical_target_move_index": int(critical_event_id[1:]),
            "target_moves_after_critical": len(events) - int(critical_event_id[1:]),
            "continuation_event_count": story_audit["events_after_critical"],
            **story_audit,
        }
        record["critical_relative_position"] = round(
            record["critical_event_index"] / record["story_event_count"], 6
        )
        records.append(record)

    if records[0]["answer"] == records[1]["answer"]:
        raise ValueError(f"Counterfactual pair did not flip the answer: {pair_id}")
    for field in (
        "critical_event_index",
        "story_event_count",
        "events_after_critical",
        "target_move_count",
    ):
        if records[0][field] != records[1][field]:
            raise ValueError(f"Counterfactual pair differs on {field}: {pair_id}")
    return records


def order_schedule(split: str, pairs: int) -> list[int]:
    if split == "test":
        return [4] * pairs
    if split == "val":
        counts = {1: pairs // 4, 2: pairs // 2}
        counts[3] = pairs - counts[1] - counts[2]
    else:
        counts = {1: round(pairs * 0.25), 2: round(pairs * 0.47)}
        counts[3] = pairs - counts[1] - counts[2]
    return [order for order in sorted(counts) for _ in range(counts[order])]


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]


def audit_dataset(records_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for records in records_by_split.values() for row in records]
    pairs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pairs[row["global_pair_id"]].append(row)
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {row["intervention_type"] for row in pair} != {
            "observed",
            "hidden",
        }:
            raise ValueError(f"Invalid pair: {pair_id}")
        if len({row["split"] for row in pair}) != 1:
            raise ValueError(f"Pair crosses splits: {pair_id}")
        if len({row["answer"] for row in pair}) != 2:
            raise ValueError(f"Pair answer did not flip: {pair_id}")

    heuristic_names = sorted(rows[0]["shortcut_predictions"])
    heuristic_accuracy = {
        name: round(
            sum(row["shortcut_predictions"][name] == row["answer"] for row in rows)
            / len(rows),
            6,
        )
        for name in heuristic_names
    }
    belief_rows = [row for row in rows if int(row["question_order"]) >= 2]
    hard_rate = sum(
        all(prediction != row["answer"] for prediction in row["shortcut_predictions"].values())
        for row in belief_rows
    ) / len(belief_rows)
    state_separation_rate = sum(
        len(
            {
                row["latent_state"]["world_state"],
                row["latent_state"]["outer_belief"],
                *(step["location"] for step in row["latent_state"]["belief_trace"]),
            }
        )
        == int(row["question_order"]) + 2
        for row in belief_rows
    ) / len(belief_rows)
    positions = [row["critical_relative_position"] for row in rows]
    prompt_lengths = [row["process_prompt_token_count"] for row in rows]
    sequence_lengths = [row["process_sequence_token_count"] for row in rows]
    return {
        "records": len(rows),
        "pairs": len(pairs),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "pair_split_counts": {
            split: len({row["global_pair_id"] for row in split_rows})
            for split, split_rows in records_by_split.items()
        },
        "order_counts": dict(Counter(str(row["question_order"]) for row in rows)),
        "intervention_counts": dict(Counter(row["intervention_type"] for row in rows)),
        "answer_letter_counts": dict(Counter(row["answer_letter"] for row in rows)),
        "heuristic_accuracy": heuristic_accuracy,
        "order_2plus_all_shallow_heuristics_disagree_rate": round(hard_rate, 6),
        "order_2plus_state_separation_rate": round(state_separation_rate, 6),
        "suffix_3_accuracy": round(
            sum(row["suffix_3_prediction"] == row["answer"] for row in rows) / len(rows),
            6,
        ),
        "suffix_5_accuracy": round(
            sum(row["suffix_5_prediction"] == row["answer"] for row in rows) / len(rows),
            6,
        ),
        "pair_answer_flip_rate": 1.0,
        "pair_position_match_rate": 1.0,
        "critical_relative_position": {
            "min": min(positions),
            "max": max(positions),
            "mean": round(sum(positions) / len(positions), 6),
        },
        "events_after_critical_counts": dict(
            Counter(str(row["events_after_critical"]) for row in rows)
        ),
        "process_prompt_tokens": {
            "max": max(prompt_lengths),
            "p95": percentile(prompt_lengths, 0.95),
            "mean": round(sum(prompt_lengths) / len(prompt_lengths), 3),
        },
        "process_sequence_tokens": {
            "max": max(sequence_lengths),
            "p95": percentile(sequence_lengths, 0.95),
            "mean": round(sum(sequence_lengths) / len(sequence_lengths), 3),
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def generate_dataset(
    output_dir: Path,
    seed: int,
    train_pairs: int,
    val_pairs: int,
    test_pairs: int,
    max_tokens: int,
    tokenizer_name: str | None,
) -> dict[str, Any]:
    counter = TokenCounter(tokenizer_name)
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, pair_count in (
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ):
        schedule = order_schedule(split, pair_count)
        stable_rng(seed, split, "schedule").shuffle(schedule)
        records: list[dict[str, Any]] = []
        for pair_number, order in enumerate(schedule):
            records.extend(
                build_pair(pair_number, split, order, seed, counter, max_tokens)
            )
        stable_rng(seed, split, "records").shuffle(records)
        records_by_split[split] = records

    audit = audit_dataset(records_by_split)
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in records_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    manifest = {
        "name": "Symbolic State-Separated Counterfactual ToM v3",
        "generator_version": GENERATOR_VERSION,
        "process_target_version": PROCESS_TARGET_VERSION,
        "seed": seed,
        "tokenizer": counter.name,
        "max_process_and_sequence_tokens": max_tokens,
        "epistemic_semantics": {
            "joint": "common knowledge within the named observer group",
            "private": "first-order update for the isolated recipient only",
            "hidden": "world update with no belief update",
            "maximum_tom_order": 4,
        },
        **audit,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        f"""# Symbolic State-Separated Counterfactual ToM v3

This dataset is generated from symbolic movement and observation events. Labels
come from a bounded nested-belief oracle, never from an LLM. Each pair changes
one observer identity at a critical move while preserving event count and move
position. Every answer disagrees with world, initial, last, previous, and last
joint-move heuristics; order-2+ answers also disagree with outer and inner
first-order beliefs.

The source uses process-target schema `{PROCESS_TARGET_VERSION}` with a
`belief_trace` from the innermost suffix chain to the complete queried chain.
See `manifest.json` for shortcut, position, and token audits.

Rebuild from the repository root:

```bash
python3 scripts/generate_symbolic_counterfactual_v3.py \\
  --output-dir {output_dir} \\
  --tokenizer /path/to/Qwen2.5-tokenizer
```
""",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/counterfactual_process_reward_v3"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-pairs", type=int, default=1600)
    parser.add_argument("--val-pairs", type=int, default=200)
    parser.add_argument("--test-pairs", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--tokenizer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_dataset(
        args.output_dir,
        args.seed,
        args.train_pairs,
        args.val_pairs,
        args.test_pairs,
        args.max_tokens,
        args.tokenizer,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
