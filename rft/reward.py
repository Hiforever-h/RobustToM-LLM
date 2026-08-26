"""Deterministic process scorer used by sampling, training-data construction and eval."""

from __future__ import annotations

import json
import re
from typing import Any

WORLD_FIELDS = {
    "tom_order",
    "belief_chain",
    "object",
    "reasoning_mode",
    "world_state",
    "answer",
}
BELIEF_FIELDS = {
    "tom_order",
    "belief_chain",
    "object",
    "reasoning_mode",
    "final_move_observed",
    "nested_belief",
    "answer",
}
NESTED_BELIEF_FIELDS = {
    "tom_order",
    "belief_chain",
    "object",
    "reasoning_mode",
    "belief_trace",
    "answer",
}
TRACE_FIELDS = {"belief_chain", "location"}
FENCED_JSON_RE = re.compile(r"^```(?:json)?\s*(\{.*\})\s*```$", re.DOTALL | re.IGNORECASE)


def normalize(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def parse_prediction(output: str | dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    if isinstance(output, dict):
        return output, True
    if not isinstance(output, str):
        return None, False
    text = output.strip()
    strict_json = True
    match = FENCED_JSON_RE.fullmatch(text)
    if match:
        text = match.group(1)
        strict_json = False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, False
    return (parsed, strict_json) if isinstance(parsed, dict) else (None, False)


def valid_schema(prediction: dict[str, Any], fields: set[str], mode: str) -> bool:
    if set(prediction) != fields or prediction.get("reasoning_mode") != mode:
        return False
    if type(prediction.get("tom_order")) is not int:
        return False
    if not isinstance(prediction.get("belief_chain"), list) or not all(
        isinstance(item, str) for item in prediction["belief_chain"]
    ):
        return False
    if not all(
        isinstance(prediction.get(field), str)
        for field in ("object", "reasoning_mode", "answer")
    ):
        return False
    if mode == "world_state":
        return isinstance(prediction.get("world_state"), str)
    if mode == "belief":
        return (
            type(prediction.get("final_move_observed")) is bool
            and isinstance(prediction.get("nested_belief"), str)
        )
    return mode == "nested_belief" and valid_belief_trace(
        prediction.get("belief_trace")
    )


def valid_belief_trace(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(step, dict)
        and set(step) == TRACE_FIELDS
        and isinstance(step.get("belief_chain"), list)
        and all(isinstance(item, str) for item in step["belief_chain"])
        and isinstance(step.get("location"), str)
        for step in value
    )


def normalized_chain(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return [normalize(item) for item in value]


def normalized_trace(value: Any) -> list[tuple[tuple[str, ...], str]] | None:
    if not valid_belief_trace(value):
        return None
    return [
        (tuple(normalized_chain(step["belief_chain"]) or []), normalize(step["location"]))
        for step in value
    ]


def validate_target(target: dict[str, Any]) -> tuple[str, set[str]]:
    mode = target.get("reasoning_mode")
    if mode == "world_state":
        fields = WORLD_FIELDS
    elif mode == "belief":
        fields = BELIEF_FIELDS
    elif mode == "nested_belief":
        fields = NESTED_BELIEF_FIELDS
    else:
        raise ValueError(f"Unknown target reasoning_mode: {mode!r}")
    if not valid_schema(target, fields, mode):
        raise ValueError("Target does not match the process-reward schema")
    order = target["tom_order"]
    chain = target["belief_chain"]
    if mode == "world_state":
        if order != 0 or chain:
            raise ValueError("World-state targets require order 0 and an empty chain")
        if normalize(target["world_state"]) != normalize(target["answer"]):
            raise ValueError("World-state target and answer must agree")
    elif mode == "belief":
        if order < 1 or len(chain) != order:
            raise ValueError("Belief-chain length must equal the positive ToM order")
        if normalize(target["nested_belief"]) != normalize(target["answer"]):
            raise ValueError("Nested-belief target and answer must agree")
    else:
        if order < 1 or len(chain) != order:
            raise ValueError("Belief-chain length must equal the positive ToM order")
        trace = target["belief_trace"]
        if len(trace) != order:
            raise ValueError("Belief trace length must equal the ToM order")
        for depth, step in enumerate(trace, start=1):
            if normalized_chain(step["belief_chain"]) != normalized_chain(chain[-depth:]):
                raise ValueError("Belief trace must contain suffix chains in depth order")
        if normalize(trace[-1]["location"]) != normalize(target["answer"]):
            raise ValueError("Final belief-trace location and answer must agree")
    return mode, fields


def score_process_output(output: str | dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Return a scalar reward in [0, 1] plus auditable checks/components."""
    mode, fields = validate_target(target)
    prediction, strict_json = parse_prediction(output)
    if prediction is None:
        return {
            "reward": 0.0,
            "reasoning_mode": mode,
            "checks": {"parseable_json": False, "format": False},
            "components": {"format": 0.0},
        }

    schema_ok = valid_schema(prediction, fields, mode)
    order_ok = type(prediction.get("tom_order")) is int and prediction["tom_order"] == target["tom_order"]
    chain_ok = normalized_chain(prediction.get("belief_chain")) == normalized_chain(target["belief_chain"])
    object_ok = normalize(prediction.get("object", "")) == normalize(target["object"])
    format_ok = strict_json and schema_ok

    if mode == "world_state":
        state_ok = normalize(prediction.get("world_state", "")) == normalize(target["world_state"])
        answer_ok = normalize(prediction.get("answer", "")) == normalize(target["answer"])
        checks = {
            "parseable_json": True,
            "format": format_ok,
            "tom_order": order_ok,
            "belief_chain": chain_ok,
            "object": object_ok,
            "world_state": state_ok,
            "answer": answer_ok,
        }
        components = {
            "format": 0.05 if format_ok else 0.0,
            "tom_order": 0.10 if order_ok else 0.0,
            "object": 0.10 if object_ok else 0.0,
            "world_state": 0.50 if state_ok else 0.0,
            "answer": 0.25 if state_ok and answer_ok else 0.0,
        }
    elif mode == "belief":
        visibility_ok = (
            type(prediction.get("final_move_observed")) is bool
            and prediction["final_move_observed"] == target["final_move_observed"]
        )
        belief_ok = normalize(prediction.get("nested_belief", "")) == normalize(target["nested_belief"])
        answer_ok = normalize(prediction.get("answer", "")) == normalize(target["answer"])
        checks = {
            "parseable_json": True,
            "format": format_ok,
            "tom_order": order_ok,
            "belief_chain": chain_ok,
            "object": object_ok,
            "final_move_observed": visibility_ok,
            "nested_belief": belief_ok,
            "answer": answer_ok,
        }
        components = {
            "format": 0.05 if format_ok else 0.0,
            "tom_order": 0.05 if order_ok else 0.0,
            "belief_chain": 0.10 if chain_ok else 0.0,
            "object": 0.05 if object_ok else 0.0,
            "final_move_observed": 0.25 if visibility_ok else 0.0,
            "nested_belief": 0.35 if visibility_ok and belief_ok else 0.0,
            "answer": 0.15 if visibility_ok and belief_ok and answer_ok else 0.0,
        }
    else:
        target_trace = normalized_trace(target["belief_trace"]) or []
        predicted_trace = normalized_trace(prediction.get("belief_trace")) or []
        trace_steps = [
            index < len(predicted_trace) and predicted_trace[index] == expected
            for index, expected in enumerate(target_trace)
        ]
        trace_ok = len(predicted_trace) == len(target_trace) and all(trace_steps)
        trace_fraction = sum(trace_steps) / len(target_trace)
        answer_ok = normalize(prediction.get("answer", "")) == normalize(target["answer"])
        checks = {
            "parseable_json": True,
            "format": format_ok,
            "tom_order": order_ok,
            "belief_chain": chain_ok,
            "object": object_ok,
            "belief_trace": trace_ok,
            "belief_trace_steps": trace_steps,
            "answer": answer_ok,
        }
        components = {
            "format": 0.05 if format_ok else 0.0,
            "tom_order": 0.05 if order_ok else 0.0,
            "belief_chain": 0.10 if chain_ok else 0.0,
            "object": 0.05 if object_ok else 0.0,
            "belief_trace": 0.55 * trace_fraction,
            "answer": 0.20 if trace_ok and answer_ok else 0.0,
        }

    return {
        "reward": round(sum(components.values()), 10),
        "reasoning_mode": mode,
        "checks": checks,
        "components": components,
    }


def process_reward(output: str | dict[str, Any], target: dict[str, Any]) -> float:
    return float(score_process_output(output, target)["reward"])
