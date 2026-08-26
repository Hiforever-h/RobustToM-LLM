"""Stdlib-only utilities shared by the RFT commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def target_from_record(record: dict[str, Any]) -> dict[str, Any]:
    target = record.get("process_target")
    if isinstance(target, str):
        target = json.loads(target)
    if not isinstance(target, dict):
        raise ValueError(f"Missing process_target for {record.get('global_sample_id', '<unknown>')}")
    return target


def prompt_from_record(record: dict[str, Any]) -> str:
    prompt = record.get("process_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing process_prompt for {record.get('global_sample_id', '<unknown>')}")
    return prompt


def sample_id(record: dict[str, Any]) -> str:
    value = record.get("global_sample_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Every record needs a non-empty global_sample_id")
    return value


def pair_id(record: dict[str, Any]) -> str:
    value = record.get("global_pair_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Every record needs a non-empty global_pair_id: {sample_id(record)}")
    return value


def record_key(record: dict[str, Any]) -> str:
    return sample_id(record)
