"""JSONL session contract validation utilities."""

from __future__ import annotations

import json
from typing import Any, Iterator, Literal, Optional

ScenarioId = Literal["trivial_correlation", "silent_resonance", "conflicting_objectives"]

REQUIRED_META_KEYS = frozenset(
    {
        "schema_version",
        "created_at_utc",
        "scenario",
        "chat_model_a",
        "chat_model_b",
        "embedding_model",
        "config_snapshot",
        "session_status",
    }
)


class SessionSchemaError(ValueError):
    pass


def validate_session_meta(obj: dict[str, Any]) -> None:
    missing = REQUIRED_META_KEYS - obj.keys()
    if missing:
        raise SessionSchemaError(f"session_meta missing keys: {sorted(missing)}")
    if obj["scenario"] not in ("trivial_correlation", "silent_resonance", "conflicting_objectives"):
        raise SessionSchemaError(f"invalid scenario: {obj['scenario']!r}")
    if obj["schema_version"] != 1:
        raise SessionSchemaError(f"unsupported schema_version: {obj['schema_version']}")


def validate_step_record(obj: dict[str, Any]) -> None:
    required = (
        "t",
        "state_before",
        "state_after",
        "raw_text_a",
        "raw_text_b",
        "parsed_action_a",
        "parsed_action_b",
        "parse_ok_a",
        "parse_ok_b",
        "text_embed_a",
        "text_embed_b",
        "embedding_a",
        "embedding_b",
    )
    missing = [k for k in required if k not in obj]
    if missing:
        raise SessionSchemaError(f"step_record missing keys: {missing}")
    if not isinstance(obj["t"], int):
        raise SessionSchemaError("step_record.t must be int")


def iter_session_lines(path: str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if first:
                validate_session_meta(obj)
                first = False
            else:
                validate_step_record(obj)
            yield obj


def dump_jsonl_row(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"
