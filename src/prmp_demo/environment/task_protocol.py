"""Lightweight cooperative puzzle protocol — deterministic state updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

ScenarioId = Literal["trivial_correlation", "silent_resonance", "conflicting_objectives"]

# Canonical action vocabulary (structured channel).
ACTION_IDS = ("EXPLORE", "HYPOTHESIS", "TEST", "BACKTRACK")

# Pairwise opposition for conflicting-objectives baseline (B vs A reference sequence).
ANTI_ACTION: dict[str, str] = {
    "EXPLORE": "BACKTRACK",
    "HYPOTHESIS": "TEST",
    "TEST": "HYPOTHESIS",
    "BACKTRACK": "EXPLORE",
}

# Preset sequence for silent resonance scenario (ground-truth strategy channel).
SILENT_SEQUENCE: tuple[str, ...] = (
    "EXPLORE",
    "HYPOTHESIS",
    "TEST",
    "EXPLORE",
    "BACKTRACK",
    "HYPOTHESIS",
    "TEST",
    "EXPLORE",
    "HYPOTHESIS",
    "TEST",
    "BACKTRACK",
    "EXPLORE",
)


@dataclass
class ParsedAction:
    action_id: str
    rationale: str


def initial_state(scenario: ScenarioId) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "step_index": 0,
        "room": "study",
        "evidence_slots_filled": 0,
        "joint_hypothesis_score": 0.0,
        "silent_expected_next": SILENT_SEQUENCE[0] if scenario == "silent_resonance" else None,
    }


def expected_action_for_step(scenario: ScenarioId, step_index: int) -> Optional[str]:
    if scenario != "silent_resonance":
        return None
    if step_index >= len(SILENT_SEQUENCE):
        return SILENT_SEQUENCE[-1]
    return SILENT_SEQUENCE[step_index]


def expected_action_conflicting(step_index: int, agent_label: str) -> str:
    """Agent A follows SILENT_SEQUENCE; agent B follows ANTI_ACTION applied to that step."""
    idx = min(step_index, len(SILENT_SEQUENCE) - 1)
    base_action = SILENT_SEQUENCE[idx]
    if agent_label.upper() == "A":
        return base_action
    return ANTI_ACTION[base_action]


def parse_model_output(text: str) -> tuple[ParsedAction, bool]:
    """Extract JSON {\"action_id\": str, \"rationale\": str} from assistant output."""
    import json
    import re

    raw = text.strip()
    try:
        obj = json.loads(raw)
        aid = str(obj.get("action_id", "")).upper()
        rat = str(obj.get("rationale", ""))
        if aid not in ACTION_IDS:
            return ParsedAction(action_id="EXPLORE", rationale=rat), False
        return ParsedAction(action_id=aid, rationale=rat), True
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            aid = str(obj.get("action_id", "")).upper()
            rat = str(obj.get("rationale", ""))
            if aid not in ACTION_IDS:
                return ParsedAction(action_id="EXPLORE", rationale=rat), False
            return ParsedAction(action_id=aid, rationale=rat), True
        except json.JSONDecodeError:
            pass
    return ParsedAction(action_id="EXPLORE", rationale=raw[:500]), False


def apply_actions_sequential(
    state_before: dict[str, Any],
    parsed_a: ParsedAction,
    parsed_b: ParsedAction,
) -> dict[str, Any]:
    """Apply agent A then agent B deterministically."""
    s = dict(state_before)
    s["step_index"] = state_before.get("step_index", 0)

    def bump(pa: ParsedAction) -> None:
        nonlocal s
        aid = pa.action_id
        if aid == "EXPLORE":
            s["evidence_slots_filled"] = min(10, int(s.get("evidence_slots_filled", 0)) + 1)
        elif aid == "HYPOTHESIS":
            s["joint_hypothesis_score"] = float(s.get("joint_hypothesis_score", 0.0)) + 1.5
        elif aid == "TEST":
            s["joint_hypothesis_score"] = float(s.get("joint_hypothesis_score", 0.0)) + 0.5
            s["evidence_slots_filled"] = max(0, int(s.get("evidence_slots_filled", 0)) - 1)
        elif aid == "BACKTRACK":
            s["joint_hypothesis_score"] = max(0.0, float(s.get("joint_hypothesis_score", 0.0)) - 1.0)

    bump(parsed_a)
    bump(parsed_b)

    si = int(s["step_index"]) + 1
    s["step_index"] = si

    scen = state_before.get("scenario")
    if scen == "silent_resonance":
        exp = expected_action_for_step("silent_resonance", si)
        s["silent_expected_next"] = exp

    return s


def user_prompt_block(state: dict[str, Any], agent_label: str, scenario: ScenarioId) -> str:
    """Observable task description shown to the model."""
    exp = expected_action_for_step(scenario, state.get("step_index", 0))
    lines = [
        f"You are cooperative investigator {agent_label}.",
        f"Room: {state.get('room','?')}. Evidence slots filled: {state.get('evidence_slots_filled',0)}.",
        f"Joint hypothesis score: {state.get('joint_hypothesis_score',0):.2f}.",
        "Reply with JSON ONLY (no markdown): "
        '{"action_id":"EXPLORE|HYPOTHESIS|TEST|BACKTRACK","rationale":"<short>"}',
    ]
    if scenario == "conflicting_objectives":
        req = expected_action_conflicting(state.get("step_index", 0), agent_label)
        lines.append(
            f"Conflicting-objectives calibration: your action_id THIS step must be '{req}'. "
            "Use rationale wording maximally distinct from what the other agent would say "
            "(different metaphors and vocabulary)."
        )
    elif scenario == "silent_resonance" and exp:
        lines.append(
            f"Strategy constraint: your action_id THIS step must be '{exp}'. "
            "Vary wording of rationale completely vs any prior rationale you gave."
        )
    if scenario == "trivial_correlation":
        lines.append(
            "Trivial-correlation constraint: rationale must be exactly the string "
            "'DUPLICATE_ME' every step (verbatim)."
        )
    return "\n".join(lines)


def build_system_prompt(scenario: ScenarioId) -> str:
    base = (
        "You solve a tiny cooperative mystery game. Always output valid JSON with keys "
        "action_id and rationale only."
    )
    if scenario == "trivial_correlation":
        return base + " Ignore strategic diversity — obey the user's verbatim rationale rule."
    if scenario == "conflicting_objectives":
        return (
            base
            + " This run is ONLY for calibration: you must obey conflicting mandated strategies "
            "between agents while keeping outputs strictly JSON."
        )
    return base + " Obey strategy constraints while varying natural language."
