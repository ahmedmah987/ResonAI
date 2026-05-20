"""Lightweight cooperative puzzle protocol — deterministic state updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

ScenarioId = Literal["trivial_correlation", "silent_resonance", "conflicting_objectives", "open_discussion"]

# Canonical action vocabulary (structured channel).
ACTION_IDS = ("EXPLORE", "HYPOTHESIS", "TEST", "BACKTRACK", "TALK")

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
    state = {
        "scenario": scenario,
        "step_index": 0,
        "room": "study",
        "evidence_slots_filled": 0,
        "joint_hypothesis_score": 0.0,
        "silent_expected_next": SILENT_SEQUENCE[0] if scenario == "silent_resonance" else None,
    }
    if scenario == "open_discussion":
        state["discussion_depth"] = 0
    return state


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


def _extract_rationale_field(raw: str) -> Optional[str]:
    """Pull rationale text from complete or truncated JSON-ish model output."""
    import re

    m = re.search(r'"rationale"\s*:\s*"(.*)', raw, re.DOTALL)
    if not m:
        return None
    body = m.group(1)
    # Trim at closing quote+brace when present; otherwise keep partial (truncated response).
    end = re.search(r'(?<!\\)"\s*\}\s*$', body)
    if end:
        body = body[: end.start()]
    else:
        body = re.sub(r'"\s*\}\s*$', "", body)
    body = body.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
    return body.strip()


def normalize_discussion_text(text: str) -> str:
    """Clean discussion text for display and history."""
    import re

    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    # Drop leaked chain-of-thought / draft markers.
    t = re.sub(r"\*\s*Drafting the content.*$", "", t, flags=re.IGNORECASE | re.DOTALL)
    t = re.sub(r"\(internal monologue[^)]*\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:2000]


def parse_discussion_output(text: str) -> tuple[ParsedAction, bool]:
    """Parse open-discussion replies (plain text preferred; JSON tolerated)."""
    raw = text.strip()
    if not raw:
        return ParsedAction(action_id="TALK", rationale=""), False

    if '"rationale"' in raw or raw.startswith("{"):
        extracted = _extract_rationale_field(raw)
        if extracted:
            return ParsedAction(action_id="TALK", rationale=normalize_discussion_text(extracted)), True
        pa, ok = parse_model_output(raw)
        if ok and pa.rationale and not pa.rationale.strip().startswith("{"):
            return ParsedAction(action_id="TALK", rationale=normalize_discussion_text(pa.rationale)), True

    return ParsedAction(action_id="TALK", rationale=normalize_discussion_text(raw)), True


def parse_model_output(text: str) -> tuple[ParsedAction, bool]:
    """Extract JSON {"action_id": str, "rationale": str} from assistant output."""
    import json
    import re

    raw = text.strip()

    # Try to find JSON block even if surrounded by markdown or text
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
    if not m:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            candidate = m.group(0)
            if "```json" in candidate:
                candidate = candidate.split("```json")[-1].split("```")[0]
            elif "```" in candidate:
                candidate = candidate.split("```")[-1].split("```")[0]

            candidate = candidate.strip()
            obj = json.loads(candidate)
            aid = str(obj.get("action_id", "")).upper()
            rat = str(obj.get("rationale", ""))
            if aid in ACTION_IDS:
                return ParsedAction(action_id=aid, rationale=rat), True
            if "action_id" in obj and (aid == "TALK" or "TALK" in aid):
                return ParsedAction(action_id="TALK", rationale=rat), True
        except json.JSONDecodeError:
            extracted = _extract_rationale_field(raw)
            if extracted:
                return ParsedAction(action_id="TALK", rationale=extracted), False

    return ParsedAction(action_id="EXPLORE", rationale=raw[:1000]), False


def append_history_turn(state: dict[str, Any], agent_label: str, parsed: ParsedAction) -> dict[str, Any]:
    """Return a copy of state with one agent turn appended (for turn-based open discussion)."""
    s = dict(state)
    history = list(s.get("history", []))
    history.append({"agent": agent_label, "rationale": parsed.rationale})
    s["history"] = history
    return s


def apply_actions_sequential(
    state_before: dict[str, Any],
    parsed_a: ParsedAction,
    parsed_b: ParsedAction,
) -> dict[str, Any]:
    """Apply agent A then agent B deterministically."""
    s = dict(state_before)
    s["step_index"] = state_before.get("step_index", 0)
    scen = state_before.get("scenario")

    # Store history for discussion scenarios (avoid duplicating A if turn-based flow already added it)
    if scen == "open_discussion":
        hist = list(s.get("history", []))
        if not hist or hist[-1].get("agent") != "A" or hist[-1].get("rationale") != parsed_a.rationale:
            hist.append({"agent": "A", "rationale": parsed_a.rationale})
        hist.append({"agent": "B", "rationale": parsed_b.rationale})
        s["history"] = hist

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
        elif aid == "TALK":
            # Open discussion just increments a "depth" or "progress" counter
            s["discussion_depth"] = int(s.get("discussion_depth", 0)) + 1

    bump(parsed_a)
    bump(parsed_b)

    si = int(s["step_index"]) + 1
    s["step_index"] = si

    if scen == "silent_resonance":
        exp = expected_action_for_step("silent_resonance", si)
        s["silent_expected_next"] = exp

    return s


def user_prompt_block(state: dict[str, Any], agent_label: str, scenario: ScenarioId, topic: Optional[str] = None, redirect_hint: Optional[str] = None) -> str:
    """Observable task description shown to the model."""
    if scenario == "open_discussion":
        t = topic or "The future of autonomous coordination and topological resonance."
        history_str = ""
        if "history" in state and state["history"]:
            history_str = "\n\nRecent Discussion History:\n"
            # Show last 4 turns to keep context window clean but relevant
            for entry in state["history"][-6:]:
                history_str += f"Agent {entry['agent']}: {entry['rationale']}\n"

        hint_str = f"\nAlignment note: {redirect_hint}\n" if redirect_hint else ""

        return (
            f"You are Agent {agent_label} in a deep philosophical and technical discussion.\n"
            f"Topic: {t}\n"
            f"{history_str}"
            f"{hint_str}\n"
            "Rules:\n"
            "- Reply in plain English only (2–5 sentences).\n"
            "- Do NOT use JSON, markdown code blocks, or bullet lists.\n"
            "- Do NOT include internal notes, drafts, or phrases like 'I'd like to build upon'.\n"
            "- Respond directly to the other agent's last point; add one new idea.\n"
        )

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
    if scenario == "open_discussion":
        return (
            "You are a sophisticated AI agent in a multi-agent discussion. "
            "Output only your spoken contribution as plain text—no JSON, no metadata, no preamble."
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
