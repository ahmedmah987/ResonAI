"""Build chat messages for cooperative investigators."""

from __future__ import annotations

from typing import Any

from prmp_demo.environment import task_protocol


def message_payload(agent_label: str, state: dict[str, Any], scenario: task_protocol.ScenarioId) -> list[dict[str, str]]:
    system = task_protocol.build_system_prompt(scenario)
    user = task_protocol.user_prompt_block(state, agent_label, scenario)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
