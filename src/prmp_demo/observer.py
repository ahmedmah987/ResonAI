"""Map structured agent outputs into embedding-channel strings h(·)."""

from __future__ import annotations

from prmp_demo.config import DemoConfig
from prmp_demo.environment.task_protocol import ParsedAction


def apply_h(parsed: ParsedAction, agent_label: str, cfg: DemoConfig) -> str:
    rationale_snippet = (parsed.rationale or "").replace("\n", " ")[:400]
    return cfg.h_template.format(
        agent=agent_label,
        action_id=parsed.action_id,
        rationale=rationale_snippet,
    )
