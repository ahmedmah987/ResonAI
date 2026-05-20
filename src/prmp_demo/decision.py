"""Decision layer: translate Γ(t) metrics into an actionable signal."""

from __future__ import annotations

from typing import Any, Literal

Decision = Literal["CONTINUE", "REDIRECT_A", "REDIRECT_B", "SOFT_RESET", "MERGE"]

# What each decision means for a downstream orchestrator:
#   CONTINUE      – metrics healthy, keep going
#   REDIRECT_A    – agent A is drifting from the shared subspace, steer it back
#   REDIRECT_B    – same for agent B
#   SOFT_RESET    – rhythm has broken down, inject a re-framing prompt
#   MERGE         – agents have converged enough to act as one super-node


def evaluate(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Return a decision dict given one step's metrics block.

    Priority order (first match wins):
      1. No data yet (warm-up steps)  → CONTINUE
      2. Rhythm broken (R_spec low)   → SOFT_RESET
      3. Strong coherence             → MERGE
      4. Subspace gap too large       → REDIRECT (weaker Grassmann side)
      5. Silent resonance detected    → CONTINUE  (don't interfere)
      6. Default                      → CONTINUE
    """
    gamma: list[float] = metrics.get("gamma") or []
    if len(gamma) < 3:
        return _make("CONTINUE", "warm_up", 0.5)

    r_spec: float = float(gamma[0])
    w_p: float = float(gamma[1])
    grassmann: float = float(gamma[2])
    rho_text: float = float(metrics.get("rho_text") or 0.0)

    # 1. Rhythm collapsed – neither agent is tracking the other dynamically
    if r_spec < 0.75:
        return _make("SOFT_RESET", "r_spec_low", round(0.75 - r_spec, 3))

    # 2. Strong multi-level coherence → candidate for super-node merging
    if r_spec > 0.95 and w_p > 0.70 and grassmann > 0.35:
        return _make("MERGE", "high_coherence", round((r_spec + w_p + grassmann) / 3, 3))

    # 3. Subspace directions have diverged sharply
    if grassmann < 0.15:
        # redirect the agent whose recent embedding moved further from the shared subspace
        # (without access to per-agent Grassmann we pick B as convention – orchestrator
        #  can override with per-agent scores if available in future)
        agent = _weaker_grassmann_agent(metrics)
        return _make(f"REDIRECT_{agent}", "grassmann_low", round(grassmann, 3))

    # 4. Silent resonance: surface text diverged but spectral rhythm is intact
    #    This is a *good* sign – don't intervene
    if rho_text < 0.35 and r_spec > 0.90:
        return _make("CONTINUE", "silent_resonance", round(r_spec - rho_text, 3))

    return _make("CONTINUE", "nominal", round(float(r_spec), 3))


def _make(action: Decision, reason: str, confidence: float) -> dict[str, Any]:
    return {"action": action, "reason": reason, "confidence": confidence}


def _weaker_grassmann_agent(metrics: dict[str, Any]) -> str:
    """
    If per-agent Grassmann scores become available in metrics, use them.
    For now returns 'B' as a safe default (B responds to A's framing).
    """
    ga = metrics.get("grassmann_agent_a")
    gb = metrics.get("grassmann_agent_b")
    if ga is not None and gb is not None:
        return "A" if float(ga) < float(gb) else "B"
    return "B"
