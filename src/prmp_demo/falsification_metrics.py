"""Δ_RG, η_topo, phenomenological regime labels, session-level τ_lead summary."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from prmp_demo.config import DemoConfig

PhenoState = Literal[
    "unknown",
    "stochastic_independence",
    "phase_locking",
    "emergent_coherence",
    "forced_synchronization",
]


def delta_resonance_gap(gamma_vec: list[float], rho_text: float) -> float | None:
    """Δ_RG = ||Γ|| − |ρ| (embedding-channel similarity proxy)."""
    if len(gamma_vec) != 3:
        return None
    g = np.asarray(gamma_vec, dtype=float)
    return float(np.linalg.norm(g) - abs(float(rho_text)))


def eta_topo_ratio(w_p_raw_curr: float | None, w_p_raw_prev: float | None, kappa: float | None, eps: float = 1e-8) -> float | None:
    """η_topo ≈ Ẇ_p / κ — discrete derivative of raw W_p vs curvature proxy."""
    if w_p_raw_curr is None or w_p_raw_prev is None or kappa is None:
        return None
    if any(math.isnan(x) for x in (w_p_raw_curr, w_p_raw_prev)):
        return None
    if kappa != kappa:
        return None
    dot_wp = float(w_p_raw_curr - w_p_raw_prev)
    return float(dot_wp / (kappa + eps))


def classify_phenomenological(
    gamma_vec: list[float],
    *,
    dw_p_norm_dt: float | None,
    dr_spec_dt: float | None,
    w_p_norm: float,
    r_spec: float,
    kappa: float | None,
    gamma_norm_var_recent: float | None,
) -> PhenoState:
    """
    Rule-based mapping inspired by not-named-yet.md phenomenology table.
    Uses coarse thresholds on trajectory derivatives (finite differences).
    """
    gn = float(np.linalg.norm(np.asarray(gamma_vec, dtype=float)))
    kap = 0.0 if kappa is None or kappa != kappa else float(kappa)

    low_flow = r_spec < 0.25 and w_p_norm < 0.25
    grad_wp_small = dw_p_norm_dt is None or abs(dw_p_norm_dt) < 0.08

    if low_flow and grad_wp_small:
        return "stochastic_independence"

    if dw_p_norm_dt is not None and dw_p_norm_dt > 0.05 and dr_spec_dt is not None and dr_spec_dt > 0.02:
        return "phase_locking"

    stable_gamma = gamma_norm_var_recent is not None and gamma_norm_var_recent < 0.02 and gn > 0.5
    if w_p_norm > 0.55 and stable_gamma:
        return "emergent_coherence"

    if gn > 0.85 and kap > 0.15:
        return "forced_synchronization"

    return "unknown"


def augment_step_falsification(
    metrics_t: dict[str, Any],
    *,
    gamma_prev: list[float] | None,
    gamma_hist_recent: np.ndarray,
) -> None:
    """Mutates metrics_t with delta_rg, eta_topo, phenomenological_state."""
    gamma = metrics_t.get("gamma") or [0.0, 0.0, 0.0]
    rho = float(metrics_t.get("rho_text", 0.0))
    drg = delta_resonance_gap(gamma, rho)
    metrics_t["delta_rg"] = drg

    wp_curr = metrics_t.get("w_p_raw")
    wp_prev = metrics_t.get("_prev_w_p_raw")
    kap = metrics_t.get("kappa_gamma")
    wp_c = wp_curr if isinstance(wp_curr, (int, float)) and wp_curr == wp_curr else None
    wp_p = wp_prev if isinstance(wp_prev, (int, float)) and wp_prev == wp_prev else None
    kap_v = kap if isinstance(kap, (int, float)) and kap == kap else None
    eta = eta_topo_ratio(wp_c, wp_p, kap_v)
    metrics_t["eta_topo"] = eta

    # Finite differences on displayed scalars (prefix indices end at current t)
    r_curr = float(gamma[0])
    wn_curr = float(gamma[1])
    if gamma_prev is not None:
        dr_spec_dt = r_curr - float(gamma_prev[0])
        dw_p_norm_dt = wn_curr - float(gamma_prev[1])
    else:
        dr_spec_dt = None
        dw_p_norm_dt = None

    var_recent = None
    if gamma_hist_recent.shape[0] >= 2:
        norms = np.linalg.norm(gamma_hist_recent, axis=1)
        var_recent = float(np.var(norms))

    metrics_t["phenomenological_state"] = classify_phenomenological(
        gamma,
        dw_p_norm_dt=dw_p_norm_dt,
        dr_spec_dt=dr_spec_dt,
        w_p_norm=wn_curr,
        r_spec=r_curr,
        kappa=kap if isinstance(kap, (int, float)) else None,
        gamma_norm_var_recent=var_recent,
    )


def compute_session_analysis_summary(
    steps_metrics: list[dict[str, Any]],
    cfg: DemoConfig,
) -> dict[str, Any]:
    """τ_lead session heuristic + precipice flags using unity_proxy & csd_spike."""
    if not steps_metrics:
        return {"tau_lead": None, "precipice_flag": False, "notes": "empty"}

    t_csd = None
    t_unity = None
    for i, m in enumerate(steps_metrics):
        if m.get("csd_spike"):
            t_csd = i if t_csd is None else t_csd
        if m.get("unity_proxy"):
            t_unity = i if t_unity is None else t_unity

    tau_lead = None
    if t_csd is not None and t_unity is not None:
        tau_lead = int(t_unity - t_csd)

    precipice = False
    if t_unity is not None and t_csd is None:
        precipice = True
    elif tau_lead is not None and tau_lead <= 0:
        precipice = True

    return {
        "tau_lead_steps": tau_lead,
        "first_csd_spike_t": t_csd,
        "first_unity_proxy_t": t_unity,
        "precipice_flag": precipice,
        "thresholds": {
            "unity_gamma_norm": cfg.unity_gamma_norm_threshold,
            "unity_w_p_norm": cfg.unity_w_p_norm_threshold,
            "csd_variance_multiplier": cfg.csd_variance_multiplier,
        },
    }


def attach_w_p_prev_chain(metrics_list: list[dict[str, Any]]) -> None:
    """Populate _prev_w_p_raw for η_topo."""
    prev = None
    for m in metrics_list:
        curr = m.get("w_p_raw")
        if prev is not None:
            m["_prev_w_p_raw"] = prev
        else:
            m["_prev_w_p_raw"] = None
        if isinstance(curr, (int, float)) and curr == curr:
            prev = float(curr)
