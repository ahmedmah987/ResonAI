"""Predictive stability proxies: finite-time exponent on Γ, CSD variance, discrete curvature."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _gamma_norms(gamma_hist: np.ndarray) -> np.ndarray:
    """gamma_hist shape (T, 3)."""
    return np.linalg.norm(gamma_hist, axis=1)


def lambda_gamma_ftle_proxy(gamma_hist: np.ndarray, t: int, window: int) -> float | None:
    """
    Rough finite-time exponent from log(||ΔΓ||) vs step index over a trailing window.

    Positive ⇒ nearby-steps divergence (transient alignment); negative ⇒ contraction-
    like behaviour in Γ increments (operational proxy, not rigorous λ_Γ from Jacobian).
    """
    if window < 3 or t < 1:
        return None
    start = max(1, t - window + 1)
    segment = gamma_hist[start : t + 1]
    if segment.shape[0] < 3:
        return None
    deltas = np.linalg.norm(np.diff(segment, axis=0), axis=1)
    if np.any(deltas <= 0):
        deltas = np.maximum(deltas, 1e-12)
    log_d = np.log(deltas)
    x = np.arange(log_d.size, dtype=float)
    xv = x - x.mean()
    yv = log_d - log_d.mean()
    denom = float(np.dot(xv, xv)) + 1e-12
    slope = float(np.dot(xv, yv) / denom)
    if math.isnan(slope):
        return None
    return slope


def csd_variance_score(gamma_hist: np.ndarray, t: int, var_window: int) -> float | None:
    """Rolling variance of ||Γ|| — spike interpreted as critical slowing / tension."""
    norms = _gamma_norms(gamma_hist)
    end = t + 1
    start = max(0, end - var_window)
    seg = norms[start:end]
    if seg.size < 2:
        return None
    return float(np.var(seg))


def kappa_gamma_path(gamma_hist: np.ndarray, t: int) -> float | None:
    """Discrete curvature proxy via second differences of Γ in R³."""
    if t < 1 or t >= gamma_hist.shape[0] - 1:
        return None
    g = gamma_hist
    acc = g[t + 1] - 2.0 * g[t] + g[t - 1]
    return float(np.linalg.norm(acc))


def stability_bundle_at_t(
    gamma_hist: np.ndarray,
    t: int,
    *,
    stability_window: int,
    csd_variance_window: int,
    csd_multiplier: float,
    unity_gamma_norm_threshold: float,
    unity_w_p_norm_threshold: float,
    w_p_norm_at_t: float,
) -> dict[str, Any]:
    lam = lambda_gamma_ftle_proxy(gamma_hist, t, stability_window)
    csd_v = csd_variance_score(gamma_hist, t, csd_variance_window)
    kap = kappa_gamma_path(gamma_hist, t)

    norms = _gamma_norms(gamma_hist)
    baseline_window = max(3, min(t + 1, csd_variance_window * 3))
    baseline_vals: list[float] = []
    for i in range(max(1, t + 1 - baseline_window), t + 1):
        cv = csd_variance_score(gamma_hist, i, csd_variance_window)
        if cv is not None:
            baseline_vals.append(cv)
    baseline_med = float(np.median(baseline_vals)) if baseline_vals else None

    csd_spike = False
    if csd_v is not None and baseline_med is not None and baseline_med > 1e-12:
        csd_spike = bool(csd_v > csd_multiplier * baseline_med)

    gn = float(norms[t]) if t < norms.size else 0.0
    unity_proxy = bool(gn >= unity_gamma_norm_threshold and w_p_norm_at_t >= unity_w_p_norm_threshold)

    return {
        "lambda_gamma": lam,
        "csd_variance": csd_v,
        "kappa_gamma": kap,
        "csd_spike": csd_spike,
        "unity_proxy": unity_proxy,
        "gamma_norm": gn,
    }
