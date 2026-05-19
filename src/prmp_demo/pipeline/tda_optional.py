"""Persistent homology distance W_p between trajectory windows (optional ripser/persim)."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _compress_points_if_needed(points: np.ndarray) -> np.ndarray:
    """Ripser warns/fails when ambient dimension exceeds sample count; PCA-compress."""
    n, d = points.shape
    if n >= d:
        return points
    try:
        from sklearn.decomposition import PCA

        n_comp = max(2, min(n - 1, d))
        return PCA(n_components=n_comp, random_state=0).fit_transform(points)
    except Exception:
        return points[:, : max(2, min(n - 1, d))]


def _finite_h0_diagram(points: np.ndarray) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Return H0 persistence diagram as (n,2) finite birth-death pairs."""
    try:
        from ripser import ripser
    except ImportError:
        return None, "ripser_missing"

    if points.shape[0] < 3:
        return None, "tda_window_too_small"

    points = _compress_points_if_needed(np.asarray(points, dtype=float))

    try:
        res = ripser(points, maxdim=0)
        dgm = np.asarray(res["dgms"][0], dtype=float)
    except Exception as exc:  # noqa: BLE001 — propagate as unavailable
        return None, f"ripser_failed:{type(exc).__name__}"

    if dgm.size == 0:
        return np.zeros((0, 2), dtype=float), None

    finite = np.isfinite(dgm[:, 1])
    dgm_f = dgm[finite]
    if dgm_f.size == 0:
        # Only infinite component — still comparable via cornerpoint at (0, max diameter proxy)
        diam = float(np.max(np.linalg.norm(points - points.mean(axis=0), axis=1)))
        return np.array([[0.0, diam]], dtype=float), None

    return dgm_f, None


def persistence_wasserstein_h0(
    Za: np.ndarray,
    Zb: np.ndarray,
    *,
    window: int,
) -> tuple[float, Optional[str], Optional[float]]:
    """
    Wasserstein distance between H0 diagrams of trailing windows in Z.

    Returns (raw_distance or nan, unavailable_reason or None, optional diagnostic).
    Smaller distance ⇒ more topologically aligned clouds (theory uses Alignment = W_p).
    """
    try:
        import persim
    except ImportError:
        return float("nan"), "persim_missing", None

    Ta, _ = Za.shape
    if Ta < window:
        return float("nan"), "trajectory_shorter_than_window", None

    Wa = Za[Ta - window : Ta]
    Wb = Zb[Ta - window : Ta]

    dgm_a, err_a = _finite_h0_diagram(Wa)
    dgm_b, err_b = _finite_h0_diagram(Wb)
    if err_a:
        return float("nan"), err_a, None
    if err_b:
        return float("nan"), err_b, None
    assert dgm_a is not None and dgm_b is not None

    if dgm_a.shape[0] == 0:
        dgm_a = np.array([[0.0, 1e-6]], dtype=float)
    if dgm_b.shape[0] == 0:
        dgm_b = np.array([[0.0, 1e-6]], dtype=float)

    try:
        dist = float(persim.wasserstein(dgm_a, dgm_b, matching=False))
    except Exception as exc:  # noqa: BLE001
        return float("nan"), f"wasserstein_failed:{type(exc).__name__}", None

    if dist != dist:  # nan
        return float("nan"), "wasserstein_nan", None

    return dist, None, None


def w_p_display_norm(raw_w_p: float) -> float:
    """Map raw Wasserstein distance to [0,1] display score (higher = better alignment)."""
    if raw_w_p != raw_w_p:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 / (1.0 + raw_w_p))))
