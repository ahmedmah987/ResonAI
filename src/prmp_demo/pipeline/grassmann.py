"""Grassmann principal angles between subspaces spanned by trajectory windows."""

from __future__ import annotations

import numpy as np


def grassmann_similarity(
    Za: np.ndarray,
    Zb: np.ndarray,
    *,
    window: int,
    r: int,
) -> float:
    """
    Za, Zb shape (T, k). Take last `window` rows, build bases via truncated SVD (rank r).

    Returns sum_{i=1..r_eff} cos^2(theta_i) in [0, r_eff].
    """
    if Za.shape[0] < window:
        return float("nan")
    Ta, k = Za.shape
    Wa = Za[Ta - window : Ta]
    Wb = Zb[Ta - window : Ta]

    def orth_basis(W):
        # W rows time, cols features → columns as vectors in R^{window}? Plan uses k × W matrix from docs:
        # M_a = [z(t-W+1), ..., z(t)] as columns → shape (k, W)
        M = W.T  # k × W
        U, _, _ = np.linalg.svd(M, full_matrices=False)
        rr = min(r, U.shape[1])
        return U[:, :rr]

    Ua = orth_basis(Wa)
    Ub = orth_basis(Wb)
    # Principal angles via SVD of Ua.T @ Ub
    M = Ua.T @ Ub
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return float(np.sum(s**2))


def normalize_grassmann_score(score: float, r: int) -> float:
    if score != score:  # nan
        return float("nan")
    return float(max(0.0, min(1.0, score / max(r, 1))))
