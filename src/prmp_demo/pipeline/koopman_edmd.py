"""Light EDMD-style linear Koopman snapshot matrix and spectral overlap between agents."""

from __future__ import annotations

import numpy as np


def _tail(sequence: np.ndarray, window: int) -> np.ndarray:
    T, _ = sequence.shape
    if T < window:
        raise ValueError("sequence shorter than window")
    return sequence[T - window : T]


def edmd_spectral_overlap(
    Za: np.ndarray,
    Zb: np.ndarray,
    *,
    window: int,
    rank: int = 8,
) -> float:
    """
    Fit Z_{t+1} ≈ K Z_t on each trajectory's trailing window (columns = time).
    Compare truncated |λ| spectra via cosine similarity in [0, 1].

    Za, Zb: (T, k). Requires window >= 3 so Z_left has >= 2 columns.
    """
    if Za.shape[0] < window or window < 3:
        return float("nan")

    def spectrum_from_trajectory(Zseq: np.ndarray) -> np.ndarray:
        tail = _tail(Zseq, window)
        z_left = tail[:-1].T
        z_right = tail[1:].T
        if z_left.shape[1] < 1:
            return np.ones(max(rank, 1), dtype=float)
        try:
            K = z_right @ np.linalg.pinv(z_left)
        except np.linalg.LinAlgError:
            return np.ones(max(rank, 1), dtype=float)
        eigvals = np.linalg.eigvals(K)
        mag = np.abs(eigvals).astype(float)
        mag.sort()
        mag = mag[::-1]
        r = min(rank, mag.size)
        if r < 1:
            return np.ones(max(rank, 1), dtype=float)
        vec = mag[:r]
        if vec.sum() < 1e-12:
            return np.ones(max(r, 1), dtype=float)
        return vec

    v1 = spectrum_from_trajectory(Za)
    v2 = spectrum_from_trajectory(Zb)
    m = max(v1.size, v2.size)
    u1 = np.zeros(m)
    u2 = np.zeros(m)
    u1[: v1.size] = v1
    u2[: v2.size] = v2
    u1 /= np.linalg.norm(u1) + 1e-12
    u2 /= np.linalg.norm(u2) + 1e-12
    sim = float(np.dot(u1, u2))
    return max(0.0, min(1.0, sim))


def edmd_overlap_at_end(Za: np.ndarray, Zb: np.ndarray, window: int, rank: int = 8) -> float:
    if Za.shape[0] < window:
        return float("nan")
    return edmd_spectral_overlap(Za, Zb, window=window, rank=rank)
