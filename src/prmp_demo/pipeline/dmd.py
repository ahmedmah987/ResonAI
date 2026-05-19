"""
Dynamic Mode Decomposition (truncated) and spectral overlap proxy.

Definitions:
  Given snapshots matrix Z of shape (k_features, window_length-1) built from z(t)
  columns consecutive in time, we approximate modes via SVD(Z) = U Σ V^T and form
  a discrete spectrum magnitude vector from singular values.

  R_spec(t) is cosine similarity between normalized squared singular-value spectra
  for agent A and agent B on the same time indices (paired columns).
"""

from __future__ import annotations

import numpy as np


def _snapshot_matrices(sequence: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    sequence shape (T, k_features).
    Returns Z, Zprime each shape (k_features, window-1) using indices ending at T-1.
    """
    T, k = sequence.shape
    if T < window or window < 3:
        raise ValueError("window too large or small")
    tail = sequence[T - window : T]
    Z = tail[:-1].T
    Zp = tail[1:].T
    return Z, Zp


def _spectrum_from_Z(Z: np.ndarray, rank: int = 8) -> np.ndarray:
    """Approx spectrum vector from truncated SVD of Z (snapshot columns along time)."""
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    r = min(rank, len(s))
    mag = (s[:r] ** 2).astype(float)
    if mag.sum() < 1e-12:
        return np.ones(max(r, 1), dtype=float)
    return mag


def spectral_overlap_score(
    Za: np.ndarray,
    Zb: np.ndarray,
    *,
    window: int,
    rank: int = 8,
) -> float:
    """
    Za, Zb shape (T, k). Uses windows ending at last row (caller trims series per step externally).

    Returns cosine similarity in [0,1] between padded spectral magnitude vectors.
    """
    Z1, Z1p = _snapshot_matrices(Za, window)
    Z2, Z2p = _snapshot_matrices(Zb, window)
    del Z1p, Z2p

    v1 = _spectrum_from_Z(Z1, rank=rank)
    v2 = _spectrum_from_Z(Z2, rank=rank)
    m = max(v1.size, v2.size)
    u1 = np.zeros(m)
    u2 = np.zeros(m)
    u1[: v1.size] = v1
    u2[: v2.size] = v2
    u1 /= np.linalg.norm(u1) + 1e-12
    u2 /= np.linalg.norm(u2) + 1e-12
    sim = float(np.dot(u1, u2))
    return max(0.0, min(1.0, sim))


def spectral_overlap_at_end(Za: np.ndarray, Zb: np.ndarray, window: int, rank: int = 8) -> float:
    """If insufficient rows, return NaN."""
    if Za.shape[0] < window:
        return float("nan")
    return spectral_overlap_score(Za, Zb, window=window, rank=rank)
