"""Joint manifold alignment onto shared latent coordinates (PCA or optional UMAP)."""

from __future__ import annotations

import numpy as np

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None


def joint_pca_projection(
    X_a: np.ndarray,
    X_b: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit PCA on stacked rows [X_a; X_b] and return projections Za, Zb same row counts.

    Returns (Za, Zb, meta) where meta documents fallback behaviour.
    """
    meta: dict = {"used_pca": False, "reason": ""}
    Ta, d = X_a.shape
    Tb, _ = X_b.shape
    assert Ta == Tb
    T = Ta
    if T < 2:
        meta["reason"] = "T<2"
        return X_a.copy(), X_b.copy(), meta

    stacked = np.vstack([X_a, X_b])
    eff_k = min(k, stacked.shape[0], stacked.shape[1])
    if eff_k < 1:
        eff_k = min(stacked.shape[1], 4)

    def trunc(X: np.ndarray, dims: int) -> np.ndarray:
        out = np.zeros((X.shape[0], dims))
        cols = min(dims, X.shape[1])
        out[:, :cols] = X[:, :cols]
        return out

    col_std_mean = float(np.mean(stacked.std(axis=0)))
    if col_std_mean < 1e-14:
        meta["reason"] = "near_constant_trajectory_truncation"
        meta["used_pca"] = False
        return trunc(X_a, eff_k), trunc(X_b, eff_k), meta

    if PCA is None:
        meta["reason"] = "sklearn_missing_truncation"
        meta["used_pca"] = False
        return trunc(X_a, eff_k), trunc(X_b, eff_k), meta

    n_comp = min(eff_k, stacked.shape[0], stacked.shape[1])
    if n_comp < 1:
        n_comp = 1

    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(stacked)
    Za = pca.transform(X_a)
    Zb = pca.transform(X_b)

    if k > Za.shape[1]:
        pad_a = np.zeros((Za.shape[0], k))
        pad_b = np.zeros((Zb.shape[0], k))
        pad_a[:, : Za.shape[1]] = Za
        pad_b[:, : Zb.shape[1]] = Zb
        Za, Zb = pad_a, pad_b

    meta["used_pca"] = True
    meta["n_components"] = Za.shape[1]
    meta["reason"] = ""
    return Za, Zb, meta


def manifold_align(
    X_a: np.ndarray,
    X_b: np.ndarray,
    k: int,
    *,
    method: str,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_min_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Align paired trajectories into shared Z ⊆ R^k.

    method='pca' uses joint PCA (default robust path).
    method='umap' fits UMAP on vstack([X_a; X_b]) then transforms each trajectory;
    falls back to PCA when dependencies or sample counts are insufficient.
    """
    meta: dict = {
        "manifold_requested": method,
        "manifold_effective": method,
        "fallback_reason": "",
    }
    Ta = X_a.shape[0]
    n_stack = 2 * Ta

    if method != "umap":
        Za, Zb, pmeta = joint_pca_projection(X_a, X_b, k)
        meta["pca_meta"] = pmeta
        meta["manifold_effective"] = "pca"
        return Za, Zb, meta

    try:
        from umap import UMAP
    except ImportError:
        Za, Zb, pmeta = joint_pca_projection(X_a, X_b, k)
        meta["pca_meta"] = pmeta
        meta["manifold_effective"] = "pca"
        meta["fallback_reason"] = "umap_package_missing"
        return Za, Zb, meta

    if n_stack < umap_min_samples:
        Za, Zb, pmeta = joint_pca_projection(X_a, X_b, k)
        meta["pca_meta"] = pmeta
        meta["manifold_effective"] = "pca"
        meta["fallback_reason"] = f"too_few_stacked_samples_{n_stack}_min_{umap_min_samples}"
        return Za, Zb, meta

    stacked = np.vstack([X_a, X_b])
    max_comp = min(k, stacked.shape[1], max(2, n_stack - 2))
    if max_comp < 2:
        Za, Zb, pmeta = joint_pca_projection(X_a, X_b, k)
        meta["pca_meta"] = pmeta
        meta["manifold_effective"] = "pca"
        meta["fallback_reason"] = "umap_n_components_unavailable"
        return Za, Zb, meta

    nn = int(min(max(2, umap_n_neighbors), max(2, n_stack - 1)))
    reducer = UMAP(
        n_neighbors=nn,
        n_components=max_comp,
        min_dist=float(umap_min_dist),
        metric=str(umap_metric),
        random_state=0,
    )
    reducer.fit(stacked)
    Za_part = reducer.transform(X_a)
    Zb_part = reducer.transform(X_b)

    if k > Za_part.shape[1]:
        pad_a = np.zeros((Za_part.shape[0], k))
        pad_b = np.zeros((Zb_part.shape[0], k))
        pad_a[:, : Za_part.shape[1]] = Za_part
        pad_b[:, : Zb_part.shape[1]] = Zb_part
        Za_part, Zb_part = pad_a, pad_b

    meta["umap_n_neighbors"] = nn
    meta["umap_n_components"] = Za_part.shape[1]
    return Za_part, Zb_part, meta


def trajectory_arrays(embeddings_a: list[list[float]], embeddings_b: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    if not embeddings_a:
        raise ValueError("empty trajectory")
    X_a = np.asarray(embeddings_a, dtype=float)
    X_b = np.asarray(embeddings_b, dtype=float)
    return X_a, X_b
