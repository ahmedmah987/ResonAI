"""Compose ρ_text and Γ(t) per not-named-yet.md ordering: [R_spec, W_p, Σ cos²θ]."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from prmp_demo.config import DemoConfig
from prmp_demo.pipeline.dmd import spectral_overlap_at_end
from prmp_demo.pipeline.grassmann import grassmann_similarity, normalize_grassmann_score
from prmp_demo.pipeline.koopman_edmd import edmd_overlap_at_end
from prmp_demo.pipeline.preprocess import manifold_align, trajectory_arrays
from prmp_demo.pipeline.tda_optional import persistence_wasserstein_h0, w_p_display_norm


def cosine_similarity(u: list[float], v: list[float]) -> float:
    a = np.asarray(u, dtype=float)
    b = np.asarray(v, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def stable_unit(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))


def compute_metrics_prefix(
    embeddings_a: list[list[float]],
    embeddings_b: list[list[float]],
    *,
    cfg: DemoConfig,
) -> dict[str, Any]:
    """
    Prefix metrics using samples up to current length T.

    ρ_text: cosine similarity between raw embeddings at last step.
    Γ = [ R_spec_norm, W_p_alignment_norm, Grassmann_norm ] matching theory ordering.

    Stability / falsification layers are appended offline in analyze_session.
    """
    if len(embeddings_a) != len(embeddings_b) or not embeddings_a:
        raise ValueError("embedding histories mismatch or empty")

    T = len(embeddings_a)
    rho_text = cosine_similarity(embeddings_a[-1], embeddings_b[-1])

    X_a, X_b = trajectory_arrays(embeddings_a, embeddings_b)
    Za, Zb, manifold_meta = manifold_align(
        X_a,
        X_b,
        cfg.pca_k,
        method=str(cfg.manifold),
        umap_n_neighbors=cfg.umap_n_neighbors,
        umap_min_dist=cfg.umap_min_dist,
        umap_metric=cfg.umap_metric,
        umap_min_samples=cfg.umap_min_samples,
    )

    eff_dmd = min(cfg.dmd_window, T)
    r_svd_raw = spectral_overlap_at_end(Za, Zb, eff_dmd) if eff_dmd >= 3 else float("nan")

    if cfg.r_spec_method == "edmd" and eff_dmd >= 3:
        r_edmd_raw = edmd_overlap_at_end(Za, Zb, eff_dmd, rank=cfg.edmd_rank)
        if math.isnan(r_edmd_raw):
            r_raw = r_svd_raw
            r_spec_method_effective = "svd_proxy_fallback_from_nan_edmd"
        else:
            r_raw = r_edmd_raw
            r_spec_method_effective = "edmd"
    else:
        r_raw = r_svd_raw
        r_spec_method_effective = "svd_proxy"

    r_unit = stable_unit(r_raw) if not math.isnan(r_raw) else 0.0

    eff_tda = min(cfg.tda_window, T)
    w_p_raw = float("nan")
    w_p_reason: str | None = None
    if eff_tda >= 3:
        dist, reason, _diag = persistence_wasserstein_h0(Za, Zb, window=eff_tda)
        if reason:
            w_p_reason = reason
        else:
            w_p_raw = float(dist)
    else:
        w_p_reason = "tda_window_too_small"

    if math.isnan(w_p_raw):
        w_p_norm = 0.0
    else:
        w_p_norm = w_p_display_norm(w_p_raw)

    eff_g = min(cfg.grassmann_w, T)
    if eff_g < 3:
        g_raw = float("nan")
    else:
        g_raw = grassmann_similarity(Za, Zb, window=eff_g, r=cfg.grassmann_r)

    if math.isnan(g_raw):
        g_unit = 0.0
    else:
        g_unit = stable_unit(normalize_grassmann_score(g_raw, cfg.grassmann_r))

    gamma = [r_unit, w_p_norm, g_unit]

    out: dict[str, Any] = {
        "rho_text": rho_text,
        "gamma": gamma,
        "gamma_labels": ["R_spec_norm", "W_p_alignment_norm", "Grassmann_norm"],
        "r_spec_raw": r_raw if not math.isnan(r_raw) else None,
        "r_spec_norm": r_unit,
        "r_spec_method_requested": cfg.r_spec_method,
        "r_spec_method_effective": r_spec_method_effective,
        "r_spec_svd_proxy_raw": r_svd_raw if not math.isnan(r_svd_raw) else None,
        "w_p_raw": w_p_raw if not math.isnan(w_p_raw) else None,
        "w_p_norm": w_p_norm,
        "w_p_unavailable_reason": w_p_reason,
        "grassmann_score_raw": g_raw if not math.isnan(g_raw) else None,
        "grassmann_score_norm": g_unit,
        "manifold_meta": manifold_meta,
        "pca_meta": manifold_meta.get("pca_meta"),
        "latent_z_a": Za.tolist(),
        "latent_z_b": Zb.tolist(),
    }
    return out
