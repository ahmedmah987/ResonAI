"""Load immutable demo configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ManifoldMethod = Literal["pca", "umap"]
RSpecMethod = Literal["svd_proxy", "edmd"]


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if raw is None:
        return default
    return int(raw)


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    if raw is None:
        return default
    return float(raw)


@dataclass(frozen=True)
class DemoConfig:
    openrouter_api_key: Optional[str]
    chat_model_a: str
    chat_model_b: str
    embedding_model: str
    manifold: ManifoldMethod
    umap_n_neighbors: int
    umap_min_dist: float
    umap_metric: str
    umap_min_samples: int
    pca_k: int
    r_spec_method: RSpecMethod
    edmd_rank: int
    dmd_window: int
    grassmann_w: int
    grassmann_r: int
    tda_window: int
    stability_window: int
    csd_variance_window: int
    unity_gamma_norm_threshold: float
    unity_w_p_norm_threshold: float
    csd_variance_multiplier: float
    max_steps: int
    max_tokens: int
    temperature: float
    h_template: str
    http_referer: Optional[str]
    http_title: Optional[str]


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _candidate_env_paths() -> list[Path]:
    """Search order: ``private/.env`` near cwd & package, then ``.env`` up parent chains."""
    cwd = Path.cwd()
    here = Path(__file__).resolve()
    ordered: list[Path] = []
    for anchor in (cwd, here):
        for base in (anchor, *anchor.parents):
            ordered.append(base / "private" / ".env")
    for anchor in (cwd, here):
        for base in (anchor, *anchor.parents):
            ordered.append(base / ".env")
    return _dedupe(ordered)


def _load_env_files() -> None:
    """Prefer ``private/.env``, then nearest ``.env`` toward filesystem roots."""
    if not load_dotenv:
        return
    for candidate in _candidate_env_paths():
        if candidate.is_file():
            load_dotenv(candidate)
            return
    load_dotenv()


def load_config() -> DemoConfig:
    _load_env_files()

    manifold_raw = (_env("PRMP_MANIFOLD") or "pca").lower()
    manifold: ManifoldMethod = "umap" if manifold_raw == "umap" else "pca"

    r_raw = (_env("PRMP_R_SPEC_METHOD") or "svd_proxy").lower()
    r_spec_method: RSpecMethod = "edmd" if r_raw == "edmd" else "svd_proxy"

    return DemoConfig(
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        chat_model_a=_env("PRMP_CHAT_MODEL_A") or "google/gemini-3.5-flash",
        chat_model_b=_env("PRMP_CHAT_MODEL_B") or "meta-llama/llama-3.3-70b-instruct",
        embedding_model=_env("PRMP_EMBEDDING_MODEL") or "openai/text-embedding-3-small",
        manifold=manifold,
        umap_n_neighbors=_env_int("PRMP_UMAP_N_NEIGHBORS", 5),
        umap_min_dist=_env_float("PRMP_UMAP_MIN_DIST", 0.1),
        umap_metric=_env("PRMP_UMAP_METRIC") or "euclidean",
        umap_min_samples=_env_int("PRMP_UMAP_MIN_SAMPLES", 8),
        pca_k=_env_int("PRMP_PCA_K", 16),
        r_spec_method=r_spec_method,
        edmd_rank=_env_int("PRMP_EDMD_RANK", 8),
        dmd_window=_env_int("PRMP_DMD_WINDOW", 8),
        grassmann_w=_env_int("PRMP_GRASSMANN_W", 7),
        grassmann_r=_env_int("PRMP_GRASSMANN_R", 3),
        tda_window=_env_int("PRMP_TDA_WINDOW", 7),
        stability_window=_env_int("PRMP_STABILITY_WINDOW", 5),
        csd_variance_window=_env_int("PRMP_CSD_VARIANCE_WINDOW", 5),
        unity_gamma_norm_threshold=_env_float("PRMP_UNITY_GAMMA_NORM", 1.35),
        unity_w_p_norm_threshold=_env_float("PRMP_UNITY_W_P_NORM", 0.65),
        csd_variance_multiplier=_env_float("PRMP_CSD_VARIANCE_MULT", 2.0),
        max_steps=_env_int("PRMP_MAX_STEPS", 12),
        max_tokens=_env_int("PRMP_MAX_TOKENS", 512),
        temperature=_env_float("PRMP_TEMPERATURE", 0.4),
        h_template=_env("PRMP_H_TEMPLATE") or "{agent}:{action_id}:{rationale}",
        http_referer=_env("OPENROUTER_HTTP_REFERER"),
        http_title=_env("OPENROUTER_TITLE"),
    )


def config_snapshot(cfg: DemoConfig) -> dict:
    """Serializable snapshot for session_meta (omit secrets)."""
    return {
        "chat_model_a": cfg.chat_model_a,
        "chat_model_b": cfg.chat_model_b,
        "embedding_model": cfg.embedding_model,
        "manifold": cfg.manifold,
        "umap_n_neighbors": cfg.umap_n_neighbors,
        "umap_min_dist": cfg.umap_min_dist,
        "umap_metric": cfg.umap_metric,
        "umap_min_samples": cfg.umap_min_samples,
        "pca_k": cfg.pca_k,
        "r_spec_method": cfg.r_spec_method,
        "edmd_rank": cfg.edmd_rank,
        "dmd_window": cfg.dmd_window,
        "grassmann_w": cfg.grassmann_w,
        "grassmann_r": cfg.grassmann_r,
        "tda_window": cfg.tda_window,
        "stability_window": cfg.stability_window,
        "csd_variance_window": cfg.csd_variance_window,
        "unity_gamma_norm_threshold": cfg.unity_gamma_norm_threshold,
        "unity_w_p_norm_threshold": cfg.unity_w_p_norm_threshold,
        "csd_variance_multiplier": cfg.csd_variance_multiplier,
        "max_steps": cfg.max_steps,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "h_template": cfg.h_template,
    }
