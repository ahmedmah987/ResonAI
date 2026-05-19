"""CLI: replay metrics computation without calling LLM APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from prmp_demo.config import DemoConfig, load_config
from prmp_demo.falsification_metrics import (
    augment_step_falsification,
    attach_w_p_prev_chain,
    compute_session_analysis_summary,
)
from prmp_demo.gamma import compute_metrics_prefix
from prmp_demo.json_utils import sanitize_for_json
from prmp_demo.pipeline.stability import stability_bundle_at_t
from prmp_demo.session_schema import validate_session_meta, validate_step_record


def analyze(path_in: Path, path_out: Path, cfg: DemoConfig) -> None:
    lines = path_in.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit("empty input")

    meta = json.loads(lines[0])
    validate_session_meta(meta)
    meta.setdefault("analysis_note", "metrics recomputed offline")

    steps: list[dict[str, Any]] = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        validate_step_record(rec)
        steps.append(rec)

    embed_a: list[list[float]] = []
    embed_b: list[list[float]] = []

    metrics_steps: list[dict[str, Any]] = []

    for rec in steps:
        embed_a.append(rec["embedding_a"])
        embed_b.append(rec["embedding_b"])
        metrics = compute_metrics_prefix(embed_a, embed_b, cfg=cfg)
        metrics_steps.append(metrics)

    gamma_mat = np.asarray([m["gamma"] for m in metrics_steps], dtype=float)

    for t, metrics in enumerate(metrics_steps):
        gh = gamma_mat[: t + 1]
        stab = stability_bundle_at_t(
            gh,
            t,
            stability_window=cfg.stability_window,
            csd_variance_window=cfg.csd_variance_window,
            csd_multiplier=cfg.csd_variance_multiplier,
            unity_gamma_norm_threshold=cfg.unity_gamma_norm_threshold,
            unity_w_p_norm_threshold=cfg.unity_w_p_norm_threshold,
            w_p_norm_at_t=float(metrics.get("w_p_norm", 0.0)),
        )
        metrics.update(stab)

    attach_w_p_prev_chain(metrics_steps)

    for t, metrics in enumerate(metrics_steps):
        gamma_prev = gamma_mat[t - 1].tolist() if t > 0 else None
        start = max(0, t - 4)
        gh_recent = gamma_mat[start : t + 1]
        augment_step_falsification(metrics, gamma_prev=gamma_prev, gamma_hist_recent=gh_recent)
        metrics.pop("_prev_w_p_raw", None)

    summary = compute_session_analysis_summary(metrics_steps, cfg)
    meta["analysis_summary"] = summary

    out_lines = [json.dumps(sanitize_for_json(meta), ensure_ascii=False)]
    for rec, metrics in zip(steps, metrics_steps, strict=True):
        rec["metrics"] = metrics
        out_lines.append(json.dumps(sanitize_for_json(rec), ensure_ascii=False))

    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Recompute Γ / ρ metrics from saved JSONL.")
    p.add_argument("--in", dest="inp", required=True, help="Input JSONL session path")
    p.add_argument("--out", dest="out", required=True, help="Output JSONL path")
    args = p.parse_args(argv)

    cfg = load_config()
    analyze(Path(args.inp), Path(args.out), cfg)


if __name__ == "__main__":
    main()
