#!/usr/bin/env python3
"""Generate synthetic JSONL fixtures under samples/ for offline Streamlit replay."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from dataclasses import replace

from prmp_demo.config import DemoConfig, config_snapshot, load_config
from prmp_demo.environment.task_protocol import (
    ANTI_ACTION,
    ParsedAction,
    SILENT_SEQUENCE,
    apply_actions_sequential,
    initial_state,
)
from prmp_demo.gamma import compute_metrics_prefix
from prmp_demo.json_utils import sanitize_for_json
from prmp_demo.observer import apply_h


def build_meta(scenario: str, cfg: DemoConfig) -> dict:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "chat_model_a": cfg.chat_model_a,
        "chat_model_b": cfg.chat_model_b,
        "embedding_model": cfg.embedding_model,
        "git_commit": None,
        "config_snapshot": config_snapshot(cfg),
        "session_status": "completed",
        "fixture_note": "synthetic embeddings for UI replay only",
    }


def write_session(path: Path, scenario: str, cfg: DemoConfig, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(sanitize_for_json(build_meta(scenario, cfg)), ensure_ascii=False)]
    for rec in records:
        lines.append(json.dumps(sanitize_for_json(rec), ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = replace(
        load_config(),
        openrouter_api_key=None,
        chat_model_a="fixture/model-a",
        chat_model_b="fixture/model-b",
        embedding_model="fixture/model-e",
        pca_k=8,
        dmd_window=7,
        grassmann_w=6,
        grassmann_r=3,
        tda_window=6,
        max_steps=12,
        max_tokens=256,
        temperature=0.4,
        http_referer=None,
        http_title=None,
    )

    samples = ROOT / "samples"

    # --- Trivial: identical embeddings each step ---
    trivial_records = []
    state = initial_state("trivial_correlation")
    hist_a: list[list[float]] = []
    hist_b: list[list[float]] = []
    rng = np.random.RandomState(42)
    base = rng.standard_normal(24)
    for t in range(10):
        pa = ParsedAction("EXPLORE", "DUPLICATE_ME")
        pb = ParsedAction("EXPLORE", "DUPLICATE_ME")
        ta = apply_h(pa, "A", cfg)
        tb = apply_h(pb, "B", cfg)
        emb = base.tolist()
        hist_a.append(emb)
        hist_b.append(list(emb))
        sb = dict(state)
        sa = apply_actions_sequential(sb, pa, pb)
        metrics = compute_metrics_prefix(hist_a, hist_b, cfg=cfg)
        trivial_records.append(
            {
                "t": t,
                "state_before": sb,
                "state_after": sa,
                "raw_text_a": "{}",
                "raw_text_b": "{}",
                "parsed_action_a": {"action_id": pa.action_id, "rationale": pa.rationale},
                "parsed_action_b": {"action_id": pb.action_id, "rationale": pb.rationale},
                "parse_ok_a": True,
                "parse_ok_b": True,
                "text_embed_a": ta,
                "text_embed_b": tb,
                "embedding_a": emb,
                "embedding_b": list(emb),
                "usage_chat_a": None,
                "usage_chat_b": None,
                "usage_embed_a": None,
                "usage_embed_b": None,
                "errors": None,
                "metrics": metrics,
            }
        )
        state = sa

    write_session(samples / "demo_trivial_correlation.jsonl", "trivial_correlation", cfg, trivial_records)

    # --- Silent: correlated latent direction + orthogonal noise ---
    silent_records = []
    state = initial_state("silent_resonance")
    hist_a = []
    hist_b = []
    rng = np.random.RandomState(7)
    u = rng.standard_normal(24)
    u /= np.linalg.norm(u) + 1e-9
    for t in range(10):
        pa = ParsedAction("EXPLORE", f"Alpha wording variant {t}")
        pb = ParsedAction("EXPLORE", f"Beta wording divergent {t} xyz")
        ta = apply_h(pa, "A", cfg)
        tb = apply_h(pb, "B", cfg)
        coeff = t * 0.15
        noise_a = rng.standard_normal(24) * 0.2
        noise_b = rng.standard_normal(24) * 0.2
        va = coeff * u + noise_a
        vb = coeff * u + noise_b
        hist_a.append(va.tolist())
        hist_b.append(vb.tolist())
        sb = dict(state)
        sa = apply_actions_sequential(sb, pa, pb)
        metrics = compute_metrics_prefix(hist_a, hist_b, cfg=cfg)
        silent_records.append(
            {
                "t": t,
                "state_before": sb,
                "state_after": sa,
                "raw_text_a": "{}",
                "raw_text_b": "{}",
                "parsed_action_a": {"action_id": pa.action_id, "rationale": pa.rationale},
                "parsed_action_b": {"action_id": pb.action_id, "rationale": pb.rationale},
                "parse_ok_a": True,
                "parse_ok_b": True,
                "text_embed_a": ta,
                "text_embed_b": tb,
                "embedding_a": va.tolist(),
                "embedding_b": vb.tolist(),
                "usage_chat_a": None,
                "usage_chat_b": None,
                "usage_embed_a": None,
                "usage_embed_b": None,
                "errors": None,
                "metrics": metrics,
            }
        )
        state = sa

    write_session(samples / "demo_silent_resonance.jsonl", "silent_resonance", cfg, silent_records)

    # --- Conflicting objectives: opposing mandated actions + anti-correlated embeddings ---
    conflict_records = []
    state = initial_state("conflicting_objectives")
    hist_a = []
    hist_b = []
    rng = np.random.RandomState(99)
    u = rng.standard_normal(24)
    u /= np.linalg.norm(u) + 1e-9
    for t in range(10):
        idx = min(t, len(SILENT_SEQUENCE) - 1)
        ea = SILENT_SEQUENCE[idx]
        eb = ANTI_ACTION[ea]
        pa = ParsedAction(ea, f"A-aligned-{ea}-{t}")
        pb = ParsedAction(eb, f"B-opposing-{eb}-{t}")
        ta = apply_h(pa, "A", cfg)
        tb = apply_h(pb, "B", cfg)
        coeff = t * 0.12
        va = coeff * u + rng.standard_normal(24) * 0.25
        vb = -va + rng.standard_normal(24) * 0.55
        hist_a.append(va.tolist())
        hist_b.append(vb.tolist())
        sb = dict(state)
        sa = apply_actions_sequential(sb, pa, pb)
        metrics = compute_metrics_prefix(hist_a, hist_b, cfg=cfg)
        conflict_records.append(
            {
                "t": t,
                "state_before": sb,
                "state_after": sa,
                "raw_text_a": "{}",
                "raw_text_b": "{}",
                "parsed_action_a": {"action_id": pa.action_id, "rationale": pa.rationale},
                "parsed_action_b": {"action_id": pb.action_id, "rationale": pb.rationale},
                "parse_ok_a": True,
                "parse_ok_b": True,
                "text_embed_a": ta,
                "text_embed_b": tb,
                "embedding_a": va.tolist(),
                "embedding_b": vb.tolist(),
                "usage_chat_a": None,
                "usage_chat_b": None,
                "usage_embed_a": None,
                "usage_embed_b": None,
                "errors": None,
                "metrics": metrics,
            }
        )
        state = sa

    write_session(samples / "demo_conflicting_objectives.jsonl", "conflicting_objectives", cfg, conflict_records)

    print(f"Wrote {samples / 'demo_trivial_correlation.jsonl'}")
    print(f"Wrote {samples / 'demo_silent_resonance.jsonl'}")
    print(f"Wrote {samples / 'demo_conflicting_objectives.jsonl'}")


if __name__ == "__main__":
    main()
