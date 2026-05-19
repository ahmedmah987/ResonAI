"""Streamlit replay dashboard for saved JSONL sessions (offline-first)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# Ensure src/ is in path if running directly
APP_ROOT = Path(__file__).resolve().parents[1]
SRC = APP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prmp_demo.config import load_config
from prmp_demo.run_session import run_session_gen
from dataclasses import replace

# Popular OpenRouter models for the dropdown
CHAT_MODELS = [
    "google/gemini-3.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-chat",
    "mistralai/mistral-7b-instruct",
    "qwen/qwen-2.5-72b-instruct",
]

EMBED_MODELS = [
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "google/text-embedding-004",
    "cohere/embed-english-v3.0",
]

# App lives under app; bundled samples live under samples
SAMPLES = APP_ROOT / "samples"


def load_session_bytes(data: bytes) -> tuple[dict, list[dict]]:
    lines = data.decode("utf-8").splitlines()
    if not lines:
        raise ValueError("empty file")
    meta = json.loads(lines[0])
    steps = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        steps.append(json.loads(line))
    return meta, steps


def load_session_path(path: Path) -> tuple[dict, list[dict]]:
    return load_session_bytes(path.read_bytes())


def _metric(step: dict, key: str, default=None):
    m = step.get("metrics") or {}
    return m.get(key, default)


def main() -> None:
    st.set_page_config(page_title="P-RMP Resonance Monitor", layout="wide", initial_sidebar_state="expanded")
    
    # Custom "Scientific Monitor" Styling
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
            color: #e0e0e0;
        }
        .stMetric {
            background-color: #161b22;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #30363d;
        }
        .resonance-header {
            font-family: 'Courier New', Courier, monospace;
            color: #58a6ff;
            text-align: center;
            border-bottom: 2px solid #30363d;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("<h1 class='resonance-header'>P-RMP RESONANCE MONITOR v0.1.0</h1>", unsafe_allow_html=True)
    
    st.caption(
        "Monitoring Topological Resonance Γ(t) and Dynamic Coherence between autonomous agents. "
        "Observer: Shared Latent Space Z (PCA/UMAP)."
    )

    sidebar = st.sidebar
    sample_files = sorted(SAMPLES.glob("demo_*.jsonl")) if SAMPLES.exists() else []

    choice = sidebar.radio(
        "Source",
        ["Bundled sample", "Upload JSONL", "Live Session (OpenRouter)"],
        horizontal=True,
    )

    meta: dict = {}
    steps: list[dict] = []

    if choice == "Bundled sample":
        if not sample_files:
            st.warning(
                f"No demo JSONL files found under `{SAMPLES}`. "
                "Run `python scripts/generate_fixture_sessions.py` with cwd inside `github/`, "
                "or `python github/scripts/generate_fixture_sessions.py` from the workspace folder that contains `github/`."
            )
            return
        picked = sidebar.selectbox("Sample file", sample_files, format_func=lambda p: p.name)
        meta, steps = load_session_path(picked)
    elif choice == "Upload JSONL":
        up = st.file_uploader("JSONL session", type=["jsonl"])
        if up is None:
            st.info("Upload a session exported by `python -m prmp_demo.run_session`.")
            return
        meta, steps = load_session_bytes(up.read())
    else:
        # Live Session
        st.subheader("Live Resonance Session")
        cfg = load_config()

        with st.sidebar:
            st.divider()
            api_key = st.text_input("OpenRouter API Key", value=cfg.openrouter_api_key or "", type="password")
            
            st.markdown("**Model Selection**")
            model_a = st.selectbox("Agent A Model", CHAT_MODELS, index=CHAT_MODELS.index(cfg.chat_model_a) if cfg.chat_model_a in CHAT_MODELS else 0)
            model_b = st.selectbox("Agent B Model", CHAT_MODELS, index=CHAT_MODELS.index(cfg.chat_model_b) if cfg.chat_model_b in CHAT_MODELS else 1)
            embed_m = st.selectbox("Embedding Model (Observer)", EMBED_MODELS, index=EMBED_MODELS.index(cfg.embedding_model) if cfg.embedding_model in EMBED_MODELS else 0)
            
            st.divider()
            scenario = st.selectbox("Scenario", ["silent_resonance", "conflicting_objectives", "trivial_correlation"])
            max_steps = st.number_input("Max Steps", min_value=1, max_value=50, value=cfg.max_steps or 10)
            dry_run = st.checkbox("Dry Run (Simulation)", value=not bool(api_key))
            start_btn = st.button("🚀 Start Live Session")

        if not start_btn and not st.session_state.get("live_steps"):
            st.info("Configure session in the sidebar and click Start.")
            return

        if start_btn:
            st.session_state.live_steps = []
            st.session_state.live_meta = {
                "scenario": scenario, 
                "status": "running",
                "chat_model_a": model_a,
                "chat_model_b": model_b,
                "embedding_model": embed_m
            }
            
            # Update config with UI values
            cfg = replace(cfg, 
                          openrouter_api_key=api_key,
                          chat_model_a=model_a,
                          chat_model_b=model_b,
                          embedding_model=embed_m)

            # Dashboard Placeholders
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            met_r = m_col1.empty()
            met_w = m_col2.empty()
            met_g = m_col3.empty()
            met_rho = m_col4.empty()

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            chart_tab1, chart_tab2 = st.tabs(["📈 Coherence Evolution", "🌌 Latent Trajectory"])
            with chart_tab1:
                chart_placeholder = st.empty()
            with chart_tab2:
                trajectory_placeholder = st.empty()
            
            log_container = st.container()
            
            try:
                for step in run_session_gen(
                    scenario=scenario,
                    cfg=cfg,
                    dry_parse_only=dry_run,
                    max_steps_override=int(max_steps)
                ):
                    st.session_state.live_steps.append(step)
                    
                    # Update progress
                    curr_t = step["t"] + 1
                    progress = curr_t / max_steps
                    progress_bar.progress(progress)
                    status_text.text(f"SYSTEM STATUS: Step {curr_t}/{max_steps} processed.")

                    # Metrics Update
                    m = step["metrics"]
                    g = m.get("gamma", [0,0,0])
                    prev_g = st.session_state.live_steps[-2]["metrics"].get("gamma", [0,0,0]) if len(st.session_state.live_steps) > 1 else [0,0,0]
                    
                    met_r.metric("R_spec (Spectral)", f"{g[0]:.2f}", f"{g[0]-prev_g[0]:+.2f}")
                    met_w.metric("W_p (Topological)", f"{g[1]:.2f}", f"{g[1]-prev_g[1]:+.2f}")
                    met_g.metric("Grassmann (Subspace)", f"{g[2]:.2f}", f"{g[2]-prev_g[2]:+.2f}")
                    met_rho.metric("ρ_text (Surface)", f"{m.get('rho_text', 0):.2f}")

                    # Real-time chart update
                    live_df = pd.DataFrame([
                        {
                            "t": s["t"],
                            "rho_text": _metric(s, "rho_text"),
                            "gamma_0": (_metric(s, "gamma") or [0,0,0])[0],
                            "gamma_1": (_metric(s, "gamma") or [0,0,0])[1],
                            "gamma_2": (_metric(s, "gamma") or [0,0,0])[2],
                        }
                        for s in st.session_state.live_steps
                    ])
                    chart_placeholder.line_chart(live_df.set_index("t"))

                    # Trajectory Plot (2D PCA)
                    if "latent_z_a" in m:
                        za = np.array(m["latent_z_a"])
                        zb = np.array(m["latent_z_b"])
                        traj_df = pd.DataFrame({
                            "Z1": np.concatenate([za[:, 0], zb[:, 0]]),
                            "Z2": np.concatenate([za[:, 1], zb[:, 1]]),
                            "Agent": ["Agent A"] * len(za) + ["Agent B"] * len(zb),
                            "Step": list(range(len(za))) * 2
                        })
                        trajectory_placeholder.scatter_chart(traj_df, x="Z1", y="Z2", color="Agent", size="Step")

                    # Live Chat Log
                    with log_container:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.write(f"**Step {step['t']}** | Agent A: `{step['parsed_action_a']['action_id']}`")
                            st.caption(step['parsed_action_a']['rationale'][:200] + "...")
                        with st.chat_message("user", avatar="👤"):
                            st.write(f"**Step {step['t']}** | Agent B: `{step['parsed_action_b']['action_id']}`")
                            st.caption(step['parsed_action_b']['rationale'][:200] + "...")
                
                st.success("MONITORING COMPLETE: Resonance session finalized.")
                st.session_state.live_meta["status"] = "completed"
            except Exception as e:
                st.error(f"CRITICAL ERROR: {e}")
                st.session_state.live_meta["status"] = "failed"

        steps = st.session_state.get("live_steps", [])
        meta = st.session_state.get("live_meta", {})

    with st.expander("session_meta", expanded=False):
        st.json(meta)

    if meta.get("analysis_summary"):
        with st.expander("analysis_summary (offline analyze_session)", expanded=False):
            st.json(meta["analysis_summary"])

    if not steps:
        st.error("No step rows found.")
        return

    # Dashboard for Replay
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    last_m = steps[-1]["metrics"]
    g = last_m.get("gamma", [0,0,0])
    prev_g = steps[-2]["metrics"].get("gamma", [0,0,0]) if len(steps) > 1 else [0,0,0]
    
    m_col1.metric("R_spec (Spectral)", f"{g[0]:.2f}", f"{g[0]-prev_g[0]:+.2f}")
    m_col2.metric("W_p (Topological)", f"{g[1]:.2f}", f"{g[1]-prev_g[1]:+.2f}")
    m_col3.metric("Grassmann (Subspace)", f"{g[2]:.2f}", f"{g[2]-prev_g[2]:+.2f}")
    m_col4.metric("ρ_text (Surface)", f"{last_m.get('rho_text', 0):.2f}")

    tab_evol, tab_traj, tab_data = st.tabs(["📈 Coherence Evolution", "🌌 Latent Trajectory", "📑 Raw Data"])
    
    with tab_evol:
        df = pd.DataFrame(
            [
                {
                    "t": s["t"],
                    "rho_text": _metric(s, "rho_text"),
                    "gamma_0_R_spec": (_metric(s, "gamma") or [None, None, None])[0],
                    "gamma_1_W_p": (_metric(s, "gamma") or [None, None, None])[1],
                    "gamma_2_Grassmann": (_metric(s, "gamma") or [None, None, None])[2],
                    "lambda_gamma": _metric(s, "lambda_gamma"),
                    "kappa_gamma": _metric(s, "kappa_gamma"),
                    "delta_rg": _metric(s, "delta_rg"),
                    "csd_variance": _metric(s, "csd_variance"),
                    "csd_spike": _metric(s, "csd_spike"),
                    "unity_proxy": _metric(s, "unity_proxy"),
                    "eta_topo": _metric(s, "eta_topo"),
                    "phenomenological_state": _metric(s, "phenomenological_state"),
                    "parse_ok_a": s.get("parse_ok_a"),
                    "parse_ok_b": s.get("parse_ok_b"),
                }
                for s in steps
            ]
        )
        st.subheader("Γ components & ρ_text over time")
        st.line_chart(df.set_index("t")[["gamma_0_R_spec", "gamma_1_W_p", "gamma_2_Grassmann", "rho_text"]])

        sec_cols = [c for c in ("lambda_gamma", "kappa_gamma", "delta_rg", "csd_variance") if df[c].notna().any()]
        if sec_cols:
            st.subheader("Stability / falsification proxies (after analyze_session)")
            st.line_chart(df.set_index("t")[sec_cols])

    with tab_traj:
        st.subheader("Latent Space Trajectory (2D Projection)")
        if "latent_z_a" in last_m:
            za = np.array(last_m["latent_z_a"])
            zb = np.array(last_m["latent_z_b"])
            traj_df = pd.DataFrame({
                "Z1": np.concatenate([za[:, 0], zb[:, 0]]),
                "Z2": np.concatenate([za[:, 1], zb[:, 1]]),
                "Agent": ["Agent A"] * len(za) + ["Agent B"] * len(zb),
                "Step": list(range(len(za))) * 2
            })
            st.scatter_chart(traj_df, x="Z1", y="Z2", color="Agent", size="Step")
        else:
            st.info("Latent coordinates not found in this session file. Run a new live session or re-analyze to generate them.")

    with tab_data:
        st.dataframe(df, use_container_width=True)

    st.divider()
    t_max = len(steps) - 1
    t_sel = st.slider("Step t", min_value=0, max_value=t_max, value=t_max)

    row = steps[t_sel]
    pheno = _metric(row, "phenomenological_state")
    if pheno:
        st.info(f"Phenomenological state (heuristic): **{pheno}**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Agent A**")
        st.text(row.get("raw_text_a", "")[:8000])
        st.json(row.get("parsed_action_a"))
    with c2:
        st.markdown("**Agent B**")
        st.text(row.get("raw_text_b", "")[:8000])
        st.json(row.get("parsed_action_b"))

    st.markdown("#### Embedding-channel strings `h(·)`")
    em1, em2 = st.columns(2)
    with em1:
        st.text(row.get("text_embed_a", ""))
    with em2:
        st.text(row.get("text_embed_b", ""))

    st.markdown("#### Metrics at selected step")
    st.json(row.get("metrics") or {})

    st.markdown("#### All steps (scalar metrics)")
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
