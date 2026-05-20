"""CLI: run one cooperative demo session and record JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from prmp_demo import agents
from prmp_demo.config import DemoConfig, config_snapshot, load_config
from prmp_demo.environment.task_protocol import (
    ScenarioId,
    append_history_turn,
    apply_actions_sequential,
    expected_action_conflicting,
    expected_action_for_step,
    initial_state,
    parse_discussion_output,
    parse_model_output,
)
from prmp_demo.decision import evaluate as evaluate_decision, Decision
from prmp_demo.gamma import compute_metrics_prefix
from prmp_demo.observer import apply_h
from prmp_demo.openrouter_client import (
    OpenRouterError,
    extract_chat_content,
    extract_embedding_vector,
    extract_usage,
    post_chat,
    post_embeddings,
)
from prmp_demo.json_utils import sanitize_for_json
from prmp_demo.session_schema import dump_jsonl_row


def _git_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for d in [here, *here.parents]:
        if (d / ".git").exists():
            return d
    return None


def _git_commit() -> str | None:
    root = _git_repo_root()
    if root is None:
        return None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=root,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def pseudo_embedding(text: str, dim: int = 64) -> list[float]:
    seed = abs(hash(text)) % (2**31)
    rng = np.random.RandomState(seed)
    return rng.standard_normal(dim).tolist()


def simulate_chat(cfg: DemoConfig, scenario: ScenarioId, agent_label: str, state: dict[str, Any], topic: Optional[str] = None) -> tuple[str, None]:
    import json as json_lib

    if scenario == "trivial_correlation":
        payload = {
            "action_id": "EXPLORE",
            "rationale": "DUPLICATE_ME",
        }
    elif scenario == "conflicting_objectives":
        exp = expected_action_conflicting(state.get("step_index", 0), agent_label)
        payload = {
            "action_id": exp,
            "rationale": f"Conflict-{agent_label}-step-{state.get('step_index')}-{seed_variant(agent_label, state)}",
        }
    elif scenario == "open_discussion":
        step = state.get("step_index", 0)
        if agent_label == "B" and state.get("history"):
            last = state["history"][-1]["rationale"][:120]
            msg = f"On your point about {last}… I see a link to decentralized synchronization via sheaf-based consensus."
        else:
            msg = (
                f"Step {step}: autonomous coordination may need topological invariants, not just pairwise messaging."
            )
        return msg, None
    else:
        exp = expected_action_for_step(scenario, state.get("step_index", 0))
        payload = {
            "action_id": exp or "EXPLORE",
            "rationale": f"Step-{state.get('step_index')}-{agent_label}-variant-{seed_variant(agent_label, state)}",
        }
    return json_lib.dumps(payload), None


def seed_variant(agent_label: str, state: dict[str, Any]) -> int:
    return (hash(agent_label) ^ hash(state.get("step_index", 0))) % 10_000


def run_live_chat(cfg: DemoConfig, scenario: ScenarioId, agent_label: str, model: str, state: dict[str, Any], topic: Optional[str] = None, redirect_hint: Optional[str] = None) -> tuple[str, dict[str, Any] | None]:
    topic = topic or state.get("topic")
    messages = agents.message_payload(agent_label, state, scenario, topic=topic, redirect_hint=redirect_hint)
    max_tokens = max(cfg.max_tokens, 768) if scenario == "open_discussion" else cfg.max_tokens
    temperature = min(cfg.temperature, 0.35) if scenario == "open_discussion" else cfg.temperature
    body, _status = post_chat(
        cfg.openrouter_api_key or "",
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        http_referer=cfg.http_referer,
        http_title=cfg.http_title,
    )
    content = extract_chat_content(body)
    usage = extract_usage(body)
    return content, usage


def run_live_embed(cfg: DemoConfig, text: str) -> tuple[list[float], dict[str, Any] | None]:
    body, _ = post_embeddings(
        cfg.openrouter_api_key or "",
        model=cfg.embedding_model,
        input_text=text,
        http_referer=cfg.http_referer,
        http_title=cfg.http_title,
    )
    vec = extract_embedding_vector(body)
    usage = extract_usage(body)
    return vec, usage


def build_meta(cfg: DemoConfig, scenario: ScenarioId, status: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "chat_model_a": cfg.chat_model_a,
        "chat_model_b": cfg.chat_model_b,
        "embedding_model": cfg.embedding_model,
        "git_commit": _git_commit(),
        "config_snapshot": config_snapshot(cfg),
        "session_status": status,
    }


def run_session_gen(
    *,
    scenario: ScenarioId,
    cfg: DemoConfig,
    dry_parse_only: bool,
    max_steps_override: int | None,
    topic: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
    max_steps = max_steps_override if max_steps_override is not None else cfg.max_steps

    state = initial_state(scenario)
    if topic:
        state["topic"] = topic
    hist_a: list[list[float]] = []
    hist_b: list[list[float]] = []
    prev_decision: dict[str, Any] | None = None

    for t in range(max_steps):
        errors_step: list[str] = []

        # Derive redirect hints from previous step's decision
        hint_a: str | None = None
        hint_b: str | None = None
        if prev_decision and scenario == "open_discussion":
            action: str = prev_decision.get("action", "CONTINUE")
            if action == "REDIRECT_A":
                hint_a = "Re-anchor your response to the other agent's core argument before adding new ideas."
            elif action == "REDIRECT_B":
                hint_b = "Re-anchor your response to the other agent's core argument before adding new ideas."
            elif action == "SOFT_RESET":
                hint_a = "Start a fresh angle on the main topic rather than continuing the current thread."
                hint_b = "Start a fresh angle on the main topic rather than continuing the current thread."

        def chat_fn(agent_label: str, model: str, current_state: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
            hint = hint_a if agent_label == "A" else hint_b
            if dry_parse_only:
                return simulate_chat(cfg, scenario, agent_label, current_state, topic=topic)
            return run_live_chat(cfg, scenario, agent_label, model, current_state, topic=topic, redirect_hint=hint)

        # Agent A turn
        parse_fn = parse_discussion_output if scenario == "open_discussion" else parse_model_output

        raw_a, usa = chat_fn("A", cfg.chat_model_a, state)
        pa, ok_a = parse_fn(raw_a)

        if scenario == "open_discussion":
            state_for_b = append_history_turn(state, "A", pa)
            raw_b, usb = chat_fn("B", cfg.chat_model_b, state_for_b)
        else:
            raw_b, usb = chat_fn("B", cfg.chat_model_b, state)

        pb, ok_b = parse_fn(raw_b)

        ta = apply_h(pa, "A", cfg)
        tb = apply_h(pb, "B", cfg)

        if dry_parse_only:
            emb_a = pseudo_embedding(ta)
            emb_b = pseudo_embedding(tb)
            uea = ueb = None
        else:
            emb_a, uea = run_live_embed(cfg, ta)
            emb_b, ueb = run_live_embed(cfg, tb)

        state_before = dict(state)
        state_after = apply_actions_sequential(state_before, pa, pb)

        hist_a.append(emb_a)
        hist_b.append(emb_b)

        metrics = compute_metrics_prefix(hist_a, hist_b, cfg=cfg)

        current_decision = evaluate_decision(metrics)
        step_record = {
            "t": t,
            "decision": current_decision,
            "redirect_applied": {"A": hint_a, "B": hint_b},
            "state_before": state_before,
            "state_after": state_after,
            "raw_text_a": raw_a,
            "raw_text_b": raw_b,
            "parsed_action_a": {"action_id": pa.action_id, "rationale": pa.rationale},
            "parsed_action_b": {"action_id": pb.action_id, "rationale": pb.rationale},
            "parse_ok_a": ok_a,
            "parse_ok_b": ok_b,
            "text_embed_a": ta,
            "text_embed_b": tb,
            "embedding_a": emb_a,
            "embedding_b": emb_b,
            "usage_chat_a": usa,
            "usage_chat_b": usb,
            "usage_embed_a": uea,
            "usage_embed_b": ueb,
            "errors": errors_step or None,
            "metrics": metrics,
        }
        yield step_record
        prev_decision = current_decision
        state = state_after


def run_session(
    *,
    scenario: ScenarioId,
    out_path: Path,
    cfg: DemoConfig,
    dry_parse_only: bool,
    max_steps_override: int | None,
) -> None:
    if not dry_parse_only and not cfg.openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY missing (use --dry-parse-only for offline simulation).")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = build_meta(cfg, scenario, "running")
    fout = out_path.open("w", encoding="utf-8")
    fout.write(dump_jsonl_row(meta))

    try:
        for step_record in run_session_gen(
            scenario=scenario,
            cfg=cfg,
            dry_parse_only=dry_parse_only,
            max_steps_override=max_steps_override,
        ):
            fout.write(json.dumps(sanitize_for_json(step_record), ensure_ascii=False) + "\n")
            fout.flush()

        meta_done = build_meta(cfg, scenario, "completed")
        fout.close()
        _rewrite_meta_first_line(out_path, meta_done)

    except Exception:
        if not fout.closed:
            fout.close()
        _rewrite_meta_first_line(out_path, build_meta(cfg, scenario, "failed"))
        raise

    except Exception:
        if not fout.closed:
            fout.close()
        raise


def _rewrite_meta_first_line(path: Path, meta: dict[str, Any]) -> None:
    body = path.read_text(encoding="utf-8").splitlines()
    if not body:
        path.write_text(json.dumps(sanitize_for_json(meta), ensure_ascii=False) + "\n", encoding="utf-8")
        return
    body[0] = json.dumps(sanitize_for_json(meta), ensure_ascii=False)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run P-RMP demo session via OpenRouter.")
    p.add_argument(
        "--scenario",
        choices=["trivial_correlation", "silent_resonance", "conflicting_objectives"],
        required=True,
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-parse-only", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config()
    run_session(
        scenario=args.scenario,
        out_path=args.out,
        cfg=cfg,
        dry_parse_only=args.dry_parse_only,
        max_steps_override=args.max_steps,
    )


if __name__ == "__main__":
    main()
