#!/usr/bin/env python3
"""
Discover OpenRouter models relevant to a P-RMP-style demo:

  - Chat / completion models: GET https://openrouter.ai/api/v1/models
  - Embedding models:        GET https://openrouter.ai/api/v1/embeddings/models

The listings endpoints work without an API key (as of OpenRouter's public API).

Optional smoke test for embeddings (requires OPENROUTER_API_KEY):

  set OPENROUTER_API_KEY=...   # Windows PowerShell: $env:OPENROUTER_API_KEY="..."
  python scripts/openrouter_discover_models.py --smoke-embedding --embedding-model "<id>"

Never paste your API key into source control or chat logs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


BASE = "https://openrouter.ai/api/v1"


def _fetch_json(url: str, bearer: str | None = None, timeout_s: float = 60.0) -> dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": "not-named-yet-openrouter-discover/1.0"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}: {detail[:500]}") from e
    return json.loads(body)


def _post_json(url: str, payload: dict[str, Any], bearer: str, timeout_s: float = 60.0) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "not-named-yet-openrouter-discover/1.0",
        "Authorization": f"Bearer {bearer}",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}: {detail[:800]}") from e
    return json.loads(body)


def _summarize_chat(models: list[dict[str, Any]], limit: int) -> None:
    print("\n=== Chat / completions models (/api/v1/models) ===\n")
    print(f"Total: {len(models)}")
    rows: list[tuple[str, str, str | int]] = []
    for m in models:
        arch = m.get("architecture") or {}
        modality = arch.get("modality") or ""
        ctx = m.get("context_length") or ""
        rows.append((str(m.get("id") or ""), str(modality), ctx))
    rows.sort(key=lambda r: r[0].lower())
    print(f"\nFirst {limit} (sorted by id):\n")
    for mid, modality, ctx in rows[:limit]:
        print(f"- {mid}")
        print(f"    modality: {modality}")
        print(f"    context_length: {ctx}")

    # Histogram of modalities (helps spot multimodal vs text-only quickly)
    from collections import Counter

    c = Counter(modality for _, modality, _ in rows if modality)
    print("\nTop modalities (count):")
    for k, v in c.most_common(12):
        print(f"  {v:5d}  {k}")


def _summarize_embeddings(models: list[dict[str, Any]], limit: int) -> None:
    print("\n=== Embedding models (/api/v1/embeddings/models) ===\n")
    print(f"Total: {len(models)}")
    rows = []
    for m in models:
        arch = m.get("architecture") or {}
        modality = arch.get("modality") or ""
        rows.append((str(m.get("id") or ""), str(m.get("name") or ""), modality))
    rows.sort(key=lambda r: r[0].lower())
    print(f"\nEmbedding models ({len(rows)}); showing up to {limit}:\n")
    for mid, name, modality in rows[: min(len(rows), limit)]:
        print(f"- {mid}")
        if name:
            print(f"    name: {name}")
        if modality:
            print(f"    modality: {modality}")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="List OpenRouter chat and embedding models (public listings).")
    p.add_argument("--chat-limit", type=int, default=25, help="How many chat models to print in detail.")
    p.add_argument("--embedding-limit", type=int, default=200, help="Max embedding models to print (usually small).")
    p.add_argument("--smoke-embedding", action="store_true", help="POST /embeddings once (needs OPENROUTER_API_KEY).")
    p.add_argument(
        "--embedding-model",
        default="",
        help="Model id for --smoke-embedding (defaults to first model from embeddings listing).",
    )
    args = p.parse_args(argv)

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    if args.smoke_embedding and not key:
        print("ERROR: --smoke-embedding requires OPENROUTER_API_KEY in the environment.", file=sys.stderr)
        return 2

    embed_payload = _fetch_json(f"{BASE}/embeddings/models")
    embed_models = list(embed_payload.get("data") or [])

    chat_payload = _fetch_json(f"{BASE}/models")
    chat_models = list(chat_payload.get("data") or [])

    _summarize_embeddings(embed_models, args.embedding_limit)
    _summarize_chat(chat_models, args.chat_limit)

    if args.smoke_embedding:
        model_id = args.embedding_model.strip()
        if not model_id:
            if not embed_models:
                print("ERROR: No embedding models returned; cannot pick a default.", file=sys.stderr)
                return 3
            model_id = str(embed_models[0].get("id") or "")
        print("\n=== Smoke test: POST /api/v1/embeddings ===\n")
        print(f"model: {model_id}")
        resp = _post_json(
            f"{BASE}/embeddings",
            {"model": model_id, "input": "hello"},
            bearer=key,
        )
        vec = (((resp.get("data") or [{}])[0]).get("embedding")) or []
        print(f"response keys: {sorted(resp.keys())}")
        print(f"embedding dims: {len(vec)}")
        if vec[:8]:
            print(f"first values: {vec[:8]}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
