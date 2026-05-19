"""HTTP client for OpenRouter chat + embeddings with retries."""

from __future__ import annotations

import random
import time
from typing import Any, Optional

import httpx

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, snippet: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.snippet = snippet


def _should_retry(status: int) -> bool:
    return status == 429 or status >= 500


def _backoff_sleep(attempt: int) -> None:
    delay = min(30.0, (2**attempt) * 0.5 + random.random() * 0.25)
    time.sleep(delay)


def post_chat(
    api_key: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.4,
    timeout_s: float = 120.0,
    http_referer: Optional[str] = None,
    http_title: Optional[str] = None,
    max_retries: int = 4,
) -> tuple[dict[str, Any], int]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if http_title:
        headers["X-Title"] = http_title

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_status = -1
    last_snippet = ""
    with httpx.Client(timeout=timeout_s) as client:
        for attempt in range(max_retries):
            resp = client.post(f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=body)
            last_status = resp.status_code
            text = resp.text[:800]
            last_snippet = text
            if resp.status_code == 200:
                return resp.json(), resp.status_code
            if _should_retry(resp.status_code) and attempt < max_retries - 1:
                _backoff_sleep(attempt)
                continue
            raise OpenRouterError(
                f"chat completions failed: HTTP {resp.status_code}",
                status_code=resp.status_code,
                snippet=text,
            )
    raise OpenRouterError("chat completions exhausted retries", status_code=last_status, snippet=last_snippet)


def post_embeddings(
    api_key: str,
    *,
    model: str,
    input_text: str,
    timeout_s: float = 90.0,
    http_referer: Optional[str] = None,
    http_title: Optional[str] = None,
    max_retries: int = 4,
) -> tuple[dict[str, Any], int]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if http_title:
        headers["X-Title"] = http_title

    body = {"model": model, "input": input_text}

    last_status = -1
    last_snippet = ""
    with httpx.Client(timeout=timeout_s) as client:
        for attempt in range(max_retries):
            resp = client.post(f"{OPENROUTER_BASE}/embeddings", headers=headers, json=body)
            last_status = resp.status_code
            text = resp.text[:800]
            last_snippet = text
            if resp.status_code == 200:
                return resp.json(), resp.status_code
            if _should_retry(resp.status_code) and attempt < max_retries - 1:
                _backoff_sleep(attempt)
                continue
            raise OpenRouterError(
                f"embeddings failed: HTTP {resp.status_code}",
                status_code=resp.status_code,
                snippet=text,
            )
    raise OpenRouterError("embeddings exhausted retries", status_code=last_status, snippet=last_snippet)


def extract_chat_content(resp: dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


def extract_embedding_vector(resp: dict[str, Any]) -> list[float]:
    data = resp.get("data") or []
    if not data:
        return []
    emb = data[0].get("embedding")
    if emb is None:
        return []
    return [float(x) for x in emb]


def extract_usage(resp: dict[str, Any]) -> Optional[dict[str, Any]]:
    u = resp.get("usage")
    return u if isinstance(u, dict) else None
