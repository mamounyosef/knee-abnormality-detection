"""
llm_client.py — thin shared client for the Modal LLM endpoint.
=============================================================

Modal Auto Endpoints serve the OpenAI Chat Completions API under /v1 and are
authenticated with a proxy token pair passed as the `Modal-Key` and
`Modal-Secret` headers (the combined `Authorization: Bearer wk-....ws-...`
form is the alternative accepted by OpenAI-compatible clients).

Credentials live in the repo `.env`, which is gitignored.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

_cfg: dict = {}
_lock = threading.Lock()
stats = {"calls": 0, "retries": 0, "failures": 0, "rate_limited": 0,
         "in_tok": 0, "cached_tok": 0, "out_tok": 0, "reason_tok": 0}


def load_env(path: Path | None = None) -> dict:
    """Read .env once and cache the endpoint configuration."""
    global _cfg
    if _cfg:
        return _cfg
    path = path or ROOT / ".env"
    if not path.exists():
        sys.exit(f"missing {path} - needs MODAL_LLM_BASE_URL plus either "
                 f"MODAL_KEY/MODAL_SECRET or OLLAMA_API_KEY")
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    env.setdefault("MODAL_KEY", "")
    env.setdefault("MODAL_SECRET", "")
    _cfg = {
        "base": env["MODAL_LLM_BASE_URL"].rstrip("/"),
        "model": env.get("MODAL_LLM_MODEL", "Qwen/Qwen3.8-2.4T-A95B"),
        "headers": ({"Authorization": f"Bearer {env['OLLAMA_API_KEY']}",
                     "Content-Type": "application/json"}
                    if env.get("OLLAMA_API_KEY") else
                    {"Modal-Key": env["MODAL_KEY"],
                     "Modal-Secret": env["MODAL_SECRET"],
                     "Content-Type": "application/json"}),
        "sessions": int(env.get("MODAL_SESSIONS", "4")),
        # a local Ollama/vLLM endpoint: no Modal routing, no reasoning field
        "local": "localhost" in env["MODAL_LLM_BASE_URL"] or "127.0.0.1" in env["MODAL_LLM_BASE_URL"],
        "modal": "modal" in env["MODAL_LLM_BASE_URL"],
    }
    return _cfg


# Prompt caching: our system prompt is ~1,100 identical tokens on every call, and
# cached prompt tokens bill at $0.25/MTok instead of $2.00. But KV caches live in
# GPU RAM and are NOT shared between replicas, so a hit requires the request to
# land on the container that already holds the prefix. Modal routes by the
# `Modal-Session-ID` header: equal values go to the same container.
#
# Pinning everything to one session would serialise us onto a single replica, so
# we shard across a handful of sticky sessions — enough replicas for throughput,
# few enough that each one keeps the prefix warm.
_session_counter = 0


def _session_id() -> str:
    global _session_counter
    with _lock:
        _session_counter += 1
        return f"knee-labeler-{_session_counter % load_env()['sessions']}"


def chat(messages, *, schema=None, schema_name="response", effort="medium",
         max_tokens=4000, retries=8, timeout=900):
    """One chat completion. With `schema`, forces strict JSON-schema output.

    Retries on rate limits and 5xx with exponential backoff; raises on the last
    attempt so the caller can record the failure and continue.
    """
    cfg = load_env()
    payload = {"model": cfg["model"], "messages": messages, "temperature": 0.0,
               "max_tokens": max_tokens}
    if not cfg.get("local"):
        # Modal-only knob. Ollama rejects unknown top-level fields, and a local
        # endpoint has no reasoning-effort setting to give it anyway.
        payload["reasoning"] = {"effort": effort}
    if schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema}}

    headers = {**cfg["headers"], "Modal-Session-ID": _session_id()}
    for attempt in range(retries):
        try:
            r = requests.post(f"{cfg['base']}/chat/completions",
                              headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                # rate limited: nothing was generated, so this costs no tokens.
                # Wait what the server asks for, or back off generously — being
                # patient here is free, whereas giving up wastes the queue slot.
                wait = float(r.headers.get("Retry-After", 0)) or min(10 * (attempt + 1), 60)
                with _lock:
                    stats["rate_limited"] += 1
                time.sleep(wait)
                raise requests.HTTPError("status 429 (rate limited)")
            # 499 = client closed request: the proxy saw us hang up mid-generation,
            # usually because too many concurrent requests were queued behind a busy
            # replica. Retryable, and a signal to lower --workers.
            if r.status_code in (408, 499, 500, 502, 503, 504):
                raise requests.HTTPError(f"status {r.status_code}: {r.text[:120]}")
            r.raise_for_status()
            data = r.json()
            with _lock:
                stats["calls"] += 1
                usage = data.get("usage") or {}
                stats["in_tok"] += usage.get("prompt_tokens", 0)
                stats["out_tok"] += usage.get("completion_tokens", 0)
                # cache hits are billed at $0.25/MTok instead of $2.00 — track
                # them so we can see whether sticky routing is actually working
                details = usage.get("prompt_tokens_details") or {}
                stats["cached_tok"] += details.get("cached_tokens", 0)
                cdet = usage.get("completion_tokens_details") or {}
                stats["reason_tok"] += cdet.get("reasoning_tokens", 0)
            return data["choices"][0]["message"]["content"]
        except Exception:                                  # noqa: BLE001 — retry anything
            with _lock:
                stats["retries"] += 1
            if attempt == retries - 1:
                with _lock:
                    stats["failures"] += 1
                raise
            time.sleep(min(2 ** attempt * 3, 60))
    return None


def _extract_json(txt: str):
    """Parse a JSON object out of a reply that may not be pure JSON.

    Modal enforces the schema with constrained decoding, so the reply IS the object.
    Ollama Cloud does not - it treats `format`/`response_format` as a hint, and models
    wrap the object in ```json fences or add a sentence first. So: try strict, then
    strip fences, then take the outermost {...}.
    """
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    t = txt.strip()
    if "```" in t:                                   # ```json ... ``` fences
        parts = t.split("```")
        for chunk in parts[1:]:
            chunk = chunk.split(chr(10), 1)[-1] if chunk[:4].lower().startswith("json") else chunk
            try:
                return json.loads(chunk.strip())
            except json.JSONDecodeError:
                continue
    i, j = t.find("{"), t.rfind("}")                 # prose around a bare object
    if i >= 0 and j > i:
        return json.loads(t[i:j + 1])
    raise json.JSONDecodeError("no JSON object in reply", txt[:200], 0)


def chat_json(messages, **kw):
    """chat() plus a tolerant parse of the response."""
    return _extract_json(chat(messages, **kw))


def warmup(timeout: int = 1800) -> bool:
    """Block until the endpoint has a container ready.

    This is a 2.4-trillion-parameter model. When the endpoint has scaled to zero,
    the first request has to boot a container and load the weights, which takes
    minutes. A short client timeout cancels that boot, the container spins back
    down, and the next request starts the same cold start again - an endless loop
    that looks like the endpoint is broken.

    So: send one trivial request and wait a long time for it. Once it returns,
    the container is warm and normal short timeouts are safe.
    """
    cfg = load_env()
    if not cfg.get("modal"):
        # Only Modal Auto Endpoints scale to zero. Ollama (local or cloud) is either
        # up or it is not, so waiting 30 minutes for a boot that never happens is
        # just a hang.
        print("[warmup] non-Modal endpoint - skipping the cold-start wait")
        return True
    t0 = time.time()
    print(f"[warmup] waiting for a container (up to {timeout // 60} min) ...", flush=True)
    body = {"model": cfg["model"],
            "messages": [{"role": "user", "content": "Reply with the word OK."}],
            "temperature": 0.0, "max_tokens": 10}
    attempt = 0
    while time.time() - t0 < timeout:
        attempt += 1
        try:
            r = requests.post(
                f"{cfg['base']}/chat/completions",
                headers={**cfg["headers"], "Modal-Session-ID": "warmup"},
                json=body, timeout=min(600, timeout))
            # A scaled-to-zero endpoint answers 503 immediately instead of
            # queueing, so 503/502/504 mean "still booting", not "broken".
            if r.status_code in (502, 503, 504):
                if attempt == 1 or attempt % 5 == 0:
                    print(f"[warmup] {r.status_code} — container still starting "
                          f"({time.time() - t0:.0f}s)", flush=True)
                time.sleep(10)
                continue
            r.raise_for_status()
            print(f"[warmup] ready after {time.time() - t0:.0f}s "
                  f"({attempt} attempt(s))", flush=True)
            return True
        except requests.Timeout:
            continue                                     # boot is slow; keep waiting
        except Exception as exc:                          # noqa: BLE001
            print(f"[warmup] {type(exc).__name__}: {str(exc)[:90]} — retrying", flush=True)
            time.sleep(10)
    print(f"[warmup] FAILED after {time.time() - t0:.0f}s", flush=True)
    return False
