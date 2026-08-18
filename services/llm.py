"""LLM client: providers with fallback and a disk cache.

Provider chain (all speak the OpenAI API, so one client library covers
them). Providers whose key is not set are skipped, so which ones answer
is controlled entirely from the environment:

1. Gemini Flash — primary free tier (``GEMINI_API_KEY``)
2. Groq — free-tier fallback when Gemini errors or rate-limits (``GROQ_API_KEY``)
3. OpenAI — paid, used when it is the only key set (``OPENAI_API_KEY``)

Every successful response is cached on disk, keyed by a hash of the request
(messages + temperature + any explicit model). Re-running an evaluation that
asks the same questions costs zero API calls and returns in milliseconds —
which is what makes iterating on LLM-as-judge experiments practical on free
tiers. Delete ``data/llm_cache/`` to force fresh calls.

Note: the cache key is the *request*; the cached file records which provider
actually answered. If Gemini was down and Groq answered, that answer is
reused until the cache entry is removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from openai import OpenAI

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "llm_cache"

PROVIDERS = [
    {
        "name": "gemini",
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-3.6-flash",
    },
    {
        "name": "groq",
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",
    },
    {
        "name": "openai",
        "key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
    },
]


def chat(
    messages: list[dict] | str,
    temperature: float = 0.0,
    model: str | None = None,
    use_cache: bool = True,
) -> str:
    """Send a chat request, trying providers in order. Returns the text.

    ``messages`` may be a plain string (treated as a single user message)
    or a full OpenAI-style message list. ``model`` pins a specific model on
    whichever provider serves it; leave unset to use each provider's default.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    cache_path = _cache_path(messages, temperature, model)
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())["text"]

    errors = []
    for provider in PROVIDERS:
        api_key = os.getenv(provider["key_env"])
        if not api_key:
            continue
        chosen_model = model or os.getenv(provider["model_env"], provider["default_model"])
        try:
            client = OpenAI(api_key=api_key, base_url=provider["base_url"])
            response = _create_with_backoff(
                client, chosen_model, messages, temperature
            )
            text = response.choices[0].message.content or ""
            if use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        {
                            "provider": provider["name"],
                            "model": chosen_model,
                            "text": text,
                            "usage": response.usage.model_dump() if response.usage else None,
                        },
                        indent=2,
                    )
                )
            return text
        except Exception as error:  # noqa: BLE001 — any failure moves to the next provider
            errors.append(f"{provider['name']}: {error}")

    raise RuntimeError(
        "All LLM providers failed or no API keys are set. "
        f"Tried: {errors or 'none (set GEMINI_API_KEY and/or GROQ_API_KEY)'}"
    )


def _create_with_backoff(
    client: OpenAI, model: str, messages: list[dict], temperature: float
):
    """Retry rate-limited requests (429) with exponential backoff instead of
    failing over to the next provider — a per-minute token limit is
    transient, and eval sweeps hit it routinely."""
    delay = 4.0
    for attempt in range(6):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
        except Exception as error:  # noqa: BLE001 — re-raised unless a 429
            if "429" not in str(error) or attempt == 5:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


def _cache_path(messages: list[dict], temperature: float, model: str | None) -> Path:
    key = hashlib.sha256(
        json.dumps(
            {"messages": messages, "temperature": temperature, "model": model},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return CACHE_DIR / f"{key}.json"


if __name__ == "__main__":
    print(chat("Reply with exactly: ok", use_cache=False))
