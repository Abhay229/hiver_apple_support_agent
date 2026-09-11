"""
Unified LLM client so agent.py and judge.py don't care which provider is
behind them. Supports:

  - ANTHROPIC (paid): requires ANTHROPIC_API_KEY
  - GROQ (free, no credit card, OpenAI-compatible endpoint): requires
    GROQ_API_KEY, get one at https://console.groq.com/keys

Switch providers with the LLM_PROVIDER environment variable (default: groq,
since it's free -- flip to "anthropic" if you have a paid key and prefer it).

Free-tier note: Groq's free tier is ~14,400 requests/day and 30/minute,
which comfortably covers this project's ~600 total calls (golden set +
baselines + judge + human-agreement subset), so no cost either way.
"""

import os
import random
import time

PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "groq": "openai/gpt-oss-20b",
}

# Free-tier rate limits (esp. Groq's 30 req/min) are comfortably above what
# this project needs in total, but concurrent workers can still burst past
# the per-minute cap and get a 429. Retry with exponential backoff + jitter
# instead of letting that crash the whole eval run.
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "6"))
BASE_DELAY_SECONDS = float(os.environ.get("LLM_BASE_DELAY", "2.0"))


def get_model(env_var: str) -> str:
    return os.environ.get(env_var, DEFAULT_MODELS[PROVIDER])


def _is_rate_limit_or_transient(exc: Exception) -> bool:
    """Best-effort check across the anthropic and openai(groq) SDKs without
    hard-importing their exception classes (keeps this module import-light)."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "timeout" in name or "apierror" in name or "connectionerror" in name


def _anthropic_complete(system: str, user: str, model: str, max_tokens: int) -> str:
    from anthropic import Anthropic
    client = Anthropic()  # reads ANTHROPIC_API_KEY
    response = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _groq_complete(system: str, user: str, model: str, max_tokens: int) -> str:
    from openai import OpenAI  # groq is OpenAI-API-compatible
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return response.choices[0].message.content


def complete(system: str, user: str, model: str, max_tokens: int = 500) -> str:
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            if PROVIDER == "anthropic":
                return _anthropic_complete(system, user, model, max_tokens)
            elif PROVIDER == "groq":
                return _groq_complete(system, user, model, max_tokens)
            raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER} (use 'anthropic' or 'groq')")
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limit_or_transient(exc) or attempt == MAX_RETRIES:
                raise
            delay = BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 1.0)
            time.sleep(delay)
    raise last_exc  # pragma: no cover — unreachable, satisfies type checkers