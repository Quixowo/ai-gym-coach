"""Shared Anthropic async client (spec §7.4).

A single lazily-constructed :class:`anthropic.AsyncAnthropic` reused across the
classifier and the agent loop, so:

- there is one place to configure/monkeypatch the client (tests patch
  :func:`get_anthropic_client` at this boundary — CLAUDE.md rule 10, no live API in
  CI), and
- the client is built on first use, not at import time, so importing the app with
  ``ANTHROPIC_API_KEY`` unset/empty never fails (the key is only required when a call
  is actually made — which, in CI, it never is because the client is mocked).

The API key comes from ``settings.ANTHROPIC_API_KEY``; when empty, the SDK falls
back to its own env-var lookup, but no request is issued in tests so the empty key
is harmless.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from app.core.config import settings

_client: AsyncAnthropic | None = None


def get_anthropic_client() -> AsyncAnthropic:
    """Return the process-wide async Anthropic client, constructing it on first use."""
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY or None)
    return _client
