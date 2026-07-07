"""Shared Voyage async client (spec §9.2).

The mirror image of :mod:`app.llm.client` for the retrieval path: a single
lazily-constructed :class:`voyageai.AsyncClient`, reused by
``knowledge_service.search_knowledge_base`` to embed the incoming query. The async
client is aiohttp-based, so ``await vo.embed(...)`` never blocks the event loop and
stalls the concurrent SSE streams (the sync ``voyageai.Client`` would — hence the
async client here, and the sync one only in the offline ingestion script, §9.1).

Two reasons this is the single construction site:

- there is one place to configure/monkeypatch the client (tests patch
  :func:`get_voyage_client` at this boundary — CLAUDE.md rule 10, no live API in
  CI), and
- the client is built on first use, not at import time. A ``voyageai`` client can
  raise at construction when no API key is resolvable (neither ``VOYAGE_API_KEY``
  nor the SDK's env fallback), so lazy construction keeps importing the app safe
  with an empty key — the key is only required when a real embed call is made,
  which in CI never happens because the client is mocked.

The API key comes from ``settings.VOYAGE_API_KEY``; when empty we pass ``None`` so
the SDK falls back to its own env-var lookup, matching :mod:`app.llm.client`.
"""

from __future__ import annotations

import voyageai

from app.core.config import settings

_client: voyageai.AsyncClient | None = None


def get_voyage_client() -> voyageai.AsyncClient:
    """Return the process-wide async Voyage client, constructing it on first use."""
    global _client
    if _client is None:
        _client = voyageai.AsyncClient(api_key=settings.VOYAGE_API_KEY or None)
    return _client
