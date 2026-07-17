"""Exercise catalog service.

The exercise catalog is global and read-only — it is not user-scoped, so these
functions are the deliberate exception to the "every service function takes a
``user_id``" rule: there is no user data here to
protect.

``search_exercises`` is the deterministic fuzzy-match used by both the
``GET /exercises/search`` endpoint and (Phase 4) the ``search_exercises`` tool.
The matching is done in code with ``rapidfuzz``, never
delegated to LLM judgment.
"""

from __future__ import annotations

import uuid

from rapidfuzz import fuzz, process
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.exercise import Exercise

# Minimum WRatio score (0-100) for a fuzzy match to count. WRatio is
# tolerant of word order, partial tokens, and length differences, so casual
# names like "bench" or "back squat" still resolve; below this threshold the
# match is too weak to trust and we return nothing rather than guess. Tunable —
# raise it to reduce false positives, lower it to catch sloppier input.
MIN_MATCH_SCORE = 60.0

# Top-N candidates returned to the caller / agent.
MAX_MATCHES = 5


async def list_exercises(db: AsyncSession) -> list[Exercise]:
    """Return the full exercise catalog, ordered by name (global, read-only)."""
    result = await db.execute(select(Exercise).order_by(Exercise.name))
    return list(result.scalars().all())


async def search_exercises(db: AsyncSession, query: str) -> list[dict]:
    """Fuzzy-match ``query`` against exercise names; return top matches.

    Returns up to :data:`MAX_MATCHES` candidates scoring at or above
    :data:`MIN_MATCH_SCORE`, as ``{"exercise_id", "name", "score"}`` dicts sorted
    best-first. An empty/whitespace query, or a query with no match above the
    threshold, returns ``[]`` — the caller (or agent) should then tell the user
    the exercise wasn't recognized rather than picking a weak closest match.
    """
    query = (query or "").strip()
    if not query:
        return []

    result = await db.execute(select(Exercise.id, Exercise.name))
    rows = result.all()  # list of (id, name)
    if not rows:
        return []

    # WRatio is case-insensitive, so names are scored as-is; this map recovers the
    # (id, name) for each matched name string.
    names = [name for (_id, name) in rows]
    by_name: dict[str, tuple[uuid.UUID, str]] = {name: (row_id, name) for (row_id, name) in rows}

    extracted = process.extract(
        query,
        names,
        scorer=fuzz.WRatio,
        limit=MAX_MATCHES,
        score_cutoff=MIN_MATCH_SCORE,
    )

    matches: list[dict] = []
    for name, score, _index in extracted:
        row_id, original_name = by_name[name]
        matches.append(
            {
                "exercise_id": str(row_id),
                "name": original_name,
                "score": round(float(score), 2),
            }
        )
    return matches
