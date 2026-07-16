"""Episodic memory pipeline — extract, store, threshold-check, consolidate.

Runs once per chat turn, *after* the SSE response has fully streamed, as a
background task (see ``app.api.routes.chat``). One Haiku pass extracts durable,
user-specific facts ("observations") from the latest turn; observations accumulate
per ``(user_id, topic_key)``; once a topic has been seen in enough DISTINCT
conversations (``settings.MEMORY_CONSOLIDATION_THRESHOLD``) a second Haiku pass
consolidates its observations into a single durable ``user_memories`` row, which the
orchestrator injects into the system prompt every turn.

Design decisions worth flagging:
- **Never raises out.** This is post-response background work; every failure path
  (bad model output, DB error, provider outage) is caught, logged, and swallowed so
  it can neither break nor delay a chat turn (CLAUDE.md rule 4 spirit). Partial
  progress is fine — a topic that fails to consolidate this turn is retried on the
  next observation.
- **Deterministic logic in code** (CLAUDE.md rule 3): the threshold is a
  ``COUNT(DISTINCT conversation_id)`` in SQL, not an LLM judgment. Haiku only does the
  two language tasks (extract, synthesize).
- **Security** (CLAUDE.md rule 2): ``user_id`` comes only from the caller (JWT-derived),
  never from model output; every query filters ``WHERE user_id = :user_id``
  (application-level access control). Memory text is a third-person paraphrase, never
  verbatim user input, and is never used to construct tool arguments anywhere.

The client factory (:func:`get_anthropic_client`) is imported and called here so
tests patch it at THIS module and no live API is hit in CI (CLAUDE.md rule 10).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable

from sqlalchemy import distinct, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.client import get_anthropic_client
from app.models.memory_observation import MemoryObservation
from app.models.user_memory import UserMemory

log = get_logger(__name__)

# Fixed category enum — Haiku is told to use only these; anything else is dropped.
VALID_CATEGORIES = frozenset(
    {"goals", "preferences", "schedule", "equipment", "physical_context", "lifestyle"}
)

# Hard cap on stored fact length (defense-in-depth against a runaway generation
# entering the system prompt; the prompt also asks for short facts).
MAX_CONTENT_CHARS = 300

_EXTRACTION_MAX_TOKENS = 500
_CONSOLIDATION_MAX_TOKENS = 300

_EXTRACTION_PROMPT = (
    "You extract durable, user-specific facts from ONE turn of a conversation between "
    "a user and their AI weightlifting coach. These facts become long-term memory the "
    "coach sees in every future chat, so record only things that stay true over time "
    "and are specifically about THIS user.\n\n"
    "Return a STRICT JSON array and nothing else. Each element is an object with "
    "exactly these keys:\n"
    '  "category": one of goals, preferences, schedule, equipment, physical_context, '
    "lifestyle\n"
    '  "topic_key": a short lowercase snake_case noun key naming the fact\'s subject\n'
    '  "content": a third-person declarative sentence stating the fact about the user\n\n'
    "Most turns contain NO durable facts. When in doubt, extract nothing. An empty "
    "array [] is the normal, expected result.\n\n"
    "topic_key rules:\n"
    "- Reuse one of the user's existing topic_keys below whenever the new fact concerns "
    "that same subject; only coin a new key when none of them fits.\n"
    "- Existing topic_keys for this user: {existing_keys}\n\n"
    "content rules:\n"
    '- Write in the third person about the user, e.g. "User trains in a home gym with '
    'dumbbells up to 50 lbs." Never quote the user verbatim and never use second person '
    '("you").\n'
    "- State only what the user themselves conveyed; do not infer or add advice.\n\n"
    "Do NOT extract:\n"
    "- Medical diagnoses or inferences. Pain or physical limitations may be recorded "
    'ONLY as facts the user explicitly stated about themselves (e.g. "User has said '
    'their left shoulder hurts during overhead pressing"), never as a diagnosis of a '
    "condition.\n"
    "- Anything the app already stores and can look up: specific workout numbers, sets, "
    "reps, weights, dates, or logged sessions.\n"
    '- Transient or momentary states ("tired today", "sore from yesterday\'s session").\n'
    "- Coaching advice, or anything about the coach; record facts about the USER only.\n\n"
    "Conversation turn:\n"
    "User: {user_message}\n"
    "Coach: {assistant_reply}\n\n"
    "JSON array:"
)

_CONSOLIDATION_PROMPT = (
    "You maintain a single long-term memory about a user for their AI weightlifting "
    "coach. Below are dated observations about one topic, oldest first. Synthesize them "
    "into ONE concise, third-person memory of at most two sentences that the coach can "
    "rely on going forward.\n\n"
    "When observations conflict, the most recent information wins — drop anything a "
    "later observation contradicts, and keep only what is still true and durable. Write "
    "in the third person about the user, never second person, and add no advice or "
    "commentary. Output only the memory text.\n\n"
    "Topic: {topic_key}\n"
    "Observations (oldest to newest):\n"
    "{observations}\n\n"
    "Consolidated memory:"
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
async def process_turn(
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    user_message: str,
    assistant_reply: str,
    db: AsyncSession,
) -> None:
    """Extract → store → threshold-check → consolidate for one chat turn.

    Best-effort and non-raising by contract: this runs post-response in a background
    task and must NEVER propagate an exception (it would surface nowhere useful and
    could disrupt the task runner). Every phase is guarded; a per-topic failure is
    isolated so it doesn't abort consolidation of the other affected topics.
    """
    try:
        existing_keys = await _existing_topic_keys(user_id, db)
        observations = await _extract_observations(user_message, assistant_reply, existing_keys)
        if not observations:
            return

        affected_topics = await _store_observations(user_id, conversation_id, observations, db)

        for topic_key in affected_topics:
            try:
                await _maybe_consolidate(user_id, topic_key, db)
            except Exception:  # noqa: BLE001 — isolate one topic's failure from the rest
                log.exception(
                    "memory_consolidation_error",
                    extra={"user_id": str(user_id), "topic_key": topic_key},
                )
                await _safe_rollback(db)
    except Exception:  # noqa: BLE001 — background work must never raise out
        log.exception("memory_process_turn_error", extra={"user_id": str(user_id)})
        await _safe_rollback(db)


async def get_memories_for_prompt(user_id: uuid.UUID, db: AsyncSession) -> list[UserMemory]:
    """Return the user's consolidated memories for system-prompt injection.

    Top ``settings.MEMORY_MAX_INJECTED`` by ``updated_at`` descending (most recently
    refreshed first). Filtered to the caller's own ``user_id`` (application-level
    access control). Used by the orchestrator each turn and by the ``GET /memories``
    route.
    """
    result = await db.execute(
        select(UserMemory)
        .where(UserMemory.user_id == user_id)
        .order_by(UserMemory.updated_at.desc())
        .limit(settings.MEMORY_MAX_INJECTED)
    )
    return list(result.scalars().all())


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
async def _existing_topic_keys(user_id: uuid.UUID, db: AsyncSession) -> set[str]:
    """Distinct topic_keys this user already has observations for (for key reuse)."""
    result = await db.execute(
        select(MemoryObservation.topic_key).where(MemoryObservation.user_id == user_id).distinct()
    )
    return set(result.scalars().all())


async def _extract_observations(
    user_message: str, assistant_reply: str, existing_keys: set[str]
) -> list[dict]:
    """One Haiku call over just this turn; returns validated observation dicts."""
    prompt = _EXTRACTION_PROMPT.format(
        existing_keys=_format_existing_keys(existing_keys),
        user_message=user_message,
        assistant_reply=assistant_reply,
    )
    client = get_anthropic_client()
    response = await client.messages.create(
        model=settings.HAIKU_MODEL_ID,
        max_tokens=_EXTRACTION_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_observations(response.content[0].text)


def _format_existing_keys(keys: Iterable[str]) -> str:
    ordered = sorted(keys)
    return ", ".join(ordered) if ordered else "(none yet)"


def _parse_observations(raw: str) -> list[dict]:
    """Defensively parse Haiku's JSON array; drop invalid rows, never raise.

    Slices from the first ``[`` to the last ``]`` so surrounding prose or code fences
    are tolerated. Each row must have a valid ``category`` (from the enum), a non-empty
    ``topic_key``, and non-empty ``content``; anything else is skipped. ``content`` is
    truncated to ``MAX_CONTENT_CHARS`` and ``topic_key`` is normalized in code.
    """
    data = _loads_json_array(raw)
    if not isinstance(data, list):
        log.warning("memory_extraction_unparseable", extra={"raw_prefix": raw[:200]})
        return []

    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        topic_key = item.get("topic_key")
        content = item.get("content")
        if category not in VALID_CATEGORIES:
            continue
        if not isinstance(topic_key, str) or not topic_key.strip():
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        out.append(
            {
                "category": category,
                "topic_key": _normalize_topic_key(topic_key),
                "content": content.strip()[:MAX_CONTENT_CHARS],
            }
        )
    return out


def _loads_json_array(raw: str) -> object:
    """Parse the first ``[...]`` array found in ``raw``; return None on failure."""
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_topic_key(topic_key: str) -> str:
    """Deterministically normalize a key so reuse matching isn't foiled by casing/spaces."""
    return topic_key.strip().lower().replace(" ", "_")


# --------------------------------------------------------------------------- #
# Storage + threshold + consolidation
# --------------------------------------------------------------------------- #
async def _store_observations(
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    observations: list[dict],
    db: AsyncSession,
) -> set[str]:
    """Persist this turn's observations; return the distinct topic_keys touched."""
    affected: set[str] = set()
    for obs in observations:
        db.add(
            MemoryObservation(
                user_id=user_id,
                conversation_id=conversation_id,
                category=obs["category"],
                topic_key=obs["topic_key"],
                content=obs["content"],
            )
        )
        affected.add(obs["topic_key"])
    await db.commit()
    return affected


async def _maybe_consolidate(user_id: uuid.UUID, topic_key: str, db: AsyncSession) -> None:
    """Consolidate iff the topic crossed the distinct-conversation threshold OR is graduated.

    Deterministic gate (CLAUDE.md rule 3): count DISTINCT conversation_ids for
    ``(user, topic_key)``. Consolidate when that reaches
    ``settings.MEMORY_CONSOLIDATION_THRESHOLD``, or when a ``user_memories`` row already
    exists (post-graduation refresh — any new observation on a graduated topic updates
    the memory so recent info wins).
    """
    distinct_convs = (
        await db.execute(
            select(func.count(distinct(MemoryObservation.conversation_id))).where(
                MemoryObservation.user_id == user_id,
                MemoryObservation.topic_key == topic_key,
            )
        )
    ).scalar_one()

    already_graduated = (
        await db.execute(
            select(UserMemory.id).where(
                UserMemory.user_id == user_id, UserMemory.topic_key == topic_key
            )
        )
    ).scalar_one_or_none() is not None

    if distinct_convs < settings.MEMORY_CONSOLIDATION_THRESHOLD and not already_graduated:
        return

    await _consolidate(user_id, topic_key, distinct_convs, db)


async def _consolidate(
    user_id: uuid.UUID, topic_key: str, source_chat_count: int, db: AsyncSession
) -> None:
    """Synthesize all observations for a topic into one memory row (upsert)."""
    observations = (
        (
            await db.execute(
                select(MemoryObservation)
                .where(
                    MemoryObservation.user_id == user_id,
                    MemoryObservation.topic_key == topic_key,
                )
                .order_by(MemoryObservation.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    if not observations:
        return

    # Category of the most recent observation wins if a topic's category ever drifts.
    category = observations[-1].category
    rendered = "\n".join(f"- ({obs.created_at:%Y-%m-%d}) {obs.content}" for obs in observations)

    client = get_anthropic_client()
    response = await client.messages.create(
        model=settings.HAIKU_MODEL_ID,
        max_tokens=_CONSOLIDATION_MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": _CONSOLIDATION_PROMPT.format(topic_key=topic_key, observations=rendered),
            }
        ],
    )
    content = response.content[0].text.strip()[:MAX_CONTENT_CHARS]
    if not content:
        log.warning(
            "memory_consolidation_empty",
            extra={"user_id": str(user_id), "topic_key": topic_key},
        )
        return

    # Upsert on the (user_id, topic_key) unique constraint: a concurrent turn
    # consolidating the same topic can't create a duplicate (last write wins).
    stmt = (
        pg_insert(UserMemory)
        .values(
            user_id=user_id,
            category=category,
            topic_key=topic_key,
            content=content,
            source_chat_count=source_chat_count,
        )
        .on_conflict_do_update(
            index_elements=["user_id", "topic_key"],
            set_={
                "content": content,
                "category": category,
                "source_chat_count": source_chat_count,
                "updated_at": func.now(),
            },
        )
    )
    await db.execute(stmt)
    await db.commit()
    log.info(
        "memory_consolidated",
        extra={
            "user_id": str(user_id),
            "topic_key": topic_key,
            "source_chat_count": source_chat_count,
        },
    )


async def _safe_rollback(db: AsyncSession) -> None:
    """Roll back a possibly-poisoned session, swallowing any rollback error."""
    try:
        await db.rollback()
    except Exception:  # noqa: BLE001 — cleanup must not mask the original failure
        log.exception("memory_rollback_error")
