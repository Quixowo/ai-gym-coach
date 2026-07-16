"""Pure-code memory-pipeline tests (deterministic logic, no live API).

These exercise the parts CLAUDE.md rule 3 keeps in code, not in the model: the
DISTINCT-conversation threshold gate, post-graduation re-consolidation, upsert
idempotency, and the system-prompt injection rendering + cap. The only model
involvement is a trivial stubbed consolidation reply (a fixed string), replayed via
:class:`FakeAnthropicClient` with ``get_anthropic_client`` patched at the
``memory_service`` boundary — no live Haiku call (CLAUDE.md rule 10).

All DB work runs through the NullPool ``test_session_maker`` (never the app's pooled
maker) to stay clear of the closed-loop bugs in LESSONS.md.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

import app.services.memory_service as memory_service
from app.agent.orchestrator import (
    _SYSTEM_PROMPT_TEMPLATE,
    _render_memory_block,
    build_system_prompt,
)
from app.core.config import settings
from app.models.memory_observation import MemoryObservation
from app.models.user_memory import UserMemory
from app.services.memory_service import (
    _consolidate,
    _maybe_consolidate,
    get_memories_for_prompt,
)
from tests.conftest import test_session_maker as session_maker
from tests.helpers import create_db_user


async def _add_observation(
    db,
    user_id: uuid.UUID,
    topic_key: str,
    conversation_id: uuid.UUID,
    category: str = "goals",
    content: str = "An observation.",
) -> None:
    db.add(
        MemoryObservation(
            user_id=user_id,
            conversation_id=conversation_id,
            category=category,
            topic_key=topic_key,
            content=content,
        )
    )
    await db.commit()


async def _memory_row(db, user_id: uuid.UUID, topic_key: str) -> UserMemory | None:
    return (
        await db.execute(
            select(UserMemory).where(
                UserMemory.user_id == user_id, UserMemory.topic_key == topic_key
            )
        )
    ).scalar_one_or_none()


def _patch_consolidation_reply(monkeypatch: pytest.MonkeyPatch, *texts: str) -> None:
    from tests.fixtures._replay import FakeAnthropicClient

    fake = FakeAnthropicClient(create_texts=list(texts))
    monkeypatch.setattr(memory_service, "get_anthropic_client", lambda: fake)


# --------------------------------------------------------------------------- #
# Threshold counts DISTINCT conversations, not observation rows
# --------------------------------------------------------------------------- #
async def test_three_observations_one_conversation_does_not_consolidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 observations in ONE conversation is 1 distinct conversation -> no graduation."""
    # Patch the client to a fake with NO replies: if consolidation were (wrongly)
    # attempted it would raise on the empty create script, failing loudly.
    _patch_consolidation_reply(monkeypatch)

    async with session_maker() as db:
        user_id = await create_db_user(db)
        one_conversation = uuid.uuid4()
        for i in range(3):
            await _add_observation(db, user_id, "bench_goal", one_conversation, content=f"obs {i}")

        await _maybe_consolidate(user_id, "bench_goal", db)

        assert await _memory_row(db, user_id, "bench_goal") is None


async def test_three_distinct_conversations_consolidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 observations across 3 distinct conversations reaches the threshold -> memory row."""
    _patch_consolidation_reply(monkeypatch, "User is working toward a 225 lb bench press.")

    async with session_maker() as db:
        user_id = await create_db_user(db)
        for i in range(settings.MEMORY_CONSOLIDATION_THRESHOLD):
            await _add_observation(db, user_id, "bench_goal", uuid.uuid4(), content=f"obs {i}")

        await _maybe_consolidate(user_id, "bench_goal", db)

        mem = await _memory_row(db, user_id, "bench_goal")
        assert mem is not None
        assert mem.source_chat_count == 3
        assert mem.category == "goals"
        assert mem.content == "User is working toward a 225 lb bench press."


# --------------------------------------------------------------------------- #
# Post-graduation refresh: one new observation on a 4th conversation re-consolidates
# --------------------------------------------------------------------------- #
async def test_post_graduation_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_consolidation_reply(
        monkeypatch,
        "User's left shoulder used to ache on overhead work.",
        "User's left shoulder now feels fine on overhead pressing.",
    )

    async with session_maker() as db:
        user_id = await create_db_user(db)
        for i in range(settings.MEMORY_CONSOLIDATION_THRESHOLD):
            await _add_observation(
                db,
                user_id,
                "shoulder_history",
                uuid.uuid4(),
                category="physical_context",
                content=f"obs {i}",
            )

        # Graduation.
        await _maybe_consolidate(user_id, "shoulder_history", db)
        mem = await _memory_row(db, user_id, "shoulder_history")
        assert mem is not None and mem.source_chat_count == 3
        before_updated = mem.updated_at
        before_content = mem.content

        # One new observation from a 4th distinct conversation.
        await _add_observation(
            db,
            user_id,
            "shoulder_history",
            uuid.uuid4(),
            category="physical_context",
            content="obs 4 (recovered)",
        )
        await _maybe_consolidate(user_id, "shoulder_history", db)

        await db.refresh(mem)
        assert mem.source_chat_count == 4
        assert mem.content != before_content
        assert mem.content == "User's left shoulder now feels fine on overhead pressing."
        assert mem.updated_at > before_updated


# --------------------------------------------------------------------------- #
# Upsert idempotency: consolidating the same (user, topic_key) twice = one row
# --------------------------------------------------------------------------- #
async def test_consolidate_is_idempotent_on_user_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_consolidation_reply(monkeypatch, "First synthesis.", "Second synthesis.")

    async with session_maker() as db:
        user_id = await create_db_user(db)
        for i in range(3):
            await _add_observation(db, user_id, "grip_preference", uuid.uuid4(), content=f"obs {i}")

        await _consolidate(user_id, "grip_preference", 3, db)
        await _consolidate(user_id, "grip_preference", 3, db)

        count = (
            await db.execute(
                select(func.count(UserMemory.id)).where(
                    UserMemory.user_id == user_id,
                    UserMemory.topic_key == "grip_preference",
                )
            )
        ).scalar_one()
        assert count == 1
        mem = await _memory_row(db, user_id, "grip_preference")
        assert mem.content == "Second synthesis."


# --------------------------------------------------------------------------- #
# Rendering: empty block omitted (prompt == pre-feature shape); non-empty format
# --------------------------------------------------------------------------- #
def test_render_memory_block_empty_returns_empty_string() -> None:
    assert _render_memory_block([]) == ""


async def test_build_system_prompt_omits_block_when_no_memories() -> None:
    """With no memories the built prompt equals the template with an empty memory_block."""
    async with session_maker() as db:
        user_id = await create_db_user(db)
        prompt = await build_system_prompt(user_id, db)

    # create_db_user uses these exact profile values.
    expected = _SYSTEM_PROMPT_TEMPLATE.format(
        display_name="Test Lifter",
        experience_level="intermediate",
        primary_goal="hypertrophy",
        injury_notes="none provided",
        memory_block="",
    )
    assert prompt == expected
    assert "What you remember about this user" not in prompt


def test_render_memory_block_format() -> None:
    """Non-empty block: delimited header + ``- [category] content (as of Mon YYYY)`` lines."""
    memories = [
        UserMemory(
            category="goals",
            content="User wants to bench 225 lbs.",
            updated_at=datetime(2026, 3, 15, tzinfo=UTC),
        ),
        UserMemory(
            category="equipment",
            content="User trains at home with dumbbells.",
            updated_at=datetime(2026, 7, 1, tzinfo=UTC),
        ),
    ]
    block = _render_memory_block(memories)

    assert block.startswith("\n") and block.endswith("\n")
    assert "What you remember about this user" in block
    assert "- [goals] User wants to bench 225 lbs. (as of Mar 2026)" in block
    assert "- [equipment] User trains at home with dumbbells. (as of Jul 2026)" in block


# --------------------------------------------------------------------------- #
# Injection cap: more than MEMORY_MAX_INJECTED -> only the 15 most-recent injected
# --------------------------------------------------------------------------- #
async def test_injection_caps_at_max_injected_most_recent_first() -> None:
    over = settings.MEMORY_MAX_INJECTED + 2  # 17
    base = datetime(2026, 1, 1, tzinfo=UTC)

    async with session_maker() as db:
        user_id = await create_db_user(db)
        for i in range(over):
            db.add(
                UserMemory(
                    user_id=user_id,
                    category="preferences",
                    topic_key=f"topic_{i}",
                    content=f"Fact number {i}.",
                    source_chat_count=3,
                    # Explicit, staggered updated_at so recency ordering is deterministic.
                    updated_at=base + timedelta(minutes=i),
                )
            )
        await db.commit()

        memories = await get_memories_for_prompt(user_id, db)
        prompt = await build_system_prompt(user_id, db)

    # Only the cap's worth returned, most-recently-updated first.
    assert len(memories) == settings.MEMORY_MAX_INJECTED
    assert memories[0].topic_key == f"topic_{over - 1}"
    assert memories[-1].topic_key == "topic_2"

    # The prompt injects exactly 15 memory lines; the two oldest are excluded.
    assert prompt.count("- [preferences]") == settings.MEMORY_MAX_INJECTED
    assert "Fact number 0." not in prompt
    assert "Fact number 1." not in prompt
    assert f"Fact number {over - 1}." in prompt
