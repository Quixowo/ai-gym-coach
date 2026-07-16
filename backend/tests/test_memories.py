"""Memory endpoint tests: GET /memories, DELETE /memories/{id}.

Rows are seeded directly through the NullPool test session (``test_session_maker``)
rather than via the pipeline itself (that's covered by the memory-extraction suite),
so these tests are pure route-behavior checks: user scoping, the no-cap GET, and the
DELETE-cascades-observations contract. Two users share the client jar (see
``tests/helpers.py``), so ``register_user``/``authed_client`` establish whose cookies
are active before each request, matching the pattern in ``test_access_control.py``.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.rate_limit import limiter
from app.models.memory_observation import MemoryObservation
from app.models.user_memory import UserMemory
from tests.conftest import test_session_maker
from tests.helpers import authed_client, register_user, unique_email


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    limiter.reset()


async def _add_memory(
    user_id: uuid.UUID,
    topic_key: str,
    category: str = "preferences",
    content: str = "User trains in a home gym.",
    source_chat_count: int = 3,
) -> uuid.UUID:
    async with test_session_maker() as db:
        memory = UserMemory(
            user_id=user_id,
            category=category,
            topic_key=topic_key,
            content=content,
            source_chat_count=source_chat_count,
        )
        db.add(memory)
        await db.commit()
        await db.refresh(memory)
        return memory.id


async def _add_observations(user_id: uuid.UUID, topic_key: str, count: int) -> None:
    async with test_session_maker() as db:
        for i in range(count):
            db.add(
                MemoryObservation(
                    user_id=user_id,
                    conversation_id=uuid.uuid4(),
                    category="preferences",
                    topic_key=topic_key,
                    content=f"Observation {i}",
                )
            )
        await db.commit()


async def _observation_count(user_id: uuid.UUID, topic_key: str) -> int:
    async with test_session_maker() as db:
        result = await db.execute(
            select(MemoryObservation).where(
                MemoryObservation.user_id == user_id,
                MemoryObservation.topic_key == topic_key,
            )
        )
        return len(result.scalars().all())


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def test_endpoints_require_auth(client: TestClient) -> None:
    client.cookies.clear()
    assert client.get("/memories").status_code == 401
    assert client.delete(f"/memories/{uuid.uuid4()}").status_code == 401


# --------------------------------------------------------------------------- #
# GET — no memories, and no MEMORY_MAX_INJECTED cap
# --------------------------------------------------------------------------- #
def test_get_memories_empty(client: TestClient) -> None:
    register_user(client)
    resp = client.get("/memories")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_memories_returns_only_callers_own(client: TestClient) -> None:
    a_email = unique_email()
    a_user = register_user(client, email=a_email)
    a_id = uuid.UUID(a_user["id"])

    asyncio.run(_add_memory(a_id, topic_key="home_gym"))
    asyncio.run(_add_memory(a_id, topic_key="shoulder_history"))

    b_email = unique_email()
    b_user = register_user(client, email=b_email)
    b_id = uuid.UUID(b_user["id"])
    asyncio.run(_add_memory(b_id, topic_key="squat_frequency"))

    # B sees only B's memory.
    b_resp = client.get("/memories")
    assert b_resp.status_code == 200
    b_body = b_resp.json()
    assert len(b_body) == 1
    assert b_body[0]["topic_key"] == "squat_frequency"

    # A sees only A's two memories, most-recently-updated first, full response shape.
    with authed_client(client, a_email):
        a_resp = client.get("/memories")
        assert a_resp.status_code == 200
        a_body = a_resp.json()
        assert {m["topic_key"] for m in a_body} == {"home_gym", "shoulder_history"}
        row = a_body[0]
        assert set(row.keys()) == {
            "id",
            "category",
            "topic_key",
            "content",
            "source_chat_count",
            "created_at",
            "updated_at",
        }


# --------------------------------------------------------------------------- #
# DELETE — removes memory + topic's observations, leaves other topics intact
# --------------------------------------------------------------------------- #
def test_delete_removes_memory_and_topic_observations(client: TestClient) -> None:
    user = register_user(client)
    user_id = uuid.UUID(user["id"])

    memory_id = asyncio.run(_add_memory(user_id, topic_key="shoulder_history"))
    asyncio.run(_add_observations(user_id, "shoulder_history", count=3))
    # A different topic's observations must survive the delete.
    asyncio.run(_add_observations(user_id, "home_gym", count=2))

    assert asyncio.run(_observation_count(user_id, "shoulder_history")) == 3
    assert asyncio.run(_observation_count(user_id, "home_gym")) == 2

    resp = client.delete(f"/memories/{memory_id}")
    assert resp.status_code == 204
    assert resp.content == b""

    # Memory gone.
    assert client.get("/memories").json() == []
    # This topic's observations gone.
    assert asyncio.run(_observation_count(user_id, "shoulder_history")) == 0
    # Other topic's observations untouched.
    assert asyncio.run(_observation_count(user_id, "home_gym")) == 2


def test_delete_other_users_memory_404s_and_survives(client: TestClient) -> None:
    a_email = unique_email()
    a_user = register_user(client, email=a_email)
    a_id = uuid.UUID(a_user["id"])
    memory_id = asyncio.run(_add_memory(a_id, topic_key="shoulder_history"))
    asyncio.run(_add_observations(a_id, "shoulder_history", count=3))

    b_email = unique_email()
    register_user(client, email=b_email)
    resp = client.delete(f"/memories/{memory_id}")
    assert resp.status_code == 404

    # A's memory and observations are untouched by B's attempt.
    with authed_client(client, a_email):
        assert len(client.get("/memories").json()) == 1
    assert asyncio.run(_observation_count(a_id, "shoulder_history")) == 3


def test_delete_random_uuid_404s(client: TestClient) -> None:
    register_user(client)
    resp = client.delete(f"/memories/{uuid.uuid4()}")
    assert resp.status_code == 404
