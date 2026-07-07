"""Shared helpers for the Phase 3a route tests.

A registered user's cookies live in the shared session-scoped ``TestClient``
jar. Because multiple tests (and both users A and B in the access-control checks)
share one client, tests that need a specific authenticated identity call
``register_user`` to (re)establish whose cookies are active, then act. The
``authed_client`` context manager makes "act as this user" explicit and restores
the previous jar afterwards.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.exercise import Exercise
from app.models.set_entry import SetEntry
from app.models.user import User
from app.models.workout_session import WorkoutSession


def unique_email() -> str:
    return f"user_{uuid.uuid4().hex}@example.com"


def register_payload(email: str | None = None, password: str = "password123") -> dict:
    return {
        "email": email or unique_email(),
        "password": password,
        "display_name": "Test Lifter",
        "experience_level": "intermediate",
        "primary_goal": "hypertrophy",
        "injury_notes": None,
    }


def register_user(client: TestClient, email: str | None = None) -> dict:
    """Register a fresh user; leaves only that user's auth cookies active on ``client``.

    Clears the jar first so the register response's Set-Cookie is the only
    ``access_token`` present (httpx2 raises ``CookieConflict`` on duplicate names,
    which happens when tests register A then B on the shared client).

    Returns the created user's profile body (includes ``id``).
    """
    client.cookies.clear()
    payload = register_payload(email=email)
    resp = client.post("/auth/register", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def login_user(client: TestClient, email: str, password: str = "password123") -> None:
    """Log a user in, making their cookies the active ones on ``client``."""
    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text


@contextlib.contextmanager
def authed_client(
    client: TestClient, email: str, password: str = "password123"
) -> Iterator[TestClient]:
    """Temporarily act as ``email`` on the shared client, leaving a clean jar after.

    Clears the jar before logging in (so the login's Set-Cookie is the only
    ``access_token`` present — httpx2 raises ``CookieConflict`` if two cookies
    share a name) and clears it again on exit, so no stale identity leaks into the
    next test.
    """
    client.cookies.clear()
    login_user(client, email, password)
    try:
        yield client
    finally:
        client.cookies.clear()


def get_exercise_ids(client: TestClient, count: int = 2) -> list[str]:
    """Return ``count`` real exercise ids from the seeded catalog (auth required)."""
    resp = client.get("/exercises")
    assert resp.status_code == 200, resp.text
    exercises = resp.json()
    assert len(exercises) >= count, "seeded catalog too small for this test"
    return [e["id"] for e in exercises[:count]]


# --------------------------------------------------------------------------- #
# Direct-DB helpers for service/agent-layer tests (progression math, tool handlers)
# --------------------------------------------------------------------------- #
async def create_db_user(db: AsyncSession) -> uuid.UUID:
    """Insert a minimal user row directly and return its id (no HTTP/auth needed)."""
    user = User(
        email=unique_email(),
        hashed_password="x",  # not verified in these tests
        display_name="Test Lifter",
        experience_level="intermediate",
        primary_goal="hypertrophy",
        injury_notes=None,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user.id


async def first_exercise_id(db: AsyncSession) -> uuid.UUID:
    """Return the id of the first seeded exercise (ordered by name)."""
    return (await db.execute(select(Exercise.id).order_by(Exercise.name).limit(1))).scalar_one()


async def add_session_with_sets(
    db: AsyncSession,
    user_id: uuid.UUID,
    exercise_id: uuid.UUID,
    sets: list[tuple[float, int, float | None]],
    days_ago: int = 0,
    status: str = "finished",
) -> uuid.UUID:
    """Create one workout session dated ``days_ago`` days back, with the given sets.

    ``sets`` is a list of ``(weight, reps, rir)`` tuples. Session date drives the
    oldest->newest ordering in ``progression_service.analyze``. Returns the session id.
    """
    when = datetime.now(UTC) - timedelta(days=days_ago)
    session = WorkoutSession(user_id=user_id, program_id=None, date=when, status=status)
    db.add(session)
    await db.flush()
    for i, (weight, reps, rir) in enumerate(sets, start=1):
        db.add(
            SetEntry(
                session_id=session.id,
                user_id=user_id,
                exercise_id=exercise_id,
                set_number=i,
                weight=weight,
                reps=reps,
                rir=rir,
            )
        )
    await db.commit()
    return session.id
