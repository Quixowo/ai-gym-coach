"""Workout session / set-entry service.

Single source of truth for the workout write/read path, shared by the REST
endpoints (Phase 3) and the ``log_set`` / ``get_workout_history`` agent tools
(Phase 4). Every function that touches user data takes ``user_id`` as an explicit
argument and filters on it — this is the project's application-level access
control: a caller-supplied session or set id is
never trusted without also matching ``user_id``, so a cross-user reference
surfaces as :class:`NotFoundError` (404), never a leak.

Deterministic logic lives here, not in the model's judgment:
field validation, the open-session find-or-create, and the per-exercise
``set_number`` computation are all done in code.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.exercise import Exercise
from app.models.set_entry import SetEntry
from app.models.workout_session import WorkoutSession
from app.services.errors import ConflictError, NotFoundError
from app.services.validation import validate_set_fields

DEFAULT_HISTORY_LIMIT = 50


async def _get_open_session(db: AsyncSession, user_id: uuid.UUID) -> WorkoutSession | None:
    """Return the user's single open session, or None (application-level access control).

    The invariant is "at most one open session per user", so this expects 0
    or 1 rows. If more than one somehow exists, the most recent is returned so the
    write path stays usable rather than erroring.
    """
    result = await db.execute(
        select(WorkoutSession)
        .where(WorkoutSession.user_id == user_id, WorkoutSession.status == "open")
        .order_by(WorkoutSession.created_at.desc())
    )
    return result.scalars().first()


async def log_set(
    db: AsyncSession,
    user_id: uuid.UUID,
    exercise_id: uuid.UUID,
    weight: float,
    reps: int,
    rir: float | None = None,
) -> SetEntry:
    """Log one completed set, resolving the target session.

    Validates weight/reps/rir in-service before any write. Verifies the
    exercise exists (structured :class:`NotFoundError`, never a raw FK
    ``IntegrityError``). Resolves the session: exactly-one-open -> attach; none ->
    create a new ``status="open"`` session dated today. ``set_number`` is computed
    server-side as ``max(set_number for this exercise in this session) + 1`` —
    never taken from the caller/LLM.
    """
    validate_set_fields(weight=weight, reps=reps, rir=rir)

    exercise = (
        await db.execute(select(Exercise.id).where(Exercise.id == exercise_id))
    ).scalar_one_or_none()
    if exercise is None:
        raise NotFoundError("exercise_id not found")

    session = await _get_open_session(db, user_id)
    if session is None:
        session = WorkoutSession(
            user_id=user_id,
            program_id=None,
            date=datetime.now(UTC),
            status="open",
        )
        db.add(session)
        # Flush so session.id is available for the set_entry FK below without
        # committing yet — both rows commit together at the end.
        await db.flush()

    # set_number = max existing for this exercise in this session + 1.
    max_set_number = (
        await db.execute(
            select(func.max(SetEntry.set_number)).where(
                SetEntry.session_id == session.id,
                SetEntry.exercise_id == exercise_id,
            )
        )
    ).scalar_one_or_none()
    next_set_number = (max_set_number or 0) + 1

    entry = SetEntry(
        session_id=session.id,
        user_id=user_id,
        exercise_id=exercise_id,
        set_number=next_set_number,
        weight=weight,
        reps=reps,
        rir=rir,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


async def get_history(
    db: AsyncSession,
    user_id: uuid.UUID,
    exercise_id: uuid.UUID | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> list[tuple[SetEntry, str]]:
    """Return the user's logged sets joined to exercise display names.

    Filtered by ``user_id`` (application-level access control), optionally by
    ``exercise_id`` and an inclusive ``[start_date, end_date]`` range on
    ``created_at``. Ordered most-recent-first and capped at ``limit``. Each item
    is ``(SetEntry, exercise_name)`` so callers get the display name without a
    second query.
    """
    query = (
        select(SetEntry, Exercise.name)
        .join(Exercise, SetEntry.exercise_id == Exercise.id)
        .where(SetEntry.user_id == user_id)
    )
    if exercise_id is not None:
        query = query.where(SetEntry.exercise_id == exercise_id)
    if start_date is not None:
        query = query.where(
            SetEntry.created_at >= datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
        )
    if end_date is not None:
        # Inclusive end: everything strictly before the start of the next day.
        end_exclusive = datetime.combine(end_date, datetime.max.time(), tzinfo=UTC)
        query = query.where(SetEntry.created_at <= end_exclusive)

    query = query.order_by(SetEntry.created_at.desc()).limit(limit)
    result = await db.execute(query)
    return [(row[0], row[1]) for row in result.all()]


async def list_sessions(db: AsyncSession, user_id: uuid.UUID) -> list[WorkoutSession]:
    """Return the user's sessions, most-recent-first (application-level access control)."""
    result = await db.execute(
        select(WorkoutSession)
        .where(WorkoutSession.user_id == user_id)
        .order_by(WorkoutSession.date.desc(), WorkoutSession.created_at.desc())
    )
    return list(result.scalars().all())


async def start_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    program_id: uuid.UUID | None = None,
    notes: str | None = None,
) -> WorkoutSession:
    """Start a new open session; refuse if one is already open.

    Enforces the "at most one open session per user" invariant: raises
    :class:`ConflictError` (routes -> HTTP 409) if the user already has an open
    session, so the manual "start workout" flow can't create a duplicate that
    ``log_set``'s find-or-create would then be unable to disambiguate.
    """
    existing = await _get_open_session(db, user_id)
    if existing is not None:
        raise ConflictError("an open session already exists")

    session = WorkoutSession(
        user_id=user_id,
        program_id=program_id,
        date=datetime.now(UTC),
        status="open",
        notes=notes,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def finish_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
) -> WorkoutSession:
    """Mark the user's session ``status="finished"``.

    Filters on ``user_id`` as well as ``session_id`` (application-level access
    control): another user's session id resolves to :class:`NotFoundError` (404),
    never modifying or revealing it.
    """
    result = await db.execute(
        select(WorkoutSession).where(
            WorkoutSession.id == session_id,
            WorkoutSession.user_id == user_id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise NotFoundError("session not found")

    session.status = "finished"
    await db.commit()
    await db.refresh(session)
    return session
