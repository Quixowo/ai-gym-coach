"""Workout session / set-entry endpoints.

Every endpoint is behind ``get_current_user`` and passes the resolved ``user_id``
explicitly into ``workout_service`` (application-level access control)
— the service filters on it, so a cross-user session id 404s rather than leaking.

Service domain exceptions are translated to HTTP status here: ``ValidationError``
-> 422, ``NotFoundError`` -> 404, ``ConflictError`` -> 409. The service stays
HTTP-agnostic so the Phase-4 tool path can reuse it and render ``{"error": ...}``
instead.

Rate limiting: explicit ``@limiter.limit`` per route (LESSONS.md — middleware-only
limits skip router-mounted routes; handler takes ``request: Request``).
"""

from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import DEFAULT_RATE_LIMIT, limiter
from app.deps import get_current_user, get_db
from app.schemas.workout import (
    HistoryEntryResponse,
    LogSetRequest,
    SessionResponse,
    SetEntryResponse,
    StartSessionRequest,
)
from app.services import workout_service
from app.services.errors import ConflictError, NotFoundError, ValidationError

router = APIRouter(prefix="/workouts", tags=["workouts"])


@router.get("/sessions", response_model=list[SessionResponse])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def list_sessions(
    request: Request,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[SessionResponse]:
    """List the user's sessions, most-recent-first."""
    sessions = await workout_service.list_sessions(db, user_id)
    return [SessionResponse.model_validate(s) for s in sessions]


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def start_session(
    request: Request,
    payload: StartSessionRequest,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Start a new open session (manual "start workout").

    Returns 409 if the user already has an open session (at-most-one
    invariant).
    """
    try:
        session = await workout_service.start_session(
            db, user_id, program_id=payload.program_id, notes=payload.notes
        )
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return SessionResponse.model_validate(session)


@router.post("/sessions/{session_id}/finish", response_model=SessionResponse)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def finish_session(
    request: Request,
    session_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Finish the user's session; 404 if it isn't theirs (or doesn't exist)."""
    try:
        session = await workout_service.finish_session(db, user_id, session_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return SessionResponse.model_validate(session)


@router.post("/sets", response_model=SetEntryResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def log_set(
    request: Request,
    payload: LogSetRequest,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SetEntryResponse:
    """Log a set — same ``workout_service.log_set`` the ``log_set`` tool calls.

    Session resolution and ``set_number`` are computed server-side. Field
    validation runs in-service; an unknown ``exercise_id`` returns a
    structured 404, never a 500.
    """
    try:
        entry = await workout_service.log_set(
            db,
            user_id,
            exercise_id=payload.exercise_id,
            weight=payload.weight,
            reps=payload.reps,
            rir=payload.rir,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    return SetEntryResponse.model_validate(entry)


@router.get("/history", response_model=list[HistoryEntryResponse])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_history(
    request: Request,
    exercise_id: uuid.UUID | None = Query(default=None),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[HistoryEntryResponse]:
    """Query the user's logged sets (same service as ``get_workout_history``)."""
    rows = await workout_service.get_history(
        db,
        user_id,
        exercise_id=exercise_id,
        start_date=start_date,
        end_date=end_date,
    )
    return [
        HistoryEntryResponse(
            id=entry.id,
            session_id=entry.session_id,
            exercise_id=entry.exercise_id,
            exercise_name=name,
            set_number=entry.set_number,
            weight=entry.weight,
            reps=entry.reps,
            rir=entry.rir,
            created_at=entry.created_at,
        )
        for (entry, name) in rows
    ]
