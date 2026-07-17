"""Exercise catalog endpoints.

The catalog is global and read-only, so these read paths carry no ``user_id``
filter (the deliberate exception to application-level access control — there is
no user data here). Both endpoints are still behind ``get_current_user``: the
catalog is only exposed to authenticated users, and 401s without a cookie.

Rate limiting: each route carries an explicit ``@limiter.limit`` decorator — per
LESSONS.md, slowapi's middleware-only default limits silently skip
``include_router`` routes, so the decorator (handler needs ``request: Request``)
is what actually enforces the 60/min tier here.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import DEFAULT_RATE_LIMIT, limiter
from app.deps import get_current_user, get_db
from app.schemas.exercise import ExerciseResponse, ExerciseSearchResponse
from app.services import exercise_service

router = APIRouter(prefix="/exercises", tags=["exercises"])


@router.get("", response_model=list[ExerciseResponse])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def list_exercises(
    request: Request,
    _user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ExerciseResponse]:
    """List the full seeded exercise catalog (auth required)."""
    exercises = await exercise_service.list_exercises(db)
    return [ExerciseResponse.model_validate(e) for e in exercises]


@router.get("/search", response_model=ExerciseSearchResponse)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def search_exercises(
    request: Request,
    q: str = Query(..., description="Casual/ambiguous exercise name to resolve"),
    _user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ExerciseSearchResponse:
    """Fuzzy-search the catalog (same service as the ``search_exercises`` tool)."""
    matches = await exercise_service.search_exercises(db, q)
    return ExerciseSearchResponse(matches=matches)
