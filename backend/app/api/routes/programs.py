"""Program endpoints.

Behind ``get_current_user``; ``user_id`` is passed explicitly into
``program_service`` (application-level access control) so another
user's program 404s. ``PUT /programs/{id}`` is subject to the same 10% load-jump
cap as the ``update_program`` tool — enforced in the service, surfaced
here as 422 with the structured detail (exercise / prior / requested).

Service domain exceptions translate to HTTP: ``NotFoundError`` -> 404,
``ValidationError`` (incl. ``LoadJumpCapError``) -> 422. The service stays
HTTP-agnostic for Phase-4 tool reuse.

Rate limiting: explicit ``@limiter.limit`` per route (LESSONS.md; ``request:
Request`` required).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import DEFAULT_RATE_LIMIT, limiter
from app.deps import get_current_user, get_db
from app.models.program import Program, ProgramExercise
from app.schemas.program import (
    CreateProgramRequest,
    ProgramDetailResponse,
    ProgramExerciseResponse,
    ProgramSummaryResponse,
    UpdateProgramRequest,
)
from app.services import program_service
from app.services.errors import NotFoundError, ValidationError

router = APIRouter(prefix="/programs", tags=["programs"])


def _detail_response(
    program: Program, exercises: list[tuple[ProgramExercise, str]]
) -> ProgramDetailResponse:
    return ProgramDetailResponse(
        id=program.id,
        name=program.name,
        created_at=program.created_at,
        exercises=[
            ProgramExerciseResponse(
                id=pe.id,
                exercise_id=pe.exercise_id,
                exercise_name=name,
                order_index=pe.order_index,
                target_sets=pe.target_sets,
                target_reps=pe.target_reps,
                target_rir=pe.target_rir,
                target_weight=pe.target_weight,
            )
            for (pe, name) in exercises
        ],
    )


@router.get("", response_model=list[ProgramSummaryResponse])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def list_programs(
    request: Request,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ProgramSummaryResponse]:
    """List the user's programs, most-recent-first."""
    programs = await program_service.list_programs(db, user_id)
    return [ProgramSummaryResponse.model_validate(p) for p in programs]


@router.post("", response_model=ProgramDetailResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def create_program(
    request: Request,
    payload: CreateProgramRequest,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProgramDetailResponse:
    """Create a program (manual "save workout to repeat" or plan-ahead)."""
    try:
        program, exercises = await program_service.create_program(
            db,
            user_id,
            name=payload.name,
            exercises=[ex.model_dump() for ex in payload.exercises],
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _detail_response(program, exercises)


@router.get("/{program_id}", response_model=ProgramDetailResponse)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_program(
    request: Request,
    program_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProgramDetailResponse:
    """Program detail; 404 if not the user's."""
    try:
        program, exercises = await program_service.get_program_detail(db, user_id, program_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _detail_response(program, exercises)


@router.put("/{program_id}", response_model=ProgramDetailResponse)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def update_program(
    request: Request,
    program_id: uuid.UUID,
    payload: UpdateProgramRequest,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProgramDetailResponse:
    """Edit a program, subject to the 10% load-jump cap.

    A single over-cap ``target_weight`` rejects the whole update (422, detail
    names the exercise / prior / requested). 404 if the program isn't the user's.
    """
    try:
        program, exercises = await program_service.update_program(
            db,
            user_id,
            program_id,
            name=payload.name,
            exercises=[ex.model_dump() for ex in payload.exercises],
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValidationError as exc:
        # LoadJumpCapError is a ValidationError subclass -> also 422.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    return _detail_response(program, exercises)
