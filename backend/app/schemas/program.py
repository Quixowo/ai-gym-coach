"""Pydantic request/response models for the program endpoints.

Target-field bounds (sets/reps/rir/weight) are mirrored here as ``Field``
constraints, matching what the service accepts. The 10% load-jump cap is NOT a
schema concern — it is a cross-exercise, prior-vs-requested comparison that
depends on stored state, so it lives in ``program_service.update_program``
where both the REST and Phase-4 tool paths share it.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.services.validation import MAX_REPS, MAX_RIR, MIN_RIR


class ProgramExerciseInput(BaseModel):
    exercise_id: uuid.UUID
    order_index: int | None = Field(default=None, ge=0)
    target_sets: int | None = Field(default=None, ge=1)
    target_reps: int | None = Field(default=None, ge=1, le=MAX_REPS)
    target_rir: float | None = Field(default=None, ge=MIN_RIR, le=MAX_RIR)
    target_weight: float | None = Field(default=None, ge=0)


class CreateProgramRequest(BaseModel):
    name: str = Field(min_length=1)
    exercises: list[ProgramExerciseInput] = Field(default_factory=list)


class UpdateProgramRequest(BaseModel):
    # Name optional on update; exercises replace the existing set wholesale.
    name: str | None = Field(default=None, min_length=1)
    exercises: list[ProgramExerciseInput] = Field(default_factory=list)


class ProgramExerciseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    exercise_id: uuid.UUID
    exercise_name: str
    order_index: int
    target_sets: int | None
    target_reps: int | None
    target_rir: float | None
    target_weight: float | None


class ProgramSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime


class ProgramDetailResponse(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime
    exercises: list[ProgramExerciseResponse]
