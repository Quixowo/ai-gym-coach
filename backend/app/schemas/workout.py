"""Pydantic request/response models for the workout endpoints.

The ``LogSetRequest`` field constraints mirror the rules enforced in
``workout_service.log_set`` (via ``services.validation``). This duplication is
intentional: the schema gives the REST path an early, well-formed 422, while the
service copy is the real guard on the agent tool path, which bypasses this model.
Keep the two in sync — the service is the source of truth.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.validation import MAX_REPS, MAX_RIR, MIN_REPS, MIN_RIR, RIR_INCREMENT


class LogSetRequest(BaseModel):
    exercise_id: uuid.UUID
    weight: float = Field(ge=0, description="Load in the user's unit; must be >= 0")
    reps: int = Field(ge=MIN_REPS, le=MAX_REPS)
    rir: float | None = Field(default=None, ge=MIN_RIR, le=MAX_RIR)

    @field_validator("rir")
    @classmethod
    def _rir_increment(cls, v: float | None) -> float | None:
        if v is not None:
            remainder = round(v / RIR_INCREMENT, 6)
            if remainder != round(remainder):
                raise ValueError("rir must be in 0.5 increments")
        return v


class StartSessionRequest(BaseModel):
    program_id: uuid.UUID | None = None
    notes: str | None = None


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    program_id: uuid.UUID | None
    date: datetime
    status: str
    notes: str | None
    created_at: datetime


class SetEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    exercise_id: uuid.UUID
    set_number: int
    weight: float
    reps: int
    rir: float | None
    created_at: datetime


class HistoryEntryResponse(BaseModel):
    """A logged set plus its exercise display name (joined)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    exercise_id: uuid.UUID
    exercise_name: str
    set_number: int
    weight: float
    reps: int
    rir: float | None
    created_at: datetime
