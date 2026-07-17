"""Pydantic response models for the exercise catalog endpoints."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict


class ExerciseResponse(BaseModel):
    # Built straight from the ORM Exercise row.
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    primary_muscle_group: str
    movement_pattern: str
    equipment: str


class ExerciseMatch(BaseModel):
    """One fuzzy-search candidate. Mirrors the service dict shape."""

    exercise_id: str
    name: str
    score: float


class ExerciseSearchResponse(BaseModel):
    matches: list[ExerciseMatch]
