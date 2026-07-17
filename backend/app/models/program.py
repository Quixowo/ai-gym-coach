from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Program(Base):
    __tablename__ = "programs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str]  # e.g. "Push Day", "Week 3 - Leg Day"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProgramExercise(Base):
    __tablename__ = "program_exercises"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    program_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("programs.id"), index=True)
    exercise_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("exercises.id"))
    order_index: Mapped[int]  # display/execution order within the program
    target_sets: Mapped[int | None]
    target_reps: Mapped[int | None]
    target_rir: Mapped[float | None]
    # Needed so update_program's load-jump guardrail has a week-over-week
    # comparison baseline.
    target_weight: Mapped[float | None]
