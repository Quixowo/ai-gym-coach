from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SetEntry(Base):
    __tablename__ = "set_entries"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workout_sessions.id"), index=True)
    # Denormalized from workout_sessions.user_id: lets every
    # access-control query filter with a single WHERE set_entries.user_id = :uid,
    # no join required — making the filter structural, not remembered.
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    exercise_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("exercises.id"), index=True)
    set_number: Mapped[int]
    weight: Mapped[float]  # lbs or kg — single unit in v1 (per user/env setting)
    reps: Mapped[int]
    rir: Mapped[float | None]  # 0-4+, 0.5 increments allowed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        # Exact access pattern for get_workout_history and analyze_progression.
        Index(
            "ix_set_entries_user_exercise_created",
            "user_id",
            "exercise_id",
            "created_at",
        ),
    )
