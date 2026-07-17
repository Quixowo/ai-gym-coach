from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class WorkoutSession(Base):
    __tablename__ = "workout_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    # Nullable — freeform sessions (no program) are allowed.
    program_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("programs.id"))
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # "open" | "finished" — drives log_set's find-or-create.
    status: Mapped[str] = mapped_column(default="open")
    notes: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
