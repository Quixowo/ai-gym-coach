from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(unique=True, index=True)
    hashed_password: Mapped[str]
    display_name: Mapped[str]
    experience_level: Mapped[str]  # "beginner" | "intermediate" | "advanced"
    primary_goal: Mapped[str]  # "hypertrophy" | "strength" | "fat_loss" | "general"
    # Free text, user-authored profile context (not agent-written); surfaced in the
    # agent system prompt each session.
    injury_notes: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
