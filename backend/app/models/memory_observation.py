from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class MemoryObservation(Base):
    """One durable user-fact extracted from a single chat turn (episodic memory pipeline).

    Raw pipeline state, never shown to the agent directly. Observations accumulate
    across conversations; once a ``(user_id, topic_key)`` has been seen in enough
    DISTINCT conversations, they are consolidated into a ``user_memories`` row — which
    is what the agent actually reads. ``conversation_id`` is the frontend-generated
    per-chat UUID and is deliberately NOT a foreign key: this project has no
    conversations table (chat is otherwise stateless server-side). ``user_id`` is
    denormalized so every access-control query filters with a single
    ``WHERE user_id = :uid`` (application-level access control); it always comes from
    the verified JWT, never from model output (CLAUDE.md rule 2).
    """

    __tablename__ = "memory_observations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # Frontend-generated per-chat UUID; intentionally not a FK (no conversations table).
    conversation_id: Mapped[uuid.UUID]
    category: Mapped[str]  # one of the fixed memory categories (validated in code)
    topic_key: Mapped[str]  # short lowercase snake_case subject key
    content: Mapped[str]  # third-person declarative fact; capped at 300 chars in code
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        # Threshold counting and consolidation both scan by (user, topic_key).
        Index("ix_memory_observations_user_topic", "user_id", "topic_key"),
    )
