from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class UserMemory(Base):
    """A consolidated, durable memory injected into the agent's prompt (episodic memory).

    Produced by consolidating the ``memory_observations`` for one ``(user_id,
    topic_key)`` once that topic has appeared in enough distinct conversations. Exactly
    one row per ``(user_id, topic_key)`` — enforced by the unique constraint and
    maintained via upsert, so two turns consolidating at once can't create duplicates
    (last write wins; recency wins on contradictions during synthesis). ``user_id`` is
    denormalized and always sourced from the verified JWT (application-level access
    control; CLAUDE.md rule 2). The stored text is a third-person fact, never verbatim
    user input, and is never used to build tool arguments.
    """

    __tablename__ = "user_memories"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    category: Mapped[str]
    topic_key: Mapped[str]
    content: Mapped[str]
    # Distinct conversations that contributed observations as of the last consolidation.
    source_chat_count: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (UniqueConstraint("user_id", "topic_key", name="uq_user_memories_user_topic"),)
