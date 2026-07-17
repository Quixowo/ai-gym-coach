from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_title: Mapped[str]  # e.g. "Progressive Overload Principles"
    category: Mapped[str]  # "training" | "nutrition" | "injury_prevention"
    chunk_text: Mapped[str]
    chunk_index: Mapped[int]  # position within the source document
    source_citation: Mapped[str | None]  # populated for injury_prevention chunks
    # voyage-4-lite default output dimension.
    embedding: Mapped[list[float]] = mapped_column(Vector(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        # HNSW index, cosine distance. Voyage embeddings are unit-normalized, so
        # cosine and dot-product ranking are equivalent — cosine used for
        # readability.
        Index(
            "ix_knowledge_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
