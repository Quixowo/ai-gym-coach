"""Pydantic response model for the memory endpoints (episodic memory pipeline).

One schema: ``GET``/``DELETE /memories`` never accept a memory body (there's no
create/update via the API — memories are only ever written by
``app.services.memory_service``), so there's nothing to mirror here the way
``workout.py``/``program.py`` mirror service-layer write constraints.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    category: str
    topic_key: str
    content: str
    source_chat_count: int
    created_at: datetime
    updated_at: datetime
