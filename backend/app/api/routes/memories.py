"""User-memory inspection/control endpoints (episodic memory pipeline).

The episodic-memory pipeline (``app.services.memory_service``) writes consolidated
``user_memories`` rows in a background task after each chat turn (see
``app.api.routes.chat``). This module is the user's own control surface over that
derived data: ``GET /memories`` to inspect it, ``DELETE /memories/{id}`` to remove
one.

Both routes sit behind ``get_current_user`` exactly like ``workouts.py``/
``programs.py``, and every query filters ``WHERE user_id = :uid`` (application-level
access control — never call it RLS): a memory id that doesn't
exist, or exists but belongs to another user, 404s identically so ownership is never
leaked.

``GET`` deliberately does NOT apply ``settings.MEMORY_MAX_INJECTED`` — that cap only
bounds what gets injected into the agent's system prompt (a prompt-injection-surface
control); this endpoint is the user's inspection view and returns every row they own.

``DELETE`` removes the consolidated row AND every ``memory_observations`` row sharing
its ``(user_id, topic_key)``. Observations are retained after consolidation (as
provenance / re-consolidation input — see ``memory_service._maybe_consolidate``), so
deleting only the consolidated row would leave the memory to silently reappear the
next time the topic comes up in conversation.

Rate limiting: explicit ``@limiter.limit`` per route, matching ``workouts.py``
(slowapi's middleware-only limits silently skip router-mounted routes;
the handler needs a ``request: Request`` param).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import DEFAULT_RATE_LIMIT, limiter
from app.deps import get_current_user, get_db
from app.models.memory_observation import MemoryObservation
from app.models.user_memory import UserMemory
from app.schemas.memory import MemoryResponse

router = APIRouter(prefix="/memories", tags=["memories"])


@router.get("", response_model=list[MemoryResponse])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def list_memories(
    request: Request,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[MemoryResponse]:
    """List ALL of the user's consolidated memories, most-recently-updated first."""
    result = await db.execute(
        select(UserMemory)
        .where(UserMemory.user_id == user_id)
        .order_by(UserMemory.updated_at.desc())
    )
    memories = result.scalars().all()
    return [MemoryResponse.model_validate(m) for m in memories]


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def delete_memory(
    request: Request,
    memory_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete one consolidated memory and its topic's retained observations.

    404s for both a nonexistent id and another user's id (application-level access
    control — the two cases are indistinguishable to the caller on purpose).
    """
    result = await db.execute(
        select(UserMemory).where(UserMemory.id == memory_id, UserMemory.user_id == user_id)
    )
    memory = result.scalar_one_or_none()
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found")

    # Otherwise this topic's retained observations would re-consolidate the
    # "deleted" memory right back into existence on its next mention.
    await db.execute(
        delete(MemoryObservation).where(
            MemoryObservation.user_id == user_id,
            MemoryObservation.topic_key == memory.topic_key,
        )
    )
    await db.delete(memory)
    await db.commit()
