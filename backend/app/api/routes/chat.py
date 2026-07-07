"""Agent chat endpoint — ``POST /chat`` (spec §7.5, §10.1, §6.4, §11.2).

Streams the agent's response as Server-Sent Events. Flow (§7.5 ordering note):

1. Auth via ``get_current_user`` (the JWT-verified user is the only source of
   identity — CLAUDE.md rule 2).
2. Run the injury red-flag classifier **first**. On a flag, stream the fixed
   redirect string and a ``turn_complete``, then stop — the orchestrator is never
   called (§10.1). This puts one blocking Haiku call on the critical path of every
   message, an accepted safety-for-latency tradeoff (§7.5).
3. Otherwise delegate to ``run_agent_turn``, forwarding its events.

The frontend replays conversation history with every request (§7.5); the backend is
stateless between messages. ``history`` is untrusted conversational content — every
tool still injects ``current_user_id`` server-side regardless of what it claims.

Rate limiting: explicit ``@limiter.limit(CHAT_RATE_LIMIT)`` (20/hour) — stricter
than the general tier because each message can fan out into several Claude calls
(§6.4). Per LESSONS.md the decorator is required (middleware-only limits skip
router-mounted routes); the handler takes ``request: Request``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.classifier import FIXED_INJURY_REDIRECT_RESPONSE, classify_acute_injury
from app.agent.events import ErrorEvent, TextDeltaEvent, TurnCompleteEvent
from app.agent.orchestrator import run_agent_turn
from app.core.logging import get_logger
from app.core.rate_limit import CHAT_RATE_LIMIT, limiter
from app.deps import get_current_user, get_db

router = APIRouter(tags=["chat"])
log = get_logger(__name__)


class ChatMessage(BaseModel):
    """One prior conversational turn replayed by the frontend (spec §7.5).

    ``role`` is constrained to ``user``/``assistant`` — a ``system`` or ``tool`` role
    injection attempt in history is rejected at validation (422). ``content`` is the
    turn text; it's untrusted conversational content, never privileged instruction.
    """

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


def _sse_frame(payload: dict) -> str:
    """Serialize one event dict to an SSE ``data:`` frame."""
    return f"data: {json.dumps(payload, default=str)}\n\n"


async def _event_stream(
    message: str,
    history: list[dict],
    current_user_id: uuid.UUID,
    db: AsyncSession,
) -> AsyncGenerator[str]:
    """Yield SSE frames for one chat message (classifier short-circuit or full turn).

    Any exception inside this generator is converted to an ``error`` frame followed
    by a clean end of stream — an unhandled raise would abort the SSE response
    mid-flight and surface as a broken connection to the client.
    """
    try:
        if await classify_acute_injury(message):
            # Short-circuit: fixed redirect, no orchestrator call (§10.1). Still a
            # normal SSE turn (text_delta + turn_complete), just not LLM-generated.
            yield _sse_frame(TextDeltaEvent(text=FIXED_INJURY_REDIRECT_RESPONSE).as_dict())
            yield _sse_frame(TurnCompleteEvent(iterations=0, total_latency_ms=0).as_dict())
            return

        async for event in run_agent_turn(message, history, current_user_id, db):
            yield _sse_frame(event.as_dict())
    except Exception:  # noqa: BLE001 — turn any failure into a graceful error frame
        log.exception("chat_stream_error", extra={"user_id": str(current_user_id)})
        yield _sse_frame(ErrorEvent(message="Something went wrong. Please try again.").as_dict())


@router.post("/chat")
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(
    request: Request,
    payload: ChatRequest,
    current_user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream the agent's response to one chat message as Server-Sent Events."""
    history = [{"role": m.role, "content": m.content} for m in payload.history]
    return StreamingResponse(
        _event_stream(payload.message, history, current_user_id, db),
        media_type="text/event-stream",
    )
