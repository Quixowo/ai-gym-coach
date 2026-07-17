"""Agent chat endpoint — ``POST /chat``.

Streams the agent's response as Server-Sent Events. Flow:

1. Auth via ``get_current_user`` (the JWT-verified user is the only source of
   identity).
2. Run the injury red-flag classifier **first**. On a flag, stream the fixed
   redirect string and a ``turn_complete``, then stop — the orchestrator is never
   called. This puts one blocking Haiku call on the critical path of every
   message, an accepted safety-for-latency tradeoff.
3. Otherwise delegate to ``run_agent_turn``, forwarding its events.
4. After the response has fully streamed, kick off the episodic-memory pipeline as a
   background task (only when the request carried a ``conversation_id`` and the turn
   produced assistant text).

The frontend still replays conversation history with every request and the
backend still persists NO transcripts. What the backend now *does* persist is derived
memory: durable, third-person observations extracted from each turn and periodically
consolidated into ``user_memories`` (see ``app.services.memory_service``). ``history``
is untrusted conversational content — every tool still injects ``current_user_id``
server-side regardless of what it claims.

Rate limiting: explicit ``@limiter.limit(CHAT_RATE_LIMIT)`` (20/hour) — stricter
than the general tier because each message can fan out into several Claude calls.
The decorator is required (middleware-only limits skip
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
from starlette.background import BackgroundTask

import app.db.session as db_session
from app.agent.classifier import FIXED_INJURY_REDIRECT_RESPONSE, classify_acute_injury
from app.agent.events import ErrorEvent, TextDeltaEvent, TurnCompleteEvent
from app.agent.orchestrator import run_agent_turn
from app.core.logging import get_logger
from app.core.rate_limit import CHAT_RATE_LIMIT, limiter
from app.deps import get_current_user, get_db
from app.services.memory_service import process_turn

router = APIRouter(tags=["chat"])
log = get_logger(__name__)


class ChatMessage(BaseModel):
    """One prior conversational turn replayed by the frontend.

    ``role`` is constrained to ``user``/``assistant`` — a ``system`` or ``tool`` role
    injection attempt in history is rejected at validation (422). ``content`` is the
    turn text; it's untrusted conversational content, never privileged instruction.
    """

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    # Frontend-generated per-chat UUID (crypto.randomUUID() at mount). Groups this
    # turn's memory observations with the rest of the same chat. Optional: when absent
    # the memory pipeline is skipped entirely (backward compatible). Only ever used to
    # group the caller's OWN observations — never to scope access (that is always the
    # JWT-derived user_id), so a forged value can at worst distort this user's counting.
    conversation_id: uuid.UUID | None = None


def _sse_frame(payload: dict) -> str:
    """Serialize one event dict to an SSE ``data:`` frame."""
    return f"data: {json.dumps(payload, default=str)}\n\n"


async def _event_stream(
    message: str,
    history: list[dict],
    current_user_id: uuid.UUID,
    db: AsyncSession,
    reply_parts: list[str],
) -> AsyncGenerator[str]:
    """Yield SSE frames for one chat message (classifier short-circuit or full turn).

    Every streamed assistant text delta (including the fixed injury redirect and the
    cap-exhausted fallback) is appended to ``reply_parts`` so the post-stream memory
    task can read the full assistant reply once the generator finishes.

    Any exception inside this generator is converted to an ``error`` frame followed by
    a clean end of stream — an unhandled raise would abort the SSE response mid-flight
    and surface as a broken connection to the client.
    """
    try:
        if await classify_acute_injury(message):
            # Short-circuit: fixed redirect, no orchestrator call. Still a
            # normal SSE turn (text_delta + turn_complete), just not LLM-generated.
            reply_parts.append(FIXED_INJURY_REDIRECT_RESPONSE)
            yield _sse_frame(TextDeltaEvent(text=FIXED_INJURY_REDIRECT_RESPONSE).as_dict())
            yield _sse_frame(TurnCompleteEvent(iterations=0, total_latency_ms=0).as_dict())
            return

        async for event in run_agent_turn(message, history, current_user_id, db):
            if isinstance(event, TextDeltaEvent):
                reply_parts.append(event.text)
            yield _sse_frame(event.as_dict())
    except Exception:  # noqa: BLE001 — turn any failure into a graceful error frame
        log.exception("chat_stream_error", extra={"user_id": str(current_user_id)})
        yield _sse_frame(ErrorEvent(message="Something went wrong. Please try again.").as_dict())


async def _run_memory_pipeline(
    current_user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    message: str,
    reply_parts: list[str],
) -> None:
    """Post-stream background task: extract + consolidate memory for this turn.

    Opens its OWN DB session from the app sessionmaker: FastAPI closes the request's
    ``get_db`` yield-dependency before background tasks run (FastAPI >=0.106), so that
    session is unusable here. Skips entirely when the turn produced no assistant text
    (e.g. a stream error). ``process_turn`` is itself non-raising, but the session is
    still scoped by ``async with`` so it's always cleaned up.
    """
    assistant_reply = "".join(reply_parts).strip()
    if not assistant_reply:
        return
    async with db_session.async_session_maker() as db:
        await process_turn(
            user_id=current_user_id,
            conversation_id=conversation_id,
            user_message=message,
            assistant_reply=assistant_reply,
            db=db,
        )


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
    reply_parts: list[str] = []

    # Attach the memory pipeline as a Starlette background task: it runs AFTER the SSE
    # body has been fully sent, so it neither delays the stream nor reads a half-built
    # reply. Skipped when no conversation_id was supplied (backward compatible).
    background: BackgroundTask | None = None
    if payload.conversation_id is not None:
        background = BackgroundTask(
            _run_memory_pipeline,
            current_user_id,
            payload.conversation_id,
            payload.message,
            reply_parts,
        )

    return StreamingResponse(
        _event_stream(payload.message, history, current_user_id, db, reply_parts),
        media_type="text/event-stream",
        background=background,
    )
