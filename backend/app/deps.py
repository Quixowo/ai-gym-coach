from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

from fastapi import Cookie, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import TokenError, decode_token
from app.db.session import async_session_maker


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped async DB session.

    The session is closed when the request finishes.
    """
    async with async_session_maker() as session:
        yield session


async def get_current_user(access_token: str | None = Cookie(None)) -> uuid.UUID:
    """Resolve the authenticated user's ID from the access-token cookie (spec §6.3).

    Decodes + verifies the ``access_token`` httpOnly cookie, requiring a valid
    signature, unexpired ``exp``, and ``type == "access"`` (a refresh token
    presented here is rejected). Returns the user UUID from the ``sub`` claim.

    Raises HTTP 401 on any failure — missing cookie, malformed/invalid token,
    expired token, or wrong token type — so REST endpoints and (later) the agent
    orchestrator share a single verified-user resolution path. This is the
    server-side injection point behind CLAUDE.md rule 2: user identity comes from
    here, never from client- or LLM-supplied input.
    """
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    try:
        payload = decode_token(access_token, expected_type="access")
        return uuid.UUID(payload["sub"])
    except (TokenError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc
