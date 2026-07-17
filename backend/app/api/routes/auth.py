"""Authentication endpoints.

Implements register / login / refresh / logout, plus a pragmatic ``GET /auth/me``
(see the ``/auth/me`` note below). Both the access and refresh tokens are set as
httpOnly cookies with environment-driven ``SameSite`` / ``Secure`` (``Lax`` +
insecure for local dev, ``None`` + ``Secure`` for the deployed cross-origin
setup). Tokens are never returned in the response body.

Login uses an identical 401 for "unknown email" and "wrong password" so an
attacker can't enumerate registered accounts.

Rate limiting: each route carries an explicit ``@limiter.limit(DEFAULT_RATE_LIMIT)``
decorator rather than relying solely on the app-wide ``SlowAPIMiddleware``. In
this FastAPI version, ``include_router`` nests routes under a router object that
slowapi's middleware route-lookup can't resolve, so middleware-only limits would
silently skip every ``/auth/*`` route. The decorator enforces the limit at the
handler wrapper, which works regardless of nesting.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.core.rate_limit import DEFAULT_RATE_LIMIT, limiter
from app.core.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.deps import get_current_user, get_db
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])
log = get_logger(__name__)

_ACCESS_COOKIE = "access_token"
_REFRESH_COOKIE = "refresh_token"

# Cookie lifetimes (seconds) mirror the token expiries from settings.
_ACCESS_MAX_AGE = settings.JWT_ACCESS_EXPIRE_MINUTES * 60
_REFRESH_MAX_AGE = settings.JWT_REFRESH_EXPIRE_DAYS * 24 * 60 * 60


def _set_access_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=_ACCESS_COOKIE,
        value=token,
        max_age=_ACCESS_MAX_AGE,
        httponly=True,
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE,
        path="/",
    )


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=_REFRESH_COOKIE,
        value=token,
        max_age=_REFRESH_MAX_AGE,
        httponly=True,
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE,
        path="/",
    )


def _issue_session_cookies(response: Response, user_id: uuid.UUID) -> None:
    """Set both access and refresh cookies for a freshly authenticated user."""
    _set_access_cookie(response, create_access_token(user_id))
    _set_refresh_cookie(response, create_refresh_token(user_id))


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def register(
    request: Request,
    payload: RegisterRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Create a user, hash the password, and issue both session cookies.

    Returns 409 if the email is already registered.
    """
    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        display_name=payload.display_name,
        experience_level=payload.experience_level,
        primary_goal=payload.primary_goal,
        injury_notes=payload.injury_notes,
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        ) from exc
    await db.refresh(user)

    _issue_session_cookies(response, user.id)
    # ``extra`` is scrubbed by the redaction filter; no password value passes.
    log.info("user_registered", extra={"user_id": str(user.id), "email": user.email})
    return user


@router.post("/login", response_model=UserResponse)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def login(
    request: Request,
    payload: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Verify credentials and issue both session cookies.

    Emits an identical 401 whether the email is unknown or the password is wrong
    — no user enumeration. A dummy verify against a throwaway hash is *not* used
    here; instead we short-circuit but return the same generic error.
    """
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(payload.password, user.hashed_password):
        log.info("login_failed", extra={"email": payload.email})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    _issue_session_cookies(response, user.id)
    log.info("user_logged_in", extra={"user_id": str(user.id), "email": user.email})
    return user


@router.post("/refresh", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def refresh(
    request: Request,
    response: Response,
    refresh_token: str | None = Cookie(None),
) -> Response:
    """Mint a new access cookie from a valid refresh cookie.

    Requires a ``type == "refresh"`` token — an access token presented here is
    rejected 401. The refresh cookie itself is left untouched (still valid until
    its own 7-day expiry); only a fresh access cookie is set.
    """
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token",
        )
    try:
        payload = decode_token(refresh_token, expected_type="refresh")
        user_id = uuid.UUID(payload["sub"])
    except (TokenError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        ) from exc

    _set_access_cookie(response, create_access_token(user_id))
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def logout(request: Request, response: Response) -> Response:
    """Clear both session cookies."""
    # delete_cookie must match path (and samesite/secure attributes on some
    # browsers) for the clearing Set-Cookie to actually replace the originals.
    response.delete_cookie(
        _ACCESS_COOKIE, path="/", samesite=settings.COOKIE_SAMESITE, secure=settings.COOKIE_SECURE
    )
    response.delete_cookie(
        _REFRESH_COOKIE, path="/", samesite=settings.COOKIE_SAMESITE, secure=settings.COOKIE_SECURE
    )
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/me", response_model=UserResponse)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def me(
    request: Request,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Return the authenticated user's profile.

    Not part of the original planned endpoint list, but the frontend
    AuthContext needs it to restore login state on reload, and it serves as the
    canonical protected endpoint for auth tests. Fully behind
    ``get_current_user`` (access-cookie required, 401 otherwise).
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        # Token was valid but the user no longer exists (deleted mid-session).
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user
