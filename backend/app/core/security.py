"""Password hashing and JWT token handling.

Two concerns live here, both deliberately hand-rolled (no ``fastapi-users`` /
auth framework — spec §6.1):

- **Password hashing** via Argon2id (``argon2-cffi``), the current recommended
  default over bcrypt (spec §6.2). The plaintext password is never logged or
  persisted — only the hash returned by ``hash_password`` is stored.
- **JWT** encode/decode via ``PyJWT`` (HS256, ``settings.JWT_SECRET``). Every
  token carries a ``type`` claim (``"access"`` | ``"refresh"``) in addition to
  ``sub`` / ``exp`` / ``iat``. ``decode_token`` verifies signature, expiry, *and*
  the expected ``type`` — a refresh token must never be accepted where an access
  token is required, and vice versa.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Literal

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from app.core.config import settings

_JWT_ALGORITHM = "HS256"

TokenType = Literal["access", "refresh"]

# Single hasher instance — Argon2id is argon2-cffi's default type, so no override
# is needed. Tuning parameters (time/memory cost) are left at library defaults.
_password_hasher = PasswordHasher()


class TokenError(Exception):
    """Raised when a JWT is missing, malformed, expired, or the wrong type.

    Callers (e.g. ``get_current_user``) translate this into an HTTP 401.
    """


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #
def hash_password(plaintext: str) -> str:
    """Return an Argon2id hash of ``plaintext`` suitable for storage."""
    return _password_hasher.hash(plaintext)


def verify_password(plaintext: str, hashed: str) -> bool:
    """Return True iff ``plaintext`` matches the stored Argon2 ``hashed`` value.

    Returns False (rather than raising) on any mismatch or malformed stored hash,
    so callers can treat verification as a simple boolean without leaking *why*
    it failed — supports the identical-401 no-enumeration behavior at login.
    """
    try:
        return _password_hasher.verify(hashed, plaintext)
    except (VerifyMismatchError, InvalidHashError):
        return False


# --------------------------------------------------------------------------- #
# JWT
# --------------------------------------------------------------------------- #
def _create_token(user_id: uuid.UUID | str, token_type: TokenType, expires_delta: timedelta) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=_JWT_ALGORITHM)


def create_access_token(user_id: uuid.UUID | str) -> str:
    """Mint a short-lived access token (``type="access"``, spec §6.1)."""
    return _create_token(
        user_id,
        "access",
        timedelta(minutes=settings.JWT_ACCESS_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: uuid.UUID | str) -> str:
    """Mint a long-lived refresh token (``type="refresh"``, spec §6.1)."""
    return _create_token(
        user_id,
        "refresh",
        timedelta(days=settings.JWT_REFRESH_EXPIRE_DAYS),
    )


def decode_token(token: str, expected_type: TokenType) -> dict:
    """Decode + fully verify a JWT, returning its payload.

    Verifies signature and expiry (via PyJWT) *and* that the ``type`` claim
    equals ``expected_type``. This last check is what prevents a refresh token
    from being replayed as an access token (or vice versa). Any failure raises
    :class:`TokenError`; callers map that to HTTP 401.
    """
    if not token:
        raise TokenError("missing token")
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except jwt.PyJWTError as exc:  # expired, bad signature, malformed, etc.
        raise TokenError(str(exc)) from exc

    if payload.get("type") != expected_type:
        raise TokenError(f"expected token type {expected_type!r}, got {payload.get('type')!r}")
    if "sub" not in payload:
        raise TokenError("token missing 'sub' claim")
    return payload
