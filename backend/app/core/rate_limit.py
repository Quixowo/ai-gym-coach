"""Application rate limiting via slowapi, backed by Redis.

The limiter keys on the **authenticated user id** when a valid access-token
cookie is present, falling back to the client IP for anonymous requests. This
means the 60/min baseline is per-user for logged-in traffic (fairer, and not
defeatable by a shared NAT IP) while still covering unauthenticated endpoints
like ``/auth/register`` by IP.

Redis (``settings.REDIS_URL``) is the shared store so limits hold across
multiple backend workers/instances. Locally this is the docker ``redis:7``
service; in the deployed demo it's Upstash over its TCP endpoint.

``CHAT_RATE_LIMIT`` is defined here but applied in Phase 4 on the ``/chat``
endpoint — that route can fan out into several Claude calls per message, so it
carries the stricter 20/hour guard on the Anthropic bill.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.core.config import settings
from app.core.security import TokenError, decode_token

# Applied per-endpoint in Phase 4, not app-wide. Kept here so the two limits
# live in one place.
CHAT_RATE_LIMIT = "20/hour"

# App-wide baseline abuse protection.
DEFAULT_RATE_LIMIT = "60/minute"


def _user_id_or_ip(request: Request) -> str:
    """Rate-limit key: verified user id if authenticated, else client IP.

    Reads the ``access_token`` cookie and validates it exactly the way
    ``get_current_user`` does (signature + expiry + ``type == "access"``). An
    invalid/expired/wrong-type token simply falls through to IP keying rather
    than erroring — the endpoint's own auth dependency is what rejects the
    request; here we only need a stable bucket key.
    """
    token = request.cookies.get("access_token")
    if token:
        try:
            payload = decode_token(token, expected_type="access")
            return f"user:{payload['sub']}"
        except TokenError:
            pass
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(
    key_func=_user_id_or_ip,
    default_limits=[DEFAULT_RATE_LIMIT],
    storage_uri=settings.REDIS_URL,
)
