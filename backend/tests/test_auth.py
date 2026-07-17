"""Auth endpoint tests.

Covers the full session lifecycle plus the security-relevant negative paths:
no user enumeration on login, token-type enforcement (a refresh token must not
work as an access token), and rejection of expired/garbage tokens.

Uses the shared session-scoped ``client`` fixture (NullPool DB engine — see
conftest). Emails carry a uuid suffix so reruns never collide on the unique
constraint. Rate-limit state is reset before each test so the app-wide
60/minute default can't make these flaky across reruns within a minute.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.rate_limit import limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    """Clear the Redis-backed limiter buckets before each test."""
    limiter.reset()


def _unique_email() -> str:
    return f"user_{uuid.uuid4().hex}@example.com"


def _register_payload(email: str | None = None, password: str = "password123") -> dict:
    return {
        "email": email or _unique_email(),
        "password": password,
        "display_name": "Test Lifter",
        "experience_level": "intermediate",
        "primary_goal": "hypertrophy",
        "injury_notes": None,
    }


def test_happy_path_full_lifecycle(client: TestClient) -> None:
    """register -> cookies set -> /me 200 -> refresh -> logout -> /me 401."""
    email = _unique_email()
    payload = _register_payload(email=email)

    reg = client.post("/auth/register", json=payload)
    assert reg.status_code == 201
    body = reg.json()
    assert body["email"] == email
    assert body["display_name"] == "Test Lifter"
    assert body["experience_level"] == "intermediate"
    assert body["primary_goal"] == "hypertrophy"
    assert "hashed_password" not in body  # never serialized
    # Both cookies were set on the register response.
    assert "access_token" in client.cookies
    assert "refresh_token" in client.cookies

    # /me reflects the registered profile.
    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == email

    # Refresh issues a fresh access cookie.
    old_access = client.cookies.get("access_token")
    refresh = client.post("/auth/refresh")
    assert refresh.status_code == 204
    assert "access_token" in refresh.cookies
    # New cookie is now in the jar (value may differ; iat/exp advance).
    assert "access_token" in client.cookies

    # Still authenticated after refresh.
    assert client.get("/auth/me").status_code == 200

    # Logout clears both cookies.
    logout = client.post("/auth/logout")
    assert logout.status_code == 204
    assert client.cookies.get("access_token") in (None, "")
    assert client.cookies.get("refresh_token") in (None, "")

    # No cookie -> 401.
    assert client.get("/auth/me").status_code == 401

    # Keep the linter honest about the captured value.
    assert old_access is not None


def test_duplicate_email_conflicts(client: TestClient) -> None:
    payload = _register_payload()
    first = client.post("/auth/register", json=payload)
    assert first.status_code == 201
    client.cookies.clear()

    dup = client.post("/auth/register", json=payload)
    assert dup.status_code == 409


def test_login_wrong_password_401(client: TestClient) -> None:
    email = _unique_email()
    client.post("/auth/register", json=_register_payload(email=email, password="rightpassword"))
    client.cookies.clear()

    resp = client.post("/auth/login", json={"email": email, "password": "wrongpassword"})
    assert resp.status_code == 401
    wrong_pw_body = resp.json()

    # Unknown email must return the *same* 401 body (no user enumeration).
    unknown = client.post(
        "/auth/login",
        json={"email": _unique_email(), "password": "whatever12"},
    )
    assert unknown.status_code == 401
    assert unknown.json() == wrong_pw_body


def test_login_success_sets_cookies(client: TestClient) -> None:
    email = _unique_email()
    client.post("/auth/register", json=_register_payload(email=email, password="rightpassword"))
    client.cookies.clear()

    resp = client.post("/auth/login", json={"email": email, "password": "rightpassword"})
    assert resp.status_code == 200
    assert resp.json()["email"] == email
    assert "access_token" in resp.cookies
    assert "refresh_token" in resp.cookies


def test_me_without_cookie_401(client: TestClient) -> None:
    client.cookies.clear()
    assert client.get("/auth/me").status_code == 401


def test_me_with_garbage_token_401(client: TestClient) -> None:
    client.cookies.clear()
    resp = client.get("/auth/me", cookies={"access_token": "not-a-real-jwt"})
    assert resp.status_code == 401


def test_refresh_token_rejected_as_access_cookie_401(client: TestClient) -> None:
    """A valid *refresh* token presented as the access cookie must be rejected."""
    client.cookies.clear()
    user_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    refresh = jwt.encode(
        {
            "sub": user_id,
            "type": "refresh",  # wrong type for the access-cookie slot
            "iat": now,
            "exp": now + timedelta(days=7),
        },
        settings.JWT_SECRET,
        algorithm="HS256",
    )
    resp = client.get("/auth/me", cookies={"access_token": refresh})
    assert resp.status_code == 401


def test_expired_access_token_401(client: TestClient) -> None:
    """An access token whose exp is in the past must be rejected."""
    client.cookies.clear()
    user_id = str(uuid.uuid4())
    past = datetime.now(UTC) - timedelta(hours=1)
    expired = jwt.encode(
        {
            "sub": user_id,
            "type": "access",
            "iat": past - timedelta(minutes=15),
            "exp": past,  # already expired
        },
        settings.JWT_SECRET,
        algorithm="HS256",
    )
    resp = client.get("/auth/me", cookies={"access_token": expired})
    assert resp.status_code == 401


def test_access_token_rejected_as_refresh_401(client: TestClient) -> None:
    """Symmetric to the refresh-as-access case: access token can't refresh."""
    client.cookies.clear()
    user_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    access = jwt.encode(
        {
            "sub": user_id,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=15),
        },
        settings.JWT_SECRET,
        algorithm="HS256",
    )
    resp = client.post("/auth/refresh", cookies={"refresh_token": access})
    assert resp.status_code == 401
