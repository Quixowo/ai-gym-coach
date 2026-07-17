"""Chat-route wiring for the memory background task (TestClient, no live API).

Formalizes the hand-verified wiring: the ``POST /chat`` route enqueues the memory
pipeline as a Starlette ``BackgroundTask`` that runs AFTER the SSE body is fully sent,
and only when a ``conversation_id`` was supplied. The classifier, orchestrator, and
``process_turn`` are all patched at the chat route's import boundary, so no Anthropic
call is made and we observe exactly what the route hands the
pipeline.

The background task opens its OWN session from
``app.db.session.async_session_maker`` (FastAPI closes the request's ``get_db`` yield-
dependency before background tasks run); that maker is repointed at the NullPool test
maker so the post-response session doesn't hit the closed-loop pooling bug.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

import app.api.routes.chat as chat_module
import app.db.session as db_session_module
from app.agent.events import TextDeltaEvent, TurnCompleteEvent
from app.core.rate_limit import limiter
from tests.conftest import test_session_maker
from tests.helpers import register_user


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    limiter.reset()


@pytest.fixture(autouse=True)
def _background_session_uses_test_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The memory task opens its own session — point it at the NullPool test maker."""
    monkeypatch.setattr(db_session_module, "async_session_maker", test_session_maker)


async def _no_flag(_message: str) -> bool:
    return False


def _two_delta_orchestrator(message, history, user_id, db):
    async def _gen():
        yield TextDeltaEvent(text="Warm up gradually, ")
        yield TextDeltaEvent(text="then work up in sets.")
        yield TurnCompleteEvent(iterations=1, total_latency_ms=5)

    return _gen()


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Patch classifier/orchestrator/process_turn; return a list recording pipeline calls."""
    calls: list[dict] = []

    async def _stub_process_turn(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(chat_module, "classify_acute_injury", _no_flag)
    monkeypatch.setattr(chat_module, "run_agent_turn", _two_delta_orchestrator)
    monkeypatch.setattr(chat_module, "process_turn", _stub_process_turn)
    return calls


# --------------------------------------------------------------------------- #
# No conversation_id -> pipeline never invoked (backward compatible)
# --------------------------------------------------------------------------- #
def test_no_conversation_id_skips_pipeline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_user(client)
    calls = _patch_pipeline(monkeypatch)

    resp = client.post("/chat", json={"message": "how should I warm up?"})
    assert resp.status_code == 200
    _ = resp.text  # fully drain the stream so any background task would have run

    assert calls == []


# --------------------------------------------------------------------------- #
# conversation_id -> pipeline invoked exactly once with the FULL concatenated reply
# --------------------------------------------------------------------------- #
def test_conversation_id_invokes_pipeline_once_after_stream(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = register_user(client)
    user_id = user["id"]
    calls = _patch_pipeline(monkeypatch)

    conversation_id = str(uuid.uuid4())
    resp = client.post(
        "/chat",
        json={"message": "how should I warm up?", "conversation_id": conversation_id},
    )
    assert resp.status_code == 200
    _ = resp.text  # background task runs after the body is exhausted

    assert len(calls) == 1
    call = calls[0]
    # Right identity (JWT-derived user) and the frontend-supplied conversation id.
    assert str(call["user_id"]) == user_id
    assert str(call["conversation_id"]) == conversation_id
    assert call["user_message"] == "how should I warm up?"
    # The FULL reply, concatenated from every streamed delta (runs after the stream).
    assert call["assistant_reply"] == "Warm up gradually, then work up in sets."


# --------------------------------------------------------------------------- #
# conversation_id but empty assistant reply -> pipeline skipped (no text to learn from)
# --------------------------------------------------------------------------- #
def test_empty_reply_skips_pipeline(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    register_user(client)
    calls: list[dict] = []

    async def _stub_process_turn(**kwargs):
        calls.append(kwargs)

    def _silent_orchestrator(message, history, user_id, db):
        async def _gen():
            # No text deltas — a turn that produced no assistant text.
            yield TurnCompleteEvent(iterations=1, total_latency_ms=5)

        return _gen()

    monkeypatch.setattr(chat_module, "classify_acute_injury", _no_flag)
    monkeypatch.setattr(chat_module, "run_agent_turn", _silent_orchestrator)
    monkeypatch.setattr(chat_module, "process_turn", _stub_process_turn)

    resp = client.post(
        "/chat",
        json={"message": "hey", "conversation_id": str(uuid.uuid4())},
    )
    assert resp.status_code == 200
    _ = resp.text

    assert calls == []
