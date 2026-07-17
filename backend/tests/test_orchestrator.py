"""Agent-loop tests — mocked Anthropic client, no live API.

Fakes the streaming surface the orchestrator uses:
``client.messages.stream(...)`` returns an async context manager that is itself an
async iterator of content-block-delta events and exposes
``await stream.get_final_message()``. A scripted client returns a queue of
pre-built "final messages" (text-only or containing tool_use blocks) so we can drive
the loop deterministically. Tools are exercised through the real DB (NullPool) so we
also confirm the ONE-user-message tool_result rule.
"""

from __future__ import annotations

import anthropic
import pytest

import app.agent.orchestrator as orchestrator_module
import app.db.session as db_session_module
from app.agent.events import (
    TextDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    TurnCompleteEvent,
)
from app.agent.orchestrator import MAX_ITERATIONS, run_agent_turn
from tests.conftest import test_session_maker as session_maker
from tests.helpers import create_db_user, first_exercise_id


# --------------------------------------------------------------------------- #
# Fake SDK surface
# --------------------------------------------------------------------------- #
class _Delta:
    def __init__(self, text: str) -> None:
        self.type = "text_delta"
        self.text = text


class _ContentBlockDeltaEvent:
    def __init__(self, text: str) -> None:
        self.type = "content_block_delta"
        self.delta = _Delta(text)


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    def __init__(self, tool_id: str, name: str, tool_input: dict) -> None:
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = tool_input


class _FinalMessage:
    def __init__(self, content: list) -> None:
        self.content = content


class _FakeStream:
    """Async context manager + async iterator standing in for the SDK stream."""

    def __init__(self, deltas: list[str], final_message: _FinalMessage) -> None:
        self._deltas = deltas
        self._final = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def __aiter__(self):
        for text in self._deltas:
            yield _ContentBlockDeltaEvent(text)

    async def get_final_message(self) -> _FinalMessage:
        return self._final


class _RaisingStream:
    """A stream whose iteration raises a typed SDK error mid-stream."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def __aiter__(self):
        raise self._exc
        yield  # pragma: no cover — makes this an async generator

    async def get_final_message(self):  # pragma: no cover — never reached
        raise AssertionError("should not be called")


class _FakeMessages:
    """Scripted ``messages`` object: pops the next scripted stream per .stream() call."""

    def __init__(self, streams: list) -> None:
        self._streams = list(streams)
        self.stream_calls: list[dict] = []

    def stream(self, **kwargs):
        # Snapshot messages at call time — the orchestrator reuses one mutable list
        # and appends to it after each call, so we must copy to see what was actually
        # sent on THIS call.
        snapshot = dict(kwargs)
        snapshot["messages"] = list(kwargs.get("messages", []))
        self.stream_calls.append(snapshot)
        if len(self._streams) == 1:
            return self._streams[0]
        return self._streams.pop(0)


class _FakeClient:
    def __init__(self, streams: list) -> None:
        self.messages = _FakeMessages(streams)


def _text_stream(text: str) -> _FakeStream:
    """A stream that emits ``text`` as one delta and finalizes to a text-only message."""
    return _FakeStream([text], _FinalMessage([_TextBlock(text)]))


def _tool_stream(tool_id: str, name: str, tool_input: dict, prefix: str = "") -> _FakeStream:
    """A stream that finalizes to a message containing one tool_use block."""
    deltas = [prefix] if prefix else []
    return _FakeStream(deltas, _FinalMessage([_ToolUseBlock(tool_id, name, tool_input)]))


@pytest.fixture(autouse=True)
def _mutating_handlers_use_test_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db_session_module, "async_session_maker", session_maker)


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    monkeypatch.setattr(orchestrator_module, "get_anthropic_client", lambda: client)


async def _collect(gen) -> list:
    return [event async for event in gen]


# --------------------------------------------------------------------------- #
# Text-only turn
# --------------------------------------------------------------------------- #
async def test_text_only_turn_streams_deltas_then_turn_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient([_text_stream("Hello there.")])
    _patch_client(monkeypatch, client)

    async with session_maker() as db:
        user_id = await create_db_user(db)
        events = await _collect(run_agent_turn("hi", [], user_id, db))

    text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert [e.text for e in text_events] == ["Hello there."]
    # No tool events on a text-only turn.
    assert not [e for e in events if isinstance(e, ToolCallStartedEvent)]
    # Ends with a single turn_complete reporting exactly 1 iteration.
    assert isinstance(events[-1], TurnCompleteEvent)
    assert events[-1].iterations == 1
    # Only one model call.
    assert len(client.messages.stream_calls) == 1


# --------------------------------------------------------------------------- #
# Tool-use turn: execute handler, feed result back in ONE user message, then finish
# --------------------------------------------------------------------------- #
async def test_tool_use_turn_executes_and_feeds_result_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)

        client = _FakeClient(
            [
                _tool_stream(
                    "tool_1", "log_set", {"exercise_id": str(ex), "weight": 100, "reps": 5}
                ),
                _text_stream("Logged your set."),
            ]
        )
        _patch_client(monkeypatch, client)

        events = await _collect(run_agent_turn("log 100x5", [], user_id, db))

    started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
    completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
    assert started and started[0].tools == ["log_set"]
    assert completed and completed[0].tool == "log_set"
    assert "success" in completed[0].result_summary

    # Two model calls: tool round, then final text.
    assert len(client.messages.stream_calls) == 2

    # ONE-user-message rule: the second call's messages must include a single user
    # message whose content is a list of tool_result blocks (all results for the
    # prior assistant turn in one message).
    second_call_messages = client.messages.stream_calls[1]["messages"]
    tool_result_messages = [
        m
        for m in second_call_messages
        if m["role"] == "user"
        and isinstance(m["content"], list)
        and all(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(tool_result_messages) == 1
    assert tool_result_messages[0]["content"][0]["tool_use_id"] == "tool_1"

    # Final text streamed.
    assert any(isinstance(e, TextDeltaEvent) and e.text == "Logged your set." for e in events)


# --------------------------------------------------------------------------- #
# Iteration cap: always-tool_use -> exactly MAX_ITERATIONS then the cap fallback
# --------------------------------------------------------------------------- #
async def test_iteration_cap_stops_at_max_and_emits_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)

        # A single stream reused forever -> the model "always" asks for a tool call.
        forever_tool = _tool_stream("loop", "get_workout_history", {"exercise_id": str(ex)})
        client = _FakeClient([forever_tool])  # len==1 -> reused every call
        _patch_client(monkeypatch, client)

        events = await _collect(run_agent_turn("loop forever", [], user_id, db))

    # Exactly MAX_ITERATIONS model calls.
    assert len(client.messages.stream_calls) == MAX_ITERATIONS
    # The cap fallback text is the last text delta.
    text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert text_events[-1].text == orchestrator_module.CAP_EXHAUSTED_MESSAGE
    # turn_complete reports the full MAX_ITERATIONS.
    assert isinstance(events[-1], TurnCompleteEvent)
    assert events[-1].iterations == MAX_ITERATIONS


# --------------------------------------------------------------------------- #
# SDK exception mid-stream -> error event, generator ends cleanly
# --------------------------------------------------------------------------- #
async def test_sdk_exception_yields_error_event_and_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exc = anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]
    client = _FakeClient([_RaisingStream(exc)])
    _patch_client(monkeypatch, client)

    async with session_maker() as db:
        user_id = await create_db_user(db)
        events = await _collect(run_agent_turn("hi", [], user_id, db))

    from app.agent.events import ErrorEvent

    assert any(isinstance(e, ErrorEvent) for e in events)
    # Generator ended cleanly (we reached here without an exception propagating).


# --------------------------------------------------------------------------- #
# History bounding
# --------------------------------------------------------------------------- #
async def test_history_bounded_to_last_20_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient([_text_stream("ok")])
    _patch_client(monkeypatch, client)

    long_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(50)
    ]
    async with session_maker() as db:
        user_id = await create_db_user(db)
        await _collect(run_agent_turn("newest", long_history, user_id, db))

    sent_messages = client.messages.stream_calls[0]["messages"]
    # 20 bounded history turns + the new user message.
    assert len(sent_messages) == orchestrator_module.MAX_HISTORY_TURNS + 1
    assert sent_messages[-1] == {"role": "user", "content": "newest"}
    # Oldest kept turn is turn 30 (turns 0..29 dropped).
    assert sent_messages[0]["content"] == "turn 30"
