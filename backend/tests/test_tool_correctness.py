"""Tool-call correctness eval — recorded fixtures, no live API.

Each ``claude_responses/tc_*.json`` fixture is a real ``run_agent_turn`` recorded
once against live Sonnet (see ``tests/fixtures/record_fixtures.py``). Here the REAL
orchestrator replays those streamed turns through :class:`FakeAnthropicClient` and we
assert the agent invoked the expected tool(s) with correct arguments — including two
behaviours called out specifically: the 10% load-jump cap must *reject* the
"+50 lbs squat" update (a structured-error tool_result, not just a tool call), and
the "log 3 sets" turn must produce the correct ``set_entries`` rows.

Recorded exercise UUIDs are rewritten to this DB's ids via the fixture ``id_map``
(recorded-id -> exercise NAME) + :func:`rewrite_ids`, so the replay references rows
that exist here (essential in CI, whose catalog is seeded with fresh uuids). Mutating
tools are pointed at the NullPool test session maker; a RAG tool call is kept
hermetic (fake Voyage + constant Anthropic) so no live provider is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

import app.agent.orchestrator as orchestrator_module
import app.db.session as db_session_module
import app.services.knowledge_service as knowledge_service_module
from app.agent.events import ToolCallCompletedEvent, ToolCallStartedEvent
from app.agent.orchestrator import run_agent_turn
from app.models.exercise import Exercise
from app.models.set_entry import SetEntry
from tests.conftest import test_session_maker as session_maker
from tests.fixtures._replay import FakeAnthropicClient, FakeVoyageClient, rewrite_ids
from tests.fixtures._scenario import apply_setup, name_to_id_map
from tests.helpers import create_db_user

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "claude_responses"
TOOL_FIXTURES = sorted(FIXTURES_DIR.glob("tc_*.json"))


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Hermetic RAG stand-in — a search_knowledge_base call in a recorded turn must not
# reach live Voyage/Anthropic. Retrieval stays real; synthesis/verdict are constant.
# --------------------------------------------------------------------------- #
class _ConstResponse:
    def __init__(self, text: str) -> None:
        self.content = [type("_B", (), {"text": text})()]


class _ConstMessages:
    async def create(self, **kwargs):
        return _ConstResponse("GROUNDED")


class _ConstAnthropic:
    def __init__(self) -> None:
        self.messages = _ConstMessages()


@pytest.fixture(autouse=True)
def _mutating_handlers_use_test_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating tools open their own session — point it at the NullPool test maker."""
    monkeypatch.setattr(db_session_module, "async_session_maker", session_maker)


def _patch_clients(monkeypatch: pytest.MonkeyPatch, fake: FakeAnthropicClient) -> None:
    monkeypatch.setattr(orchestrator_module, "get_anthropic_client", lambda: fake)
    # Keep any RAG tool call hermetic.
    monkeypatch.setattr(knowledge_service_module, "get_voyage_client", lambda: FakeVoyageClient())
    monkeypatch.setattr(knowledge_service_module, "get_anthropic_client", lambda: _ConstAnthropic())


async def _collect(gen) -> list:
    return [event async for event in gen]


@pytest.mark.parametrize("fixture_path", TOOL_FIXTURES, ids=lambda p: p.stem)
async def test_tool_correctness(monkeypatch: pytest.MonkeyPatch, fixture_path: Path) -> None:
    fixture = _load(fixture_path)

    async with session_maker() as db:
        user_id = await create_db_user(db)
        await apply_setup(db, user_id, fixture["setup"])

        # Rewrite recorded exercise UUIDs -> this DB's ids via id_map (recorded->name).
        names = await name_to_id_map(db)
        resolved = {rec_id: names[name] for rec_id, name in fixture["id_map"].items()}
        stream_iterations = rewrite_ids(fixture["stream_iterations"], resolved)

        fake = FakeAnthropicClient(stream_iterations=stream_iterations)
        _patch_clients(monkeypatch, fake)

        events = await _collect(run_agent_turn(fixture["user_message"], [], user_id, db))

        completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
        started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
        invoked = [e.tool for e in completed]

        # The model chose the right tools...
        for tool in fixture["must_call"]:
            assert tool in invoked, f"{fixture['id']}: expected {tool}, got {invoked}"
        for tool in fixture["must_not_call"]:
            assert tool not in invoked, f"{fixture['id']}: {tool} should not be called"

        # ...and the fake replayed exactly as many stream() calls as were recorded.
        assert len(fake.messages.stream_calls) == len(stream_iterations)

        checks = fixture["checks"]

        # The load-jump cap must REJECT (structured error tool_result), not just
        # be called.
        if checks.get("cap_rejected"):
            cap_events = [e for e in completed if e.tool == "update_program"]
            assert cap_events, "update_program was not invoked"
            assert all("10% safety cap" in e.result_summary for e in cap_events)
            assert all(e.result_summary.startswith("error:") for e in cap_events)

        # The log-sets turn must produce the correct set_entries rows.
        if "logged_sets" in checks:
            spec = checks["logged_sets"]
            exercise_id = names[spec["exercise_name"]]
            rows = (
                (
                    await db.execute(
                        select(SetEntry).where(
                            SetEntry.user_id == user_id,
                            SetEntry.exercise_id == exercise_id,
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(rows) == spec["count"]
            assert all(r.weight == spec["weight"] and r.reps == spec["reps"] for r in rows)
            # set_number is computed server-side, never taken from the model.
            assert sorted(r.set_number for r in rows) == list(range(1, spec["count"] + 1))

        # Sanity: a tool_call_started precedes every completed tool call.
        assert len(started) >= 1 if invoked else True


def test_all_six_scenarios_present() -> None:
    """The suite covers 4 representatives + 2 more (6 recorded turns)."""
    assert len(TOOL_FIXTURES) == 6


def test_recorded_turns_matched_intent() -> None:
    """Every recorded turn invoked the intended tools when captured live (metric e)."""
    for path in TOOL_FIXTURES:
        fixture = _load(path)
        assert fixture["matched_intent"], f"{fixture['id']} did not match intent at record time"


async def test_seeded_catalog_present() -> None:
    """Guard: the id_map rewrite relies on the seeded exercise catalog existing."""
    async with session_maker() as db:
        total = (await db.execute(select(func.count(Exercise.id)))).scalar_one()
    assert total >= 50
