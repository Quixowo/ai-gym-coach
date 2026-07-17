"""Tool-handler tests — ``execute_tool`` dispatch + isolation contract.

No live API here (handlers make no LLM calls). Covers: right service called with the
*injected* user_id; malformed LLM input -> ``{"error": ...}`` not an exception;
unknown tool name -> error; the 10% cap flowing through as an error naming
prior/requested; and cross-user isolation (user B's ids -> error/absent, no leak).

The mutating handlers (``log_set`` / ``update_program``) open their own session via
``app.db.session.async_session_maker``; an autouse fixture points that at the NullPool
test session maker so writes stay on the same engine as the rest of the test setup
(the closed-loop bug family, see conftest.py).
"""

from __future__ import annotations

import uuid

import pytest

import app.db.session as db_session_module
from app.agent.tools.handlers import execute_tool
from app.services import program_service, workout_service
from tests.conftest import test_session_maker as session_maker
from tests.helpers import create_db_user, first_exercise_id


@pytest.fixture(autouse=True)
def _mutating_handlers_use_test_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the mutating handlers' own-session factory at the NullPool test maker."""
    monkeypatch.setattr(db_session_module, "async_session_maker", session_maker)


# --------------------------------------------------------------------------- #
# Unknown tool + non-dict input
# --------------------------------------------------------------------------- #
async def test_unknown_tool_name_returns_error() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        result = await execute_tool("not_a_tool", {}, user_id, db)
    assert "error" in result
    assert "unknown tool" in result["error"]


async def test_non_dict_input_returns_error() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        result = await execute_tool("get_workout_history", ["bad"], user_id, db)  # type: ignore[arg-type]
    assert "error" in result


# --------------------------------------------------------------------------- #
# log_set — injected user_id, success, malformed input
# --------------------------------------------------------------------------- #
async def test_log_set_uses_injected_user_id_and_succeeds() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        result = await execute_tool(
            "log_set",
            {"exercise_id": str(ex), "weight": 135.0, "reps": 5, "rir": 2.0},
            user_id,
            db,
        )
    assert result["success"] is True
    assert "set_id" in result

    # The set was written for the injected user (not anything from tool_input).
    async with session_maker() as db:
        history = await workout_service.get_history(db, user_id)
    assert len(history) == 1
    entry, _name = history[0]
    assert entry.user_id == user_id
    assert entry.weight == 135.0


async def test_log_set_bad_uuid_returns_error_not_exception() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        result = await execute_tool(
            "log_set",
            {"exercise_id": "not-a-uuid", "weight": 100, "reps": 5},
            user_id,
            db,
        )
    assert "error" in result
    assert "invalid input" in result["error"]


async def test_log_set_missing_required_field_returns_error() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        result = await execute_tool("log_set", {"exercise_id": str(ex), "weight": 100}, user_id, db)
    assert "error" in result
    assert "reps" in result["error"]


async def test_log_set_unknown_exercise_returns_structured_error() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        result = await execute_tool(
            "log_set",
            {"exercise_id": str(uuid.uuid4()), "weight": 100, "reps": 5},
            user_id,
            db,
        )
    assert "error" in result
    assert "not found" in result["error"].lower()


# --------------------------------------------------------------------------- #
# get_workout_history
# --------------------------------------------------------------------------- #
async def test_get_workout_history_returns_injected_users_sets() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        await execute_tool(
            "log_set", {"exercise_id": str(ex), "weight": 100, "reps": 5}, user_id, db
        )
        result = await execute_tool("get_workout_history", {}, user_id, db)
    assert result["count"] == 1
    assert result["sets"][0]["weight"] == 100


# --------------------------------------------------------------------------- #
# search_exercises + analyze_progression malformed input
# --------------------------------------------------------------------------- #
async def test_search_exercises_returns_matches() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        result = await execute_tool("search_exercises", {"query": "bench"}, user_id, db)
    assert "matches" in result
    assert isinstance(result["matches"], list)


async def test_search_exercises_empty_query_returns_error() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        result = await execute_tool("search_exercises", {"query": "  "}, user_id, db)
    assert "error" in result


async def test_analyze_progression_bad_uuid_returns_error() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        result = await execute_tool("analyze_progression", {"exercise_id": "nope"}, user_id, db)
    assert "error" in result


# --------------------------------------------------------------------------- #
# update_program — create, update, 10% cap, injected user
# --------------------------------------------------------------------------- #
async def test_update_program_creates_when_absent() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        result = await execute_tool(
            "update_program",
            {
                "program_name": "Push Day",
                "exercises": [{"exercise_id": str(ex), "target_weight": 100.0}],
            },
            user_id,
            db,
        )
    assert result["success"] is True
    assert result["created"] is True
    assert result["name"] == "Push Day"


async def test_update_program_updates_existing_by_name() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        await execute_tool(
            "update_program",
            {
                "program_name": "Legs",
                "exercises": [{"exercise_id": str(ex), "target_weight": 100.0}],
            },
            user_id,
            db,
        )
        # Within the 10% cap (100 -> 108).
        result = await execute_tool(
            "update_program",
            {
                "program_name": "Legs",
                "exercises": [{"exercise_id": str(ex), "target_weight": 108.0}],
            },
            user_id,
            db,
        )
    assert result["success"] is True
    assert result["created"] is False
    assert result["exercises"][0]["target_weight"] == 108.0


async def test_update_program_over_cap_rejected_with_prior_and_requested() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        await execute_tool(
            "update_program",
            {
                "program_name": "Cap",
                "exercises": [{"exercise_id": str(ex), "target_weight": 100.0}],
            },
            user_id,
            db,
        )
        # 100 -> 130 is a 30% jump, over the 10% cap.
        result = await execute_tool(
            "update_program",
            {
                "program_name": "Cap",
                "exercises": [{"exercise_id": str(ex), "target_weight": 130.0}],
            },
            user_id,
            db,
        )
    assert "error" in result
    assert "100" in result["error"]  # prior
    assert "130" in result["error"]  # requested


async def test_update_program_bad_exercise_id_returns_error() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        result = await execute_tool(
            "update_program",
            {"program_name": "Bad", "exercises": [{"exercise_id": "not-a-uuid"}]},
            user_id,
            db,
        )
    assert "error" in result


# --------------------------------------------------------------------------- #
# Cross-user isolation — user B's program id must never resolve for user A
# --------------------------------------------------------------------------- #
async def test_cross_user_program_not_visible() -> None:
    async with session_maker() as db:
        user_a = await create_db_user(db)
        user_b = await create_db_user(db)
        ex = await first_exercise_id(db)
        # A creates a program.
        await execute_tool(
            "update_program",
            {
                "program_name": "A Only",
                "exercises": [{"exercise_id": str(ex), "target_weight": 100.0}],
            },
            user_a,
            db,
        )

    # B's get_program sees nothing of A's (application-level access control).
    async with session_maker() as db:
        b_view = await execute_tool("get_program", {}, user_b, db)
    assert b_view["count"] == 0

    # B naming A's program by name still creates B's own — never touches A's.
    async with session_maker() as db:
        b_create = await execute_tool(
            "update_program",
            {
                "program_name": "A Only",
                "exercises": [{"exercise_id": str(ex), "target_weight": 50.0}],
            },
            user_b,
            db,
        )
    assert b_create["created"] is True

    # A still has exactly one "A Only" program, unchanged at 100.
    async with session_maker() as db:
        a_progs = await program_service.list_programs(db, user_a)
        assert len(a_progs) == 1
        _, a_detail = await program_service.get_program_detail(db, user_a, a_progs[0].id)
    assert a_detail[0][0].target_weight == 100.0
