"""Tool execution — the trusted server side of every agent tool call (spec §8).

``execute_tool`` is the single dispatch point the orchestrator calls. Contract
(CLAUDE.md rules 2 & 4):

- ``current_user_id`` is injected here from the JWT-verified session and passed to
  every service call. It is NEVER read from ``tool_input`` — the LLM-supplied input
  is untrusted and only ever supplies non-identity arguments (exercise ids, weights,
  names). This is the structural prompt-injection defense (§6.5).
- Every handler returns a plain dict — success payload or ``{"error": "..."}`` — and
  never raises. Domain failures (``ServiceError``) become ``{"error": str(exc)}``;
  anything unexpected is logged and rendered as a generic error, because an unhandled
  exception here would kill the whole SSE stream, not just one tool call (§7.1).

**DB transaction isolation (§7.1 required decision):** the read-only tools reuse the
request-scoped ``db`` handed in by the orchestrator. The *mutating* tools (``log_set``,
``update_program``) each open their **own** short-lived ``AsyncSession`` from
``async_session_maker`` and commit/roll back independently. This is the "own session
per mutating call" option (not ``begin_nested``): the Phase-3 services already call
``db.commit()`` internally, which is incompatible with a caller-owned SAVEPOINT, and a
fresh session guarantees that one tool call failing mid-turn cannot poison the shared
session for the other tool calls in the same 8-iteration loop.
"""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db import session as db_session_module
from app.services import (
    exercise_service,
    program_service,
    progression_service,
    workout_service,
)
from app.services.errors import NotFoundError, ServiceError

log = get_logger(__name__)


def _new_session() -> AsyncSession:
    """Open a fresh AsyncSession for a mutating tool call (§7.1 isolation decision).

    Resolved through the module at call time (not imported once) so tests can point
    the mutating handlers at the NullPool test session maker, keeping DB writes on the
    same engine as the rest of the test setup (LESSONS.md closed-loop family).
    """
    return db_session_module.async_session_maker()


def _coerce_uuid(value: object) -> uuid.UUID:
    """Coerce untrusted LLM input to a UUID, raising ValueError on anything malformed.

    Handlers convert the resulting ValueError into ``{"error": ...}`` — a bad
    ``exercise_id`` from the model must be a structured error, never a 500.
    """
    return uuid.UUID(str(value))


def _parse_iso_date(value: object) -> date | None:
    """Parse an untrusted ISO date string, or None. Raises ValueError if malformed."""
    if value is None:
        return None
    return date.fromisoformat(str(value))


# --------------------------------------------------------------------------- #
# Read-only handlers — reuse the request-scoped session.
# --------------------------------------------------------------------------- #
async def handle_get_workout_history(
    tool_input: dict, current_user_id: uuid.UUID, db: AsyncSession
) -> dict:
    try:
        exercise_id = (
            _coerce_uuid(tool_input["exercise_id"]) if tool_input.get("exercise_id") else None
        )
        start_date = _parse_iso_date(tool_input.get("start_date"))
        end_date = _parse_iso_date(tool_input.get("end_date"))
        limit = tool_input.get("limit")
        limit = int(limit) if limit is not None else workout_service.DEFAULT_HISTORY_LIMIT

        rows = await workout_service.get_history(
            db,
            current_user_id,
            exercise_id=exercise_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    except (ValueError, TypeError) as exc:
        return {"error": f"invalid input: {exc}"}
    except ServiceError as exc:
        return {"error": str(exc)}
    except Exception:  # noqa: BLE001 — last-resort guard; must never raise into the stream
        log.exception("tool_unhandled_error", extra={"tool_name": "get_workout_history"})
        return {"error": "internal error executing get_workout_history"}

    return {
        "sets": [
            {
                "set_id": str(entry.id),
                "exercise_id": str(entry.exercise_id),
                "exercise_name": name,
                "set_number": entry.set_number,
                "weight": entry.weight,
                "reps": entry.reps,
                "rir": entry.rir,
                "created_at": entry.created_at.isoformat(),
            }
            for (entry, name) in rows
        ],
        "count": len(rows),
    }


async def handle_get_program(
    tool_input: dict, current_user_id: uuid.UUID, db: AsyncSession
) -> dict:
    try:
        name_filter = tool_input.get("name")
        programs = await program_service.list_programs(db, current_user_id)
        if name_filter:
            needle = str(name_filter).strip().lower()
            programs = [p for p in programs if p.name.lower() == needle]

        result = []
        for program in programs:
            _, exercises = await program_service.get_program_detail(db, current_user_id, program.id)
            result.append(
                {
                    "program_id": str(program.id),
                    "name": program.name,
                    "exercises": [
                        {
                            "exercise_id": str(pe.exercise_id),
                            "exercise_name": ex_name,
                            "target_sets": pe.target_sets,
                            "target_reps": pe.target_reps,
                            "target_rir": pe.target_rir,
                            "target_weight": pe.target_weight,
                        }
                        for (pe, ex_name) in exercises
                    ],
                }
            )
    except ServiceError as exc:
        return {"error": str(exc)}
    except Exception:  # noqa: BLE001
        log.exception("tool_unhandled_error", extra={"tool_name": "get_program"})
        return {"error": "internal error executing get_program"}

    return {"programs": result, "count": len(result)}


async def handle_search_exercises(
    tool_input: dict, current_user_id: uuid.UUID, db: AsyncSession
) -> dict:
    # No user data here — the catalog is global/read-only (§8.5). current_user_id is
    # accepted for a uniform handler signature but unused.
    try:
        query = tool_input.get("query")
        if not query or not str(query).strip():
            return {"error": "query is required"}
        matches = await exercise_service.search_exercises(db, str(query))
    except ServiceError as exc:
        return {"error": str(exc)}
    except Exception:  # noqa: BLE001
        log.exception("tool_unhandled_error", extra={"tool_name": "search_exercises"})
        return {"error": "internal error executing search_exercises"}
    return {"matches": matches}


async def handle_analyze_progression(
    tool_input: dict, current_user_id: uuid.UUID, db: AsyncSession
) -> dict:
    try:
        exercise_id = _coerce_uuid(tool_input["exercise_id"])
        lookback = tool_input.get("lookback_sessions")
        lookback = (
            int(lookback) if lookback is not None else progression_service.DEFAULT_LOOKBACK_SESSIONS
        )
        return await progression_service.analyze(
            db, current_user_id, exercise_id, lookback_sessions=lookback
        )
    except KeyError:
        return {"error": "exercise_id is required"}
    except (ValueError, TypeError) as exc:
        return {"error": f"invalid input: {exc}"}
    except ServiceError as exc:
        return {"error": str(exc)}
    except Exception:  # noqa: BLE001
        log.exception("tool_unhandled_error", extra={"tool_name": "analyze_progression"})
        return {"error": "internal error executing analyze_progression"}


# --------------------------------------------------------------------------- #
# Mutating handlers — own session per call (see module docstring).
# --------------------------------------------------------------------------- #
async def handle_log_set(tool_input: dict, current_user_id: uuid.UUID, db: AsyncSession) -> dict:
    try:
        exercise_id = _coerce_uuid(tool_input["exercise_id"])
        weight = float(tool_input["weight"])
        reps = int(tool_input["reps"])
        rir = tool_input.get("rir")
        rir = float(rir) if rir is not None else None
    except KeyError as exc:
        return {"error": f"missing required field: {exc.args[0]}"}
    except (ValueError, TypeError) as exc:
        return {"error": f"invalid input: {exc}"}

    try:
        async with _new_session() as own_db:
            entry = await workout_service.log_set(
                own_db,
                user_id=current_user_id,
                exercise_id=exercise_id,
                weight=weight,
                reps=reps,
                rir=rir,
            )
            return {
                "success": True,
                "set_id": str(entry.id),
                "session_id": str(entry.session_id),
                "set_number": entry.set_number,
            }
    except ServiceError as exc:
        return {"error": str(exc)}
    except Exception:  # noqa: BLE001
        log.exception("tool_unhandled_error", extra={"tool_name": "log_set"})
        return {"error": "internal error executing log_set"}


async def handle_update_program(
    tool_input: dict, current_user_id: uuid.UUID, db: AsyncSession
) -> dict:
    """Create-or-update a program by name (spec §8.4).

    Finds the user's program by ``program_name`` (case-insensitive exact match);
    creates it if absent, otherwise replaces its exercise set via the existing service.
    The 10% load-jump cap flows through unchanged from ``program_service.update_program``
    as ``{"error": ...}`` (``LoadJumpCapError`` -> ``ServiceError`` catch below).
    """
    try:
        program_name = tool_input["program_name"]
        exercises = tool_input["exercises"]
    except KeyError as exc:
        return {"error": f"missing required field: {exc.args[0]}"}
    if not isinstance(exercises, list):
        return {"error": "exercises must be a list"}

    # Validate/coerce every exercise_id up front so a malformed id is a structured
    # error, not a 500 deep in the service.
    try:
        for ex in exercises:
            if not isinstance(ex, dict) or "exercise_id" not in ex:
                return {"error": "each exercise requires an exercise_id"}
            _coerce_uuid(ex["exercise_id"])
    except (ValueError, TypeError) as exc:
        return {"error": f"invalid input: {exc}"}

    try:
        async with _new_session() as own_db:
            needle = str(program_name).strip().lower()
            existing = [
                p
                for p in await program_service.list_programs(own_db, current_user_id)
                if p.name.lower() == needle
            ]

            if existing:
                program, detail = await program_service.update_program(
                    own_db,
                    current_user_id,
                    program_id=existing[0].id,
                    name=str(program_name),
                    exercises=exercises,
                )
                created = False
            else:
                program, detail = await program_service.create_program(
                    own_db,
                    current_user_id,
                    name=str(program_name),
                    exercises=exercises,
                )
                created = True
    except NotFoundError as exc:
        return {"error": str(exc)}
    except ServiceError as exc:  # includes LoadJumpCapError (names prior + requested)
        return {"error": str(exc)}
    except Exception:  # noqa: BLE001
        log.exception("tool_unhandled_error", extra={"tool_name": "update_program"})
        return {"error": "internal error executing update_program"}

    return {
        "success": True,
        "created": created,
        "program_id": str(program.id),
        "name": program.name,
        "exercises": [
            {
                "exercise_id": str(pe.exercise_id),
                "exercise_name": ex_name,
                "target_sets": pe.target_sets,
                "target_reps": pe.target_reps,
                "target_rir": pe.target_rir,
                "target_weight": pe.target_weight,
            }
            for (pe, ex_name) in detail
        ],
    }


_HANDLERS = {
    "get_workout_history": handle_get_workout_history,
    "log_set": handle_log_set,
    "get_program": handle_get_program,
    "update_program": handle_update_program,
    "search_exercises": handle_search_exercises,
    "analyze_progression": handle_analyze_progression,
}


async def execute_tool(
    name: str, tool_input: dict, current_user_id: uuid.UUID, db: AsyncSession
) -> dict:
    """Dispatch one tool call to its handler; always returns a dict, never raises.

    ``current_user_id`` is the orchestrator's JWT-verified user, injected into every
    handler — never sourced from ``tool_input`` (CLAUDE.md rule 2). An unknown tool
    name (the model hallucinating a tool) returns a structured error rather than
    raising.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name}"}
    # tool_input is already a parsed dict from the SDK; guard against a non-dict just
    # in case (never re-parse or string-match — see build notes).
    if not isinstance(tool_input, dict):
        return {"error": "tool input must be an object"}
    return await handler(tool_input, current_user_id, db)
