"""Tool-correctness scenario definitions + DB setup applier.

The six recorded agent-turn scenarios are declared here as data so the two code
paths that need them stay in lock-step:

- ``tests.fixtures.record_fixtures`` runs each scenario's ``message`` through the
  REAL ``run_agent_turn`` (live Sonnet) after applying its ``setup``, capturing
  the streamed assistant turns into a committed fixture.
- ``tests.test_tool_correctness`` replays that fixture through the real
  orchestrator + :class:`FakeAnthropicClient`, re-applying the SAME ``setup`` so
  the recorded tool inputs (exercise ids rewritten via ``id_map``) resolve and the
  case-specific assertions (10% cap rejection, resulting ``set_entries`` rows, tool
  choice) hold.

``setup`` is *authored* deterministic DB state, not paid-for model output, so it
lives here (and is copied into the fixture for the replay to reproduce) rather than
being recorded. ``mock_script`` is a hand-written stand-in for the model's streamed
turns used only by the recorder's ``--mock`` plumbing check (zero API spend); the
live pass overwrites it with the real capture.

Exercise names below MUST match ``seed.seed_exercises.EXERCISES`` exactly — the
setup applier and the ``id_map`` both resolve names against the seeded catalog.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.exercise import Exercise
from app.services import program_service
from tests.helpers import add_session_with_sets

SQUAT = "Barbell Back Squat"
DEADLIFT = "Conventional Deadlift"
OHP = "Overhead Press"
PULLDOWN = "Lat Pulldown"


# --------------------------------------------------------------------------- #
# Scenario data (6 cases: 4 representatives + 2 more)
# --------------------------------------------------------------------------- #
# Each scenario:
#   id            fixture basename (claude_responses/tc_<id>.json)
#   message       the user turn
#   setup         declarative DB state to create for the throwaway/test user
#   must_call     tools the turn is expected to invoke (subset assertion)
#   must_not_call tools the turn must NOT invoke
#   checks        case-specific replay assertions (cap rejection / logged sets)
#   mock_script   hand-written streamed turns for the --mock plumbing check only
#
# Exercise choice note: cases that chain off search_exercises use names with a
# DOMINANT single fuzzy match (conventional deadlift / overhead press / lat
# pulldown). A casual ambiguous name like "squat" fuzzy-ties across 4+ variants,
# and the coach's system prompt then (correctly) asks the user to disambiguate
# instead of proceeding — good agent behaviour, but not a clean two-tool flow to
# assert on. The cap case keeps "squat" because it resolves via get_program (which
# returns exact ids), so the fuzzy tie never arises there.
SCENARIOS: list[dict] = [
    {
        "id": "01_deadlift_history",
        "message": "Show me my logged sets for the conventional deadlift.",
        "setup": {
            "sessions": [
                {"exercise_name": DEADLIFT, "days_ago": 7, "sets": [[315, 5, 2], [315, 5, 1]]},
                {"exercise_name": DEADLIFT, "days_ago": 2, "sets": [[325, 5, 2]]},
            ]
        },
        "must_call": ["search_exercises", "get_workout_history"],
        "must_not_call": ["search_knowledge_base"],
        "checks": {},
        "mock_script": [
            [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "search_exercises",
                    "input": {"query": "conventional deadlift"},
                }
            ],
            [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "get_workout_history",
                    "input": {"exercise_id": "<Conventional Deadlift>"},
                }
            ],
            [{"type": "text", "text": "Here's your recent deadlift work."}],
        ],
    },
    {
        "id": "02_glutes_rag",
        "message": "Is squatting good for building glutes?",
        "setup": {},
        "must_call": ["search_knowledge_base"],
        "must_not_call": ["get_workout_history", "analyze_progression"],
        "checks": {},
        "mock_script": [
            [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "search_knowledge_base",
                    "input": {"query": "does squatting build glutes"},
                }
            ],
            [{"type": "text", "text": "Squats do train the glutes, especially deeper."}],
        ],
    },
    {
        # This case is meant to prove the TOOL's 10% cap rejects the write. A
        # blatant "+50 lbs" (22%) makes the coach persona self-censor and ask the
        # user to reconsider *before* calling update_program, so the tool guard
        # never runs. A routine-looking 200->225 (12.5%) request the coach will
        # actually execute is what lets the hard cap fire (LoadJumpCapError ->
        # structured-error tool_result) — the behaviour under test.
        "id": "03_squat_cap",
        "message": "Bump my squat up to 225 pounds in my program.",
        "setup": {
            "programs": [
                {
                    "name": "My Program",
                    "exercises": [
                        {
                            "exercise_name": SQUAT,
                            "target_sets": 3,
                            "target_reps": 5,
                            "target_rir": 2,
                            "target_weight": 200,
                        }
                    ],
                }
            ]
        },
        "must_call": ["update_program"],
        "must_not_call": [],
        "checks": {"cap_rejected": True},
        "mock_script": [
            [{"type": "tool_use", "id": "t1", "name": "get_program", "input": {}}],
            [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "update_program",
                    "input": {
                        "program_name": "My Program",
                        "exercises": [
                            {"exercise_id": "<Barbell Back Squat>", "target_weight": 225}
                        ],
                    },
                }
            ],
            [
                {
                    "type": "text",
                    "text": "That jump is over the 10% safety cap, so I can't apply it.",
                }
            ],
        ],
    },
    {
        "id": "04_log_ohp",
        "message": "Log 3 sets of overhead press at 95 pounds for 5 reps.",
        "setup": {},
        "must_call": ["search_exercises", "log_set"],
        "must_not_call": [],
        "checks": {"logged_sets": {"exercise_name": OHP, "weight": 95, "reps": 5, "count": 3}},
        "mock_script": [
            [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "search_exercises",
                    "input": {"query": "overhead press"},
                }
            ],
            [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "log_set",
                    "input": {"exercise_id": "<Overhead Press>", "weight": 95, "reps": 5},
                },
                {
                    "type": "tool_use",
                    "id": "t3",
                    "name": "log_set",
                    "input": {"exercise_id": "<Overhead Press>", "weight": 95, "reps": 5},
                },
                {
                    "type": "tool_use",
                    "id": "t4",
                    "name": "log_set",
                    "input": {"exercise_id": "<Overhead Press>", "weight": 95, "reps": 5},
                },
            ],
            [{"type": "text", "text": "Logged all three overhead press sets."}],
        ],
    },
    {
        "id": "05_pulldown_progression",
        "message": "How is my lat pulldown progressing?",
        "setup": {
            "sessions": [
                {"exercise_name": PULLDOWN, "days_ago": 21, "sets": [[120, 10, 2], [120, 10, 2]]},
                {"exercise_name": PULLDOWN, "days_ago": 14, "sets": [[130, 10, 2], [130, 9, 1]]},
                {"exercise_name": PULLDOWN, "days_ago": 7, "sets": [[140, 10, 2], [140, 9, 1]]},
            ]
        },
        "must_call": ["search_exercises", "analyze_progression"],
        "must_not_call": [],
        "checks": {},
        "mock_script": [
            [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "search_exercises",
                    "input": {"query": "lat pulldown"},
                }
            ],
            [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "analyze_progression",
                    "input": {"exercise_id": "<Lat Pulldown>"},
                }
            ],
            [{"type": "text", "text": "Your estimated 1RM is trending up."}],
        ],
    },
    {
        "id": "06_get_program",
        "message": "What programs do I have saved?",
        "setup": {
            "programs": [
                {
                    "name": "Upper Lower",
                    "exercises": [
                        {
                            "exercise_name": OHP,
                            "target_sets": 3,
                            "target_reps": 8,
                            "target_weight": 95,
                        }
                    ],
                }
            ]
        },
        "must_call": ["get_program"],
        "must_not_call": ["search_knowledge_base"],
        "checks": {},
        "mock_script": [
            [{"type": "tool_use", "id": "t1", "name": "get_program", "input": {}}],
            [{"type": "text", "text": "You have one saved program: Upper Lower."}],
        ],
    },
]


async def name_to_id_map(db: AsyncSession) -> dict[str, str]:
    """Return ``{exercise_name: str(id)}`` for the whole seeded catalog."""
    rows = (await db.execute(select(Exercise.name, Exercise.id))).all()
    return {name: str(exercise_id) for (name, exercise_id) in rows}


async def apply_setup(db: AsyncSession, user_id: uuid.UUID, setup: dict) -> None:
    """Create the scenario's declarative DB state for ``user_id`` (programs + sessions).

    Exercise names are resolved against the seeded catalog in ``db``. Programs are
    created through :mod:`program_service` (create path — no load-jump cap applies),
    so a scenario can establish a prior ``target_weight`` that a later over-cap
    ``update_program`` must be rejected against. Sessions/sets are created through the
    shared ``add_session_with_sets`` helper.
    """
    names = await name_to_id_map(db)

    for program in setup.get("programs", []):
        exercises = [
            {
                "exercise_id": names[ex["exercise_name"]],
                "target_sets": ex.get("target_sets"),
                "target_reps": ex.get("target_reps"),
                "target_rir": ex.get("target_rir"),
                "target_weight": ex.get("target_weight"),
            }
            for ex in program["exercises"]
        ]
        await program_service.create_program(db, user_id, name=program["name"], exercises=exercises)

    for session in setup.get("sessions", []):
        await add_session_with_sets(
            db,
            user_id,
            uuid.UUID(names[session["exercise_name"]]),
            [tuple(s) for s in session["sets"]],
            days_ago=session.get("days_ago", 0),
        )
