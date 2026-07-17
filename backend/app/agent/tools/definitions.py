"""Tool schemas sent to Claude.

The seven tools available to the agent loop: the six non-RAG tools plus the RAG
tool ``search_knowledge_base`` (added in Phase 5).

Invariant (CLAUDE.md rule 2): **no ``user_id`` — or any user-scoping
field — appears in any schema below.** There is nothing for the model to set even
under adversarial prompting; the orchestrator injects the verified ``user_id`` into
every handler at execution time, from the JWT-resolved session. Do not add a user
field here "to be validated later" — its structural absence is the defense.
(``search_knowledge_base`` is the trivial case: the knowledge base is global/unscoped
non-user data, so it carries only a ``query`` — no user field to omit.)
"""

from __future__ import annotations

GET_WORKOUT_HISTORY_TOOL = {
    "name": "get_workout_history",
    "description": (
        "Retrieve the current user's logged sets, optionally filtered by exercise "
        "and/or date range. Use for questions about what the user has actually done, "
        "not what they should do."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "exercise_id": {
                "type": "string",
                "description": (
                    "Filter to a specific exercise. Resolve casual names via "
                    "search_exercises first."
                ),
            },
            "start_date": {"type": "string", "description": "ISO date, inclusive"},
            "end_date": {"type": "string", "description": "ISO date, inclusive"},
            "limit": {"type": "integer", "description": "Max sets to return, default 50"},
        },
    },
}

LOG_SET_TOOL = {
    "name": "log_set",
    "description": "Log a single completed set for the current user. Call once per set.",
    "input_schema": {
        "type": "object",
        "properties": {
            "exercise_id": {"type": "string"},
            "weight": {"type": "number"},
            "reps": {"type": "integer"},
            "rir": {"type": "number", "description": "0-4+, optional"},
        },
        "required": ["exercise_id", "weight", "reps"],
    },
}

GET_PROGRAM_TOOL = {
    "name": "get_program",
    "description": (
        "Retrieve the current user's saved programs (workout templates or planned "
        "structure), optionally filtered by name."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    },
}

UPDATE_PROGRAM_TOOL = {
    "name": "update_program",
    "description": (
        "Create a new program or modify an existing one's target sets/reps/RIR/weight "
        "for the current user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "program_name": {"type": "string"},
            "exercises": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "exercise_id": {"type": "string"},
                        "target_sets": {"type": "integer"},
                        "target_reps": {"type": "integer"},
                        "target_rir": {"type": "number"},
                        "target_weight": {"type": "number"},
                    },
                    "required": ["exercise_id"],
                },
            },
        },
        "required": ["program_name", "exercises"],
    },
}

SEARCH_EXERCISES_TOOL = {
    "name": "search_exercises",
    "description": (
        "Fuzzy-search the exercise catalog by name to resolve a casual or ambiguous "
        "name to an exact exercise_id. If multiple plausible matches come back, ask "
        "the user which one they mean rather than guessing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

ANALYZE_PROGRESSION_TOOL = {
    "name": "analyze_progression",
    "description": (
        "Compute progression metrics for a specific exercise from the user's history: "
        "estimated 1RM trend, RIR trend, and plateau detection. Use this instead of "
        "trying to interpret raw set data yourself — the numbers here are exact; your "
        "job is to explain what they mean."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "exercise_id": {"type": "string"},
            "lookback_sessions": {"type": "integer", "description": "default 10"},
        },
        "required": ["exercise_id"],
    },
}

SEARCH_KNOWLEDGE_BASE_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "Search the coach's curated knowledge base (training principles, nutrition, "
        "injury-prevention information) for a grounded answer. Use for questions "
        "about what's generally advisable or true — not for questions about the "
        "user's own logged data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

# The full tool list handed to Claude every turn.
ALL_TOOL_DEFINITIONS = [
    GET_WORKOUT_HISTORY_TOOL,
    LOG_SET_TOOL,
    GET_PROGRAM_TOOL,
    UPDATE_PROGRAM_TOOL,
    SEARCH_EXERCISES_TOOL,
    ANALYZE_PROGRESSION_TOOL,
    SEARCH_KNOWLEDGE_BASE_TOOL,
]
