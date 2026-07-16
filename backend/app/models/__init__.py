from __future__ import annotations

# Import every model so that Base.metadata is fully populated. Alembic
# autogenerate compares Base.metadata against the live DB, so a model that is
# never imported here would be silently omitted from migrations.
from app.db.base import Base
from app.models.exercise import Exercise
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.memory_observation import MemoryObservation
from app.models.program import Program, ProgramExercise
from app.models.set_entry import SetEntry
from app.models.user import User
from app.models.user_memory import UserMemory
from app.models.workout_session import WorkoutSession

__all__ = [
    "Base",
    "Exercise",
    "KnowledgeChunk",
    "MemoryObservation",
    "Program",
    "ProgramExercise",
    "SetEntry",
    "User",
    "UserMemory",
    "WorkoutSession",
]
