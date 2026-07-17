"""Idempotent seeding of the exercise catalog.

Run from ``backend/`` with::

    python -m seed.seed_exercises

Seeds ~50 exercises spanning the full category spread below. Idempotent: exercises
are keyed by their unique ``name``; an exercise whose name already exists is
skipped, so re-running never duplicates rows and always converges on the same
count.

Enum values, enforced here by construction:
  primary_muscle_group: chest | back | quads | hamstrings | shoulders | arms |
                        core | glutes | calves
  movement_pattern:     push | pull | hinge | squat | carry | isolation
  equipment:            barbell | dumbbell | machine | cable | bodyweight
"""

from __future__ import annotations

import asyncio

from sqlalchemy import func, select

from app.db.session import async_session_maker
from app.models.exercise import Exercise

# (name, primary_muscle_group, movement_pattern, equipment)
EXERCISES: list[tuple[str, str, str, str]] = [
    # --- Compound barbell ---
    ("Barbell Back Squat", "quads", "squat", "barbell"),
    ("Barbell Bench Press", "chest", "push", "barbell"),
    ("Conventional Deadlift", "back", "hinge", "barbell"),
    ("Overhead Press", "shoulders", "push", "barbell"),
    ("Barbell Row", "back", "pull", "barbell"),
    ("Barbell Hip Thrust", "glutes", "hinge", "barbell"),
    ("Barbell Front Rack Carry", "core", "carry", "barbell"),
    # --- Compound barbell variants ---
    ("Front Squat", "quads", "squat", "barbell"),
    ("Incline Bench Press", "chest", "push", "barbell"),
    ("Romanian Deadlift", "hamstrings", "hinge", "barbell"),
    ("Sumo Deadlift", "glutes", "hinge", "barbell"),
    ("Close-Grip Bench Press", "arms", "push", "barbell"),
    ("Pause Squat", "quads", "squat", "barbell"),
    ("Pendlay Row", "back", "pull", "barbell"),
    ("Push Press", "shoulders", "push", "barbell"),
    # --- Machine / cable compounds ---
    ("Leg Press", "quads", "squat", "machine"),
    ("Hack Squat", "quads", "squat", "machine"),
    ("Lat Pulldown", "back", "pull", "cable"),
    ("Chest Press Machine", "chest", "push", "machine"),
    ("Seated Cable Row", "back", "pull", "cable"),
    ("Assisted Pull-Up Machine", "back", "pull", "machine"),
    ("Smith Machine Squat", "quads", "squat", "machine"),
    ("Cable Fly", "chest", "push", "cable"),
    ("Cable Face Pull", "shoulders", "pull", "cable"),
    ("Pec Deck Machine", "chest", "isolation", "machine"),
    ("Farmer's Carry", "core", "carry", "dumbbell"),
    # --- Dumbbell accessories ---
    ("DB Shoulder Press", "shoulders", "push", "dumbbell"),
    ("DB Bench Press", "chest", "push", "dumbbell"),
    ("DB Incline Press", "chest", "push", "dumbbell"),
    ("DB Row", "back", "pull", "dumbbell"),
    ("DB Bicep Curl", "arms", "isolation", "dumbbell"),
    ("DB Hammer Curl", "arms", "isolation", "dumbbell"),
    ("DB Lateral Raise", "shoulders", "isolation", "dumbbell"),
    ("DB Rear Delt Fly", "shoulders", "isolation", "dumbbell"),
    ("DB Bulgarian Split Squat", "quads", "squat", "dumbbell"),
    ("DB Romanian Deadlift", "hamstrings", "hinge", "dumbbell"),
    ("DB Goblet Squat", "quads", "squat", "dumbbell"),
    ("DB Overhead Tricep Extension", "arms", "isolation", "dumbbell"),
    ("DB Walking Lunge", "glutes", "squat", "dumbbell"),
    # --- Bodyweight / isolation ---
    ("Pull-Up", "back", "pull", "bodyweight"),
    ("Chin-Up", "back", "pull", "bodyweight"),
    ("Push-Up", "chest", "push", "bodyweight"),
    ("Dip", "chest", "push", "bodyweight"),
    ("Plank", "core", "isolation", "bodyweight"),
    ("Hanging Leg Raise", "core", "isolation", "bodyweight"),
    ("Back Extension", "hamstrings", "hinge", "bodyweight"),
    ("Leg Curl", "hamstrings", "isolation", "machine"),
    ("Leg Extension", "quads", "isolation", "machine"),
    ("Tricep Pushdown", "arms", "isolation", "cable"),
    ("Cable Bicep Curl", "arms", "isolation", "cable"),
    ("Standing Calf Raise", "calves", "isolation", "machine"),
    ("Seated Calf Raise", "calves", "isolation", "machine"),
    ("Cable Crunch", "core", "isolation", "cable"),
    ("Glute Kickback Machine", "glutes", "isolation", "machine"),
]


async def seed() -> int:
    """Insert any missing exercises; return the final total exercise count."""
    async with async_session_maker() as session:
        existing_names = set((await session.execute(select(Exercise.name))).scalars().all())

        to_add = [
            Exercise(
                name=name,
                primary_muscle_group=muscle,
                movement_pattern=pattern,
                equipment=equipment,
            )
            for (name, muscle, pattern, equipment) in EXERCISES
            if name not in existing_names
        ]

        if to_add:
            session.add_all(to_add)
            await session.commit()

        total = (await session.execute(select(func.count(Exercise.id)))).scalar_one()
        print(f"Seeded {len(to_add)} new exercise(s); {total} exercises total.")
        return total


if __name__ == "__main__":
    asyncio.run(seed())
