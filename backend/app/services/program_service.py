"""Program (workout template / plan) service.

Single source of truth shared by the REST endpoints (Phase 3) and the
``get_program`` / ``update_program`` agent tools (Phase 4). Every function takes
``user_id`` explicitly and filters on it (application-level access control): a
caller-supplied ``program_id`` is never trusted without also matching
``user_id``, so another user's program resolves to :class:`NotFoundError` (404).

The 10% load-jump hard cap is enforced here, in code, on the update path
— not requested via prompt and not in a Pydantic schema — because it must hold
identically for the REST ``PUT /programs/{id}`` and the Phase-4 ``update_program``
tool. The check runs across ALL exercises in the update *before* any write, so a
single over-cap exercise rejects the entire update with no partial change.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.exercise import Exercise
from app.models.program import Program, ProgramExercise
from app.services.errors import LoadJumpCapError, NotFoundError

# Any single update raising a target_weight above prior * this factor is rejected.
# 10% is a starting value, not a rigorously derived one — tunable.
LOAD_JUMP_CAP_FACTOR = 1.10


async def _get_owned_program(
    db: AsyncSession, user_id: uuid.UUID, program_id: uuid.UUID
) -> Program | None:
    """Fetch a program only if it belongs to ``user_id`` (application-level access control)."""
    result = await db.execute(
        select(Program).where(Program.id == program_id, Program.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def _load_program_exercises(
    db: AsyncSession, program_id: uuid.UUID
) -> list[tuple[ProgramExercise, str]]:
    """Return a program's exercises joined to names, ordered by ``order_index``."""
    result = await db.execute(
        select(ProgramExercise, Exercise.name)
        .join(Exercise, ProgramExercise.exercise_id == Exercise.id)
        .where(ProgramExercise.program_id == program_id)
        .order_by(ProgramExercise.order_index)
    )
    return [(row[0], row[1]) for row in result.all()]


async def list_programs(db: AsyncSession, user_id: uuid.UUID) -> list[Program]:
    """Return the user's programs, most-recent-first (application-level access control)."""
    result = await db.execute(
        select(Program).where(Program.user_id == user_id).order_by(Program.created_at.desc())
    )
    return list(result.scalars().all())


async def get_program_detail(
    db: AsyncSession, user_id: uuid.UUID, program_id: uuid.UUID
) -> tuple[Program, list[tuple[ProgramExercise, str]]]:
    """Return ``(program, [(program_exercise, exercise_name), ...])`` for the user.

    Exercises are ordered by ``order_index``. Raises :class:`NotFoundError` (404)
    if the program does not exist or is not the user's (application-level access
    control — ownership is never leaked).
    """
    program = await _get_owned_program(db, user_id, program_id)
    if program is None:
        raise NotFoundError("program not found")
    exercises = await _load_program_exercises(db, program.id)
    return program, exercises


async def create_program(
    db: AsyncSession,
    user_id: uuid.UUID,
    name: str,
    exercises: list[dict],
) -> tuple[Program, list[tuple[ProgramExercise, str]]]:
    """Create a program and its exercise rows for the user.

    ``exercises`` is a list of dicts with ``exercise_id`` (required) and optional
    ``target_sets`` / ``target_reps`` / ``target_rir`` / ``target_weight``;
    ``order_index`` defaults to list position when absent. No load-jump cap
    applies on create — there is no prior target to compare against. Each
    ``exercise_id`` is verified to exist (:class:`NotFoundError` otherwise).
    """
    program = Program(user_id=user_id, name=name)
    db.add(program)
    await db.flush()  # need program.id for the child rows, commit together below

    rows = await _build_program_exercise_rows(db, program.id, exercises)
    db.add_all(rows)
    await db.commit()
    await db.refresh(program)

    detail = await _load_program_exercises(db, program.id)
    return program, detail


async def update_program(
    db: AsyncSession,
    user_id: uuid.UUID,
    program_id: uuid.UUID,
    name: str | None,
    exercises: list[dict],
) -> tuple[Program, list[tuple[ProgramExercise, str]]]:
    """Replace a program's exercise set, enforcing the 10% load-jump cap.

    The program must belong to the user (:class:`NotFoundError` -> 404 otherwise).
    Before writing anything, every requested ``target_weight`` is compared against
    the prior target for that same exercise in this program; if any single value
    exceeds ``prior * 1.10`` the ENTIRE update is rejected with
    :class:`LoadJumpCapError` (-> 422) naming the exercise / prior / requested
    values — no partial write. Only then are the old exercise rows replaced.
    """
    program = await _get_owned_program(db, user_id, program_id)
    if program is None:
        raise NotFoundError("program not found")

    # Prior target_weight per exercise, for the cap comparison.
    prior_rows = await db.execute(
        select(ProgramExercise.exercise_id, ProgramExercise.target_weight).where(
            ProgramExercise.program_id == program.id
        )
    )
    prior_weights: dict[uuid.UUID, float | None] = {
        exercise_id: target_weight for (exercise_id, target_weight) in prior_rows.all()
    }

    # Cap check across ALL exercises first — reject the whole update on any breach,
    # so nothing is written when even one exercise is over the cap.
    for ex in exercises:
        new_weight = ex.get("target_weight")
        if new_weight is None:
            continue
        prior = prior_weights.get(_as_uuid(ex["exercise_id"]))
        if prior is not None and new_weight > prior * LOAD_JUMP_CAP_FACTOR:
            raise LoadJumpCapError(
                exercise_id=str(ex["exercise_id"]),
                prior=prior,
                requested=new_weight,
            )

    # Validate all referenced exercises exist before mutating (also no partial write).
    new_rows = await _build_program_exercise_rows(db, program.id, exercises)

    if name is not None:
        program.name = name

    # Replace the exercise set wholesale ("modify ... target ...").
    existing = (
        (await db.execute(select(ProgramExercise).where(ProgramExercise.program_id == program.id)))
        .scalars()
        .all()
    )
    for row in existing:
        await db.delete(row)
    await db.flush()

    db.add_all(new_rows)
    await db.commit()
    await db.refresh(program)

    detail = await _load_program_exercises(db, program.id)
    return program, detail


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    """Coerce a caller/LLM-supplied exercise id to a UUID (str on the tool path)."""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


async def _build_program_exercise_rows(
    db: AsyncSession, program_id: uuid.UUID, exercises: list[dict]
) -> list[ProgramExercise]:
    """Build (unpersisted) ProgramExercise rows, verifying each exercise exists.

    ``order_index`` falls back to list position when not supplied. Raises
    :class:`NotFoundError` if any ``exercise_id`` is unknown, before the caller
    writes anything — keeping create/update all-or-nothing.
    """
    rows: list[ProgramExercise] = []
    for position, ex in enumerate(exercises):
        exercise_id = _as_uuid(ex["exercise_id"])
        exists = (
            await db.execute(select(Exercise.id).where(Exercise.id == exercise_id))
        ).scalar_one_or_none()
        if exists is None:
            raise NotFoundError(f"exercise_id {exercise_id} not found")
        # order_index falls back to list position when absent OR explicitly None
        # (the REST schema passes None through model_dump when the client omits it).
        order_index = ex.get("order_index")
        if order_index is None:
            order_index = position
        rows.append(
            ProgramExercise(
                program_id=program_id,
                exercise_id=exercise_id,
                order_index=order_index,
                target_sets=ex.get("target_sets"),
                target_reps=ex.get("target_reps"),
                target_rir=ex.get("target_rir"),
                target_weight=ex.get("target_weight"),
            )
        )
    return rows
