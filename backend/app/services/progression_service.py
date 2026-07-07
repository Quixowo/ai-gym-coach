"""Progression-analysis service (spec §8.6).

The single, deterministic implementation behind the ``analyze_progression`` agent
tool. Per CLAUDE.md rule 3 the progression math (Epley 1RM, first-half/second-half
trend, RIR trend at comparable load, plateau detection) lives here in code — the
LLM is never trusted to eyeball raw set data. No LLM calls happen in this module.

Access control (spec §6.3): every query filters on ``user_id``, so another user's
``exercise_id`` (or an exercise with no history for this user) simply yields the
empty/no-data shape — never a leak.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.exercise import Exercise
from app.models.set_entry import SetEntry
from app.models.workout_session import WorkoutSession
from app.services.errors import NotFoundError

DEFAULT_LOOKBACK_SESSIONS = 10

# §8.6 tunable thresholds — kept as module constants so tests can assert boundaries.
TREND_THRESHOLD = 0.02  # >2% mean change -> increasing/decreasing, else flat
COMPARABLE_LOAD_TOLERANCE = 0.05  # ±5% of the most common top-set weight
MIN_COMPARABLE_SESSIONS = 3  # fewer than this -> RIR "insufficient_data"
PLATEAU_MIN_SESSIONS = 3  # last N identical top sets -> plateaued


def estimated_1rm(weight: float, reps: int) -> float:
    """Epley estimated 1RM: ``weight * (1 + reps / 30)`` (spec §8.6)."""
    return weight * (1 + reps / 30)


def _half_means(series: Sequence[float]) -> tuple[float, float]:
    """Return ``(first_half_mean, second_half_mean)`` for the first-half/second-half rule.

    With an odd number of elements the middle element is dropped from both halves,
    so the two halves compare cleanly without a shared midpoint. Callers only invoke
    this with ``len(series) >= 2``.
    """
    n = len(series)
    half = n // 2
    first = series[:half]
    second = series[n - half :]
    return sum(first) / len(first), sum(second) / len(second)


def _direction(
    series: Sequence[float],
    threshold: float,
    labels: tuple[str, str, str],
) -> str:
    """Apply the first-half/second-half rule, returning one of ``labels`` (up, down, flat).

    ``labels`` is ``(increasing_label, decreasing_label, flat_label)`` so the same
    rule serves both the 1RM trend and the RIR trend with their different wording.
    Needs at least 2 points; fewer collapses to the flat label.
    """
    up, down, flat = labels
    if len(series) < 2:
        return flat
    first_mean, second_mean = _half_means(series)
    if first_mean == 0:
        return flat
    change = (second_mean - first_mean) / first_mean
    if change > threshold:
        return up
    if change < -threshold:
        return down
    return flat


class _SessionTopSet:
    """The single top set (highest Epley 1RM) for one session, oldest->newest ordering."""

    __slots__ = ("weight", "reps", "rir", "e1rm")

    def __init__(self, weight: float, reps: int, rir: float | None) -> None:
        self.weight = weight
        self.reps = reps
        self.rir = rir
        self.e1rm = estimated_1rm(weight, reps)


async def analyze(
    db: AsyncSession,
    user_id: uuid.UUID,
    exercise_id: uuid.UUID,
    lookback_sessions: int = DEFAULT_LOOKBACK_SESSIONS,
) -> dict:
    """Compute the §8.6 progression metrics for one user + exercise.

    Pulls up to ``lookback_sessions`` most-recent sessions that contain at least one
    set of this exercise (sessions with none are skipped), takes each session's top
    set by Epley 1RM, then derives:

    - ``trend``: first-half vs second-half mean of the 1RM series (>2% -> increasing
      / decreasing, else flat).
    - ``rir_trend``: same rule over the RIR of sessions whose top-set weight is within
      ±5% of the most common top-set weight; ``"insufficient_data"`` if fewer than 3
      such sessions.
    - ``plateaued`` / ``plateau_session_count``: the trailing run of sessions whose
      top set has identical weight×reps×RIR, flagged when that run is >= 3.

    Returns the §8.6 dict. Raises :class:`NotFoundError` if the exercise id is unknown
    (application-level access control makes "not the user's data" indistinguishable
    from "no data": a known exercise with no logged sets returns the empty shape with
    ``sessions_analyzed == 0``).
    """
    if lookback_sessions < 1:
        lookback_sessions = DEFAULT_LOOKBACK_SESSIONS

    exercise_name = (
        await db.execute(select(Exercise.name).where(Exercise.id == exercise_id))
    ).scalar_one_or_none()
    if exercise_name is None:
        raise NotFoundError("exercise_id not found")

    # Most-recent N sessions (by session date, tie-broken by created_at) that contain
    # >= 1 set of this exercise for this user. Filtered on user_id + exercise_id
    # (application-level access control).
    session_id_rows = (
        (
            await db.execute(
                select(WorkoutSession.id)
                .join(SetEntry, SetEntry.session_id == WorkoutSession.id)
                .where(
                    WorkoutSession.user_id == user_id,
                    SetEntry.exercise_id == exercise_id,
                )
                .group_by(WorkoutSession.id, WorkoutSession.date, WorkoutSession.created_at)
                .order_by(WorkoutSession.date.desc(), WorkoutSession.created_at.desc())
                .limit(lookback_sessions)
            )
        )
        .scalars()
        .all()
    )

    if not session_id_rows:
        return _empty_shape(exercise_name)

    # Pull every set of this exercise across those sessions in one query, then reduce
    # to one top set per session in Python.
    set_rows = (
        await db.execute(
            select(
                SetEntry.session_id,
                SetEntry.weight,
                SetEntry.reps,
                SetEntry.rir,
            ).where(
                SetEntry.user_id == user_id,
                SetEntry.exercise_id == exercise_id,
                SetEntry.session_id.in_(session_id_rows),
            )
        )
    ).all()

    top_by_session: dict[uuid.UUID, _SessionTopSet] = {}
    for session_id, weight, reps, rir in set_rows:
        candidate = _SessionTopSet(weight, reps, rir)
        current = top_by_session.get(session_id)
        if current is None or candidate.e1rm > current.e1rm:
            top_by_session[session_id] = candidate

    # Order oldest -> newest. session_id_rows is newest-first, so reverse it and keep
    # only sessions that actually produced a top set (all should, by construction).
    ordered = [top_by_session[sid] for sid in reversed(session_id_rows) if sid in top_by_session]

    sessions_analyzed = len(ordered)
    e1rm_series = [round(t.e1rm, 2) for t in ordered]
    trend = _direction(e1rm_series, TREND_THRESHOLD, ("increasing", "decreasing", "flat"))
    rir_trend = _compute_rir_trend(ordered)
    plateaued, plateau_count = _detect_plateau(ordered)

    return {
        "exercise": exercise_name,
        "sessions_analyzed": sessions_analyzed,
        "estimated_1rm_series": e1rm_series,
        "trend": trend,
        "rir_trend": rir_trend,
        "plateaued": plateaued,
        "plateau_session_count": plateau_count,
    }


def _compute_rir_trend(ordered: Sequence[_SessionTopSet]) -> str:
    """RIR trend at comparable load (spec §8.6 step 4).

    Restrict to sessions whose top-set weight is within ±5% of the *most common*
    top-set weight, drop any with a missing RIR, and apply the first-half/second-half
    rule to that RIR subsequence. Fewer than 3 comparable sessions ->
    ``"insufficient_data"`` (a misleading trend is worse than none). Declining RIR at
    constant load reads as ``"improving"``.
    """
    weights = [t.weight for t in ordered]
    if not weights:
        return "insufficient_data"

    # Most common top-set weight in the window (ties -> first encountered).
    most_common_weight = Counter(weights).most_common(1)[0][0]
    low = most_common_weight * (1 - COMPARABLE_LOAD_TOLERANCE)
    high = most_common_weight * (1 + COMPARABLE_LOAD_TOLERANCE)

    comparable_rirs = [t.rir for t in ordered if low <= t.weight <= high and t.rir is not None]
    if len(comparable_rirs) < MIN_COMPARABLE_SESSIONS:
        return "insufficient_data"

    # Declining RIR at constant load = improving; rising = declining.
    return _direction(comparable_rirs, TREND_THRESHOLD, ("declining", "improving", "stable"))


def _detect_plateau(ordered: Sequence[_SessionTopSet]) -> tuple[bool, int]:
    """Plateau detection (spec §8.6 step 5): trailing run of identical top sets.

    Walks backward from the newest session counting sessions whose top set has an
    identical ``(weight, reps, rir)`` triple. ``plateaued`` is True when that run is
    at least :data:`PLATEAU_MIN_SESSIONS`. Returns ``(plateaued, run_length)``.
    """
    if not ordered:
        return False, 0

    newest = ordered[-1]
    key = (newest.weight, newest.reps, newest.rir)
    run = 0
    for t in reversed(ordered):
        if (t.weight, t.reps, t.rir) == key:
            run += 1
        else:
            break
    return run >= PLATEAU_MIN_SESSIONS, run


def _empty_shape(exercise_name: str) -> dict:
    """The no-data §8.6 shape for a known exercise the user has never logged."""
    return {
        "exercise": exercise_name,
        "sessions_analyzed": 0,
        "estimated_1rm_series": [],
        "trend": "flat",
        "rir_trend": "insufficient_data",
        "plateaued": False,
        "plateau_session_count": 0,
    }
